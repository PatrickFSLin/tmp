#!/usr/bin/env python3
"""
VaVAM-B 12j — complete Thor ego-trajectory evaluation.

NO PyTorch.

Pipeline:
    cleaned.pkl
        +
    visual-token .npy
        |
        v
    NumPy dataset
        |
        +--> visual_tokens [1,8,18,32]
        +--> command [1,1]
        +--> GT [1,6,2]
        |
        v
    TRT Prefill FP16
        |
        v
    48 GPU-resident K/V tensors
        |
        v
    TRT Action FP16 x10
        +
    CUDA Euler x10
        |
        v
    final normalized action
        |
        v
    * 70.0
        |
        v
    prediction [1,1,6,2]
        |
        v
    NumPy minADE

The TensorRT/Euler section is based on the validated 12i GPU-resident
implementation. K/V addresses are shared directly between engines; action
and diffusion timestep remain on GPU throughout the 10-step loop.
"""

import argparse
import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cuda, nvrtc

from min_ade import min_ade
from thor_ego_trajectory_dataset import ThorEgoTrajectoryDataset


ROOT = Path.home() / "vblkdev2" / "VaVAM_Thor"

DEFAULT_PREFILL = ROOT / "Engines" / (
    "vavam_joint_kv_prefill_B_v10_fp16.engine"
)
DEFAULT_ACTION = ROOT / "Engines" / (
    "vavam_joint_action_B_fp16.engine"
)
DEFAULT_PICKLE = ROOT / "data" / "nuScenes-mini" / (
    "nuscenes_mini_data_cleaned.pkl"
)
DEFAULT_TOKENS = ROOT / "data" / "nuScenes-mini" / "tokens"

NUM_LAYERS = 24
NUM_STEPS = 10
DT = 0.1
ACTION_SHAPE = (1, 1, 6, 2)
ACTION_SCALING = 70.0


def check(result, name):
    err = result[0] if isinstance(result, tuple) else result
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {err}")
    return result


def malloc(nbytes):
    return check(
        cuda.cuMemAlloc(int(nbytes)),
        "cuMemAlloc",
    )[1]


def free(ptr):
    if ptr is not None:
        check(cuda.cuMemFree(int(ptr)), "cuMemFree")


def h2d(ptr, arr):
    arr = np.ascontiguousarray(arr)
    check(
        cuda.cuMemcpyHtoD(
            int(ptr),
            arr,
            int(arr.nbytes),
        ),
        "cuMemcpyHtoD",
    )


def d2h(arr, ptr):
    arr = np.ascontiguousarray(arr)
    check(
        cuda.cuMemcpyDtoH(
            arr,
            int(ptr),
            int(arr.nbytes),
        ),
        "cuMemcpyDtoH",
    )


def stream_create():
    return check(
        cuda.cuStreamCreate(0),
        "cuStreamCreate",
    )[1]


def stream_sync(stream):
    check(
        cuda.cuStreamSynchronize(stream),
        "cuStreamSynchronize",
    )


def event_create():
    return check(
        cuda.cuEventCreate(0),
        "cuEventCreate",
    )[1]


def event_record(event, stream):
    check(
        cuda.cuEventRecord(event, stream),
        "cuEventRecord",
    )


def elapsed_ms(start, end):
    return float(
        check(
            cuda.cuEventElapsedTime(start, end),
            "cuEventElapsedTime",
        )[1]
    )


class Engine:
    def __init__(self, path, skip_names=None):
        self.path = Path(path)
        self.skip_names = set(skip_names or [])
        self.owned = {}

        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)

        with open(self.path, "rb") as f:
            blob = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize {self.path}"
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                f"Failed to create TRT context: {self.path}"
            )

        print()
        print("=" * 80)
        print(f"Loaded: {self.path.name}")
        print("=" * 80)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))

            print(
                f"{'IN ' if mode == trt.TensorIOMode.INPUT else 'OUT'} "
                f"{name:24s} dtype={dtype} shape={shape}"
            )

            if any(d < 0 for d in shape):
                raise RuntimeError(
                    f"Dynamic shape unsupported: {name} {shape}"
                )

            if name in self.skip_names:
                continue

            np_dtype = np.dtype(trt.nptype(dtype))
            nbytes = int(np.prod(shape)) * np_dtype.itemsize
            self.owned[name] = malloc(nbytes)

        for name, ptr in self.owned.items():
            self.set_address(name, ptr)

    def set_address(self, name, ptr):
        ok = self.context.set_tensor_address(
            name,
            int(ptr),
        )
        if not ok:
            raise RuntimeError(
                f"set_tensor_address failed: {name}"
            )

    def execute(self, stream):
        ok = self.context.execute_async_v3(
            stream_handle=int(stream)
        )
        if not ok:
            raise RuntimeError(
                f"execute_async_v3 failed: {self.path.name}"
            )

    def dtype(self, name):
        return np.dtype(
            trt.nptype(
                self.engine.get_tensor_dtype(name)
            )
        )

    def shape(self, name):
        return tuple(
            self.engine.get_tensor_shape(name)
        )

    def cleanup(self):
        for ptr in self.owned.values():
            free(ptr)
        self.owned.clear()


KERNEL_SRC = r"""
#include <cuda_fp16.h>

extern "C" __global__
void euler_f32_f32(float* a, const float* v, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) a[i] = a[i] + dt * v[i];
}

extern "C" __global__
void euler_f32_f16(float* a, const __half* v, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) a[i] = a[i] + dt * __half2float(v[i]);
}

extern "C" __global__
void euler_f16_f32(__half* a, const float* v, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) {
        float x = __half2float(a[i]);
        a[i] = __float2half(x + dt * v[i]);
    }
}

extern "C" __global__
void euler_f16_f16(__half* a, const __half* v, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) {
        float x = __half2float(a[i]);
        float y = __half2float(v[i]);
        a[i] = __float2half(x + dt * y);
    }
}

extern "C" __global__
void add_t_f32(float* t, float dt)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        t[0] = t[0] + dt;
}

extern "C" __global__
void add_t_f16(__half* t, float dt)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float x = __half2float(t[0]);
        t[0] = __float2half(x + dt);
    }
}
"""


def compile_cuda_helpers():
    result = nvrtc.nvrtcCreateProgram(
        KERNEL_SRC.encode(),
        b"vavam_12j.cu",
        0,
        [],
        [],
    )
    program = result[1]

    options = [
        b"--include-path=/usr/local/cuda/include"
    ]

    result = nvrtc.nvrtcCompileProgram(
        program,
        len(options),
        options,
    )
    err = result[0] if isinstance(result, tuple) else result

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        size_result = nvrtc.nvrtcGetProgramLogSize(program)
        log = bytearray(
            size_result[1]
            if isinstance(size_result, tuple)
            else 0
        )
        if log:
            nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(
            "NVRTC compilation failed:\n"
            + bytes(log).decode(
                "utf-8",
                errors="replace",
            )
        )

    ptx_size = nvrtc.nvrtcGetPTXSize(program)[1]
    ptx = bytearray(ptx_size)
    nvrtc.nvrtcGetPTX(program, ptx)

    module = check(
        cuda.cuModuleLoadData(bytes(ptx)),
        "cuModuleLoadData",
    )[1]

    names = [
        "euler_f32_f32",
        "euler_f32_f16",
        "euler_f16_f32",
        "euler_f16_f16",
        "add_t_f32",
        "add_t_f16",
    ]

    funcs = {}
    for name in names:
        funcs[name] = check(
            cuda.cuModuleGetFunction(
                module,
                name.encode(),
            ),
            f"cuModuleGetFunction({name})",
        )[1]

    print("[OK] CUDA Euler kernels ready")
    return funcs


def launch_kernel(func, stream, args, threads=32):
    check(
        cuda.cuLaunchKernel(
            func,
            1, 1, 1,
            threads, 1, 1,
            0,
            stream,
            args,
            0,
        ),
        "cuLaunchKernel",
    )


def make_euler_args(action_ptr, velocity_ptr, dt):
    return (
        (
            int(action_ptr),
            int(velocity_ptr),
            float(dt),
        ),
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
        ),
    )


def make_t_args(t_ptr, dt):
    return (
        (
            int(t_ptr),
            float(dt),
        ),
        (
            ctypes.c_void_p,
            ctypes.c_float,
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pickle",
        default=str(DEFAULT_PICKLE),
    )
    parser.add_argument(
        "--tokens-root",
        default=str(DEFAULT_TOKENS),
    )
    parser.add_argument(
        "--prefill-engine",
        default=str(DEFAULT_PREFILL),
    )
    parser.add_argument(
        "--action-engine",
        default=str(DEFAULT_ACTION),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VaVAM-B 12j — Thor Ego Trajectory Evaluation")
    print("NO PyTorch")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = ThorEgoTrajectoryDataset(
        args.pickle,
        args.tokens_root,
        camera="CAM_FRONT",
        sequence_length=8,
        action_length=6,
        subsampling_factor=1,
    )

    print()
    print("[Dataset]")
    print(f"  valid sequences : {len(dataset)}")

    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"index={args.index}, "
            f"valid range=[0,{len(dataset)-1}]"
        )

    sample = dataset[args.index]

    visual_tokens = np.ascontiguousarray(
        sample["visual_tokens"],
        dtype=np.int64,
    )
    command = np.ascontiguousarray(
        sample["high_level_command"][-1:],
        dtype=np.int64,
    ).reshape(1, 1)
    ground_truth = np.ascontiguousarray(
        sample["positions"][-1],
        dtype=np.float32,
    ).reshape(1, 6, 2)

    print(f"  visual_tokens : {visual_tokens.shape} {visual_tokens.dtype}")
    print(f"  command       : {command.shape} {command.dtype} {command.tolist()}")
    print(f"  ground_truth  : {ground_truth.shape} {ground_truth.dtype}")
    print(f"  scene         : {sample['scene_names'][-1]}")
    print(f"  window_idx    : {sample['window_idx']}")

    print("\nGT trajectory [m]:")
    print(ground_truth[0])

    # ------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------
    check(cuda.cuInit(0), "cuInit")
    device = check(
        cuda.cuDeviceGet(0),
        "cuDeviceGet",
    )[1]
    ctx = check(
        cuda.cuDevicePrimaryCtxRetain(device),
        "cuDevicePrimaryCtxRetain",
    )[1]
    check(
        cuda.cuCtxSetCurrent(ctx),
        "cuCtxSetCurrent",
    )

    stream = stream_create()

    # ------------------------------------------------------------------
    # TensorRT
    # ------------------------------------------------------------------
    kv_names = (
        [f"visual_k_{i}" for i in range(NUM_LAYERS)]
        + [f"visual_v_{i}" for i in range(NUM_LAYERS)]
    )

    prefill = Engine(args.prefill_engine)
    action = Engine(
        args.action_engine,
        skip_names=kv_names,
    )

    # Direct K/V address sharing.
    for name in kv_names:
        if name not in prefill.owned:
            raise RuntimeError(
                f"Prefill output missing: {name}"
            )
        action.set_address(
            name,
            prefill.owned[name],
        )

    print("\n[OK] 48 K/V tensors shared GPU-to-GPU")

    funcs = compile_cuda_helpers()

    # ------------------------------------------------------------------
    # Engine I/O validation
    # ------------------------------------------------------------------
    required_inputs = [
        "noisy_actions",
        "high_level_command",
        "diffusion_step",
    ]

    for name in required_inputs:
        if (
            action.engine.get_tensor_mode(name)
            != trt.TensorIOMode.INPUT
        ):
            raise RuntimeError(
                f"Action input missing: {name}"
            )

    if (
        action.engine.get_tensor_mode("actions")
        != trt.TensorIOMode.OUTPUT
    ):
        raise RuntimeError(
            "Action output 'actions' missing"
        )

    if action.shape("noisy_actions") != ACTION_SHAPE:
        raise RuntimeError(
            f"Unexpected noisy_actions shape: "
            f"{action.shape('noisy_actions')}"
        )

    # ------------------------------------------------------------------
    # Allocate/input setup
    # ------------------------------------------------------------------
    visual_ptr = prefill.owned["visual_tokens"]
    action_ptr = action.owned["noisy_actions"]
    command_ptr = action.owned["high_level_command"]
    t_ptr = action.owned["diffusion_step"]
    velocity_ptr = action.owned["actions"]

    action_dtype = action.dtype("noisy_actions")
    velocity_dtype = action.dtype("actions")
    t_dtype = action.dtype("diffusion_step")
    command_dtype = action.dtype("high_level_command")

    rng = np.random.default_rng(args.seed)
    initial_action = np.ascontiguousarray(
        rng.standard_normal(ACTION_SHAPE).astype(
            action_dtype
        )
    )

    command_host = np.ascontiguousarray(
        command.astype(command_dtype)
    )
    t_host = np.zeros(
        action.shape("diffusion_step"),
        dtype=t_dtype,
    )

    print("\n[Action IO]")
    print(f"  noisy_actions : {action_dtype} {action.shape('noisy_actions')}")
    print(f"  actions       : {velocity_dtype} {action.shape('actions')}")
    print(f"  command       : {command_dtype}")
    print(f"  diffusion     : {t_dtype}")
    print(f"  seed          : {args.seed}")

    # Upload once. These H2D operations are NOT included in inference timing.
    h2d(visual_ptr, visual_tokens[None, ...])
    h2d(action_ptr, initial_action)
    h2d(command_ptr, command_host)
    h2d(t_ptr, t_host)
    stream_sync(stream)

    # ------------------------------------------------------------------
    # Select CUDA kernels.
    # ------------------------------------------------------------------
    if action_dtype == np.dtype(np.float16):
        if velocity_dtype == np.dtype(np.float16):
            euler_func = funcs["euler_f16_f16"]
        elif velocity_dtype == np.dtype(np.float32):
            euler_func = funcs["euler_f16_f32"]
        else:
            raise RuntimeError(
                f"Unsupported velocity dtype: {velocity_dtype}"
            )
    elif action_dtype == np.dtype(np.float32):
        if velocity_dtype == np.dtype(np.float16):
            euler_func = funcs["euler_f32_f16"]
        elif velocity_dtype == np.dtype(np.float32):
            euler_func = funcs["euler_f32_f32"]
        else:
            raise RuntimeError(
                f"Unsupported velocity dtype: {velocity_dtype}"
            )
    else:
        raise RuntimeError(
            f"Unsupported action dtype: {action_dtype}"
        )

    if t_dtype == np.dtype(np.float16):
        t_func = funcs["add_t_f16"]
    elif t_dtype == np.dtype(np.float32):
        t_func = funcs["add_t_f32"]
    else:
        raise RuntimeError(
            f"Unsupported diffusion dtype: {t_dtype}"
        )

    euler_args = make_euler_args(
        action_ptr,
        velocity_ptr,
        DT,
    )
    t_args = make_t_args(
        t_ptr,
        DT,
    )

    # ------------------------------------------------------------------
    # One complete inference
    # ------------------------------------------------------------------
    e_prefill_start = event_create()
    e_prefill_end = event_create()
    e_action_start = event_create()
    e_action_end = event_create()
    e_total_start = event_create()
    e_total_end = event_create()

    event_record(e_total_start, stream)

    event_record(e_prefill_start, stream)
    prefill.execute(stream)
    event_record(e_prefill_end, stream)

    event_record(e_action_start, stream)

    for _ in range(NUM_STEPS):
        action.execute(stream)

        launch_kernel(
            euler_func,
            stream,
            euler_args,
            threads=32,
        )

        launch_kernel(
            t_func,
            stream,
            t_args,
            threads=1,
        )

    event_record(e_action_end, stream)
    event_record(e_total_end, stream)

    stream_sync(stream)

    prefill_ms = elapsed_ms(
        e_prefill_start,
        e_prefill_end,
    )
    action_ms = elapsed_ms(
        e_action_start,
        e_action_end,
    )
    total_ms = elapsed_ms(
        e_total_start,
        e_total_end,
    )

    # Final D2H is outside the measured GPU inference region.
    final_normalized = np.empty(
        action.shape("noisy_actions"),
        dtype=action_dtype,
    )
    d2h(
        final_normalized,
        action_ptr,
    )

    # Official post-processing:
    # _post_process_traj(action) = action * action_scaling
    prediction = (
        final_normalized.astype(np.float32)
        * ACTION_SCALING
    ).reshape(1, 1, 6, 2)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    loss, best_idx = min_ade(
        prediction,
        ground_truth,
        return_idx=True,
        reduction="sum",
    )

    print()
    print("=" * 80)
    print("12j RESULT")
    print("=" * 80)
    print(f"Prefill FP16            : {prefill_ms:.4f} ms")
    print(f"Action FP16 x10 + Euler : {action_ms:.4f} ms")
    print(f"Complete E2E            : {total_ms:.4f} ms")
    print(f"Equivalent inference Hz : {1000.0 / total_ms:.4f} Hz")

    print()
    print("GPU residency:")
    print("  visual K/V    : YES")
    print("  action        : YES")
    print("  diffusion t   : YES")
    print("  Euler         : CUDA kernel")
    print("  per-step H2D  : NONE")
    print("  per-step D2H  : NONE")
    print("  final D2H     : outside timing")

    print()
    print("Prediction [m]:")
    print(prediction[0, 0])

    print()
    print("Ground truth [m]:")
    print(ground_truth[0])

    print()
    print("=" * 80)
    print("Trajectory Evaluation")
    print("=" * 80)
    print(f"minADE          : {float(loss):.6f} m")
    print(f"best mode index : {best_idx.tolist()}")

    print()
    print(
        "M=1 for this first 12j test, therefore minADE == ADE."
    )

    action.cleanup()
    prefill.cleanup()

    print("\n[OK] 12j completed.")


if __name__ == "__main__":
    main()
