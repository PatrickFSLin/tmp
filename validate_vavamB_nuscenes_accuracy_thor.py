#!/usr/bin/env python3
# VaVAM-B Thor accuracy sanity check.
# Reuses the validated 12i TensorRT + CUDA Euler structure.
# Phase A: one sample, up to 10 trajectories, nested minADE M=1/4/10.

import argparse
import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cuda, nvrtc

from thor_ego_trajectory_dataset import ThorEgoTrajectoryDataset

ROOT = Path.home() / "vblkdev2" / "VaVAM_Thor"
DEFAULT_PREFILL_ENGINE = ROOT / "Engines" / "vavam_joint_kv_prefill_B_v10_fp16.engine"
DEFAULT_ACTION_ENGINE = ROOT / "Engines" / "vavam_joint_action_B_fp16.engine"
DEFAULT_PICKLE = ROOT / "data" / "nuScenes-mini" / "nuscenes_mini_data_cleaned.pkl"
DEFAULT_TOKENS = ROOT / "data" / "nuScenes-mini" / "tokens"

NUM_LAYERS = 24
NUM_STEPS = 10
DT = 0.1
ACTION_SHAPE = (1, 1, 6, 2)

def check(result, name):
    err = result[0] if isinstance(result, tuple) else result
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {err}")
    return result

def malloc(nbytes):
    return check(cuda.cuMemAlloc(int(nbytes)), "cuMemAlloc")[1]

def free(ptr):
    if ptr is not None:
        check(cuda.cuMemFree(int(ptr)), "cuMemFree")

def h2d(ptr, arr):
    arr = np.ascontiguousarray(arr)
    check(cuda.cuMemcpyHtoD(int(ptr), arr, int(arr.nbytes)), "cuMemcpyHtoD")

def d2h(arr, ptr):
    arr = np.ascontiguousarray(arr)
    check(cuda.cuMemcpyDtoH(arr, int(ptr), int(arr.nbytes)), "cuMemcpyDtoH")

def stream_create():
    return check(cuda.cuStreamCreate(0), "cuStreamCreate")[1]

def stream_sync(stream):
    check(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize")

class Engine:
    def __init__(self, path, skip_names=None):
        self.path = Path(path)
        self.owned = {}
        self.skip_names = set(skip_names or [])
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(self.path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {self.path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create context for {self.path}")

        print(f"\nLoaded: {self.path}")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            print(f"  {'IN ' if mode == trt.TensorIOMode.INPUT else 'OUT'} "
                  f"{name:24s} dtype={dtype} shape={shape}")
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Dynamic shape unsupported: {name} {shape}")
            if name in self.skip_names:
                continue
            np_dtype = np.dtype(trt.nptype(dtype))
            self.owned[name] = malloc(int(np.prod(shape)) * np_dtype.itemsize)
        self.set_owned_addresses()

    def set_owned_addresses(self):
        for name, ptr in self.owned.items():
            if not self.context.set_tensor_address(name, int(ptr)):
                raise RuntimeError(f"set_tensor_address failed: {name}")

    def set_address(self, name, ptr):
        if not self.context.set_tensor_address(name, int(ptr)):
            raise RuntimeError(f"set_tensor_address failed: {name}")

    def execute(self, stream):
        if not self.context.execute_async_v3(stream_handle=int(stream)):
            raise RuntimeError(f"execute_async_v3 failed: {self.path.name}")

    def dtype(self, name):
        return np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))

    def shape(self, name):
        return tuple(self.engine.get_tensor_shape(name))

    def cleanup(self):
        for ptr in self.owned.values():
            free(ptr)
        self.owned.clear()

KERNEL_SRC = r'''
#include <cuda_fp16.h>
extern "C" __global__
void euler_f32_f32(float* action, const float* vel, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) action[i] += dt * vel[i];
}
extern "C" __global__
void euler_f32_f16(float* action, const __half* vel, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) action[i] += dt * __half2float(vel[i]);
}
extern "C" __global__
void euler_f16_f32(__half* action, const float* vel, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) {
        float a = __half2float(action[i]);
        action[i] = __float2half(a + dt * vel[i]);
    }
}
extern "C" __global__
void euler_f16_f16(__half* action, const __half* vel, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) {
        float a = __half2float(action[i]);
        float v = __half2float(vel[i]);
        action[i] = __float2half(a + dt * v);
    }
}
extern "C" __global__
void add_t_f32(float* t, float dt) {
    if (blockIdx.x == 0 && threadIdx.x == 0) t[0] += dt;
}
extern "C" __global__
void add_t_f16(__half* t, float dt) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        t[0] = __float2half(__half2float(t[0]) + dt);
}
'''

def compile_cuda_module():
    result = nvrtc.nvrtcCreateProgram(
        KERNEL_SRC.encode(), b"vavam_accuracy.cu", 0, [], []
    )
    err, program = result if isinstance(result, tuple) else (result, None)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram failed: {err}")

    options = [b"--include-path=/usr/local/cuda/include"]
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    err = result[0] if isinstance(result, tuple) else result
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        size_result = nvrtc.nvrtcGetProgramLogSize(program)
        log = bytearray(size_result[1])
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(
            f"NVRTC compilation failed: {err}\n"
            + bytes(log).rstrip(b"\\x00").decode("utf-8", errors="replace")
        )

    ptx_size = nvrtc.nvrtcGetPTXSize(program)[1]
    ptx = bytearray(ptx_size)
    result = nvrtc.nvrtcGetPTX(program, ptx)
    if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcGetPTX failed: {result}")

    module = check(cuda.cuModuleLoadData(bytes(ptx)), "cuModuleLoadData")[1]
    funcs = {}
    for name in [
        "euler_f32_f32", "euler_f32_f16", "euler_f16_f32",
        "euler_f16_f16", "add_t_f32", "add_t_f16"
    ]:
        funcs[name] = check(
            cuda.cuModuleGetFunction(module, name.encode()),
            f"cuModuleGetFunction({name})"
        )[1]
    return module, funcs

def launch_kernel(func, stream, args, threads=32):
    check(cuda.cuLaunchKernel(
        func, 1, 1, 1, threads, 1, 1, 0, stream, args, 0
    ), "cuLaunchKernel")

def make_euler_args(action_ptr, velocity_ptr, dt):
    return (
        (int(action_ptr), int(velocity_ptr), float(dt)),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float),
    )

def make_t_args(t_ptr, dt):
    return (
        (int(t_ptr), float(dt)),
        (ctypes.c_void_p, ctypes.c_float),
    )

class ThorVaVAMRunner:
    def __init__(self, prefill_engine, action_engine):
        self.prefill = Engine(prefill_engine)
        kv_names = (
            [f"visual_k_{i}" for i in range(NUM_LAYERS)] +
            [f"visual_v_{i}" for i in range(NUM_LAYERS)]
        )
        self.action = Engine(action_engine, skip_names=kv_names)

        for name in ["noisy_actions", "high_level_command", "diffusion_step"]:
            if self.action.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                raise RuntimeError(f"Missing Action input: {name}")
        if self.action.engine.get_tensor_mode("actions") != trt.TensorIOMode.OUTPUT:
            raise RuntimeError("Expected Action output named 'actions'")

        for name in kv_names:
            if name not in self.prefill.owned:
                raise RuntimeError(f"Prefill output missing: {name}")
            self.action.set_address(name, self.prefill.owned[name])

        print("[OK] 48 visual K/V tensors are GPU-resident and shared")

        self.stream = stream_create()
        self.module, self.funcs = compile_cuda_module()

        self.action_dtype = self.action.dtype("noisy_actions")
        self.velocity_dtype = self.action.dtype("actions")
        self.t_dtype = self.action.dtype("diffusion_step")

        if self.action.shape("noisy_actions") != ACTION_SHAPE:
            raise RuntimeError(
                f"Unexpected noisy_actions shape: "
                f"{self.action.shape('noisy_actions')}"
            )

        action_f16 = self.action_dtype == np.dtype(np.float16)
        velocity_f16 = self.velocity_dtype == np.dtype(np.float16)
        euler_name = (
            "euler_f16_f16" if action_f16 and velocity_f16 else
            "euler_f16_f32" if action_f16 else
            "euler_f32_f16" if velocity_f16 else
            "euler_f32_f32"
        )
        self.euler_func = self.funcs[euler_name]
        self.t_func = (
            self.funcs["add_t_f16"]
            if self.t_dtype == np.dtype(np.float16)
            else self.funcs["add_t_f32"]
        )

        self.visual_ptr = self.prefill.owned["visual_tokens"]
        self.action_ptr = self.action.owned["noisy_actions"]
        self.command_ptr = self.action.owned["high_level_command"]
        self.t_ptr = self.action.owned["diffusion_step"]
        self.velocity_ptr = self.action.owned["actions"]

        self.euler_args = make_euler_args(
            self.action_ptr, self.velocity_ptr, DT
        )
        self.t_args = make_t_args(self.t_ptr, DT)
        self.prefill_done = False

    def run_prefill(self, visual_tokens, command):
        visual_tokens = np.ascontiguousarray(visual_tokens, dtype=np.int64)
        command = np.ascontiguousarray(
            command, dtype=self.action.dtype("high_level_command")
        )

        expected = self.prefill.shape("visual_tokens")
        if tuple(visual_tokens.shape) != tuple(expected):
            raise RuntimeError(
                f"visual_tokens shape {visual_tokens.shape}, expected {expected}"
            )

        h2d(self.visual_ptr, visual_tokens)
        h2d(self.command_ptr, command)
        self.prefill.execute(self.stream)
        stream_sync(self.stream)
        self.prefill_done = True

    def run_trajectory(self, initial_action):
        if not self.prefill_done:
            raise RuntimeError("run_prefill() must be called first")

        initial_action = np.ascontiguousarray(
            initial_action, dtype=self.action_dtype
        )
        t_host = np.zeros(
            self.action.shape("diffusion_step"), dtype=self.t_dtype
        )

        h2d(self.action_ptr, initial_action)
        h2d(self.t_ptr, t_host)
        stream_sync(self.stream)

        for _ in range(NUM_STEPS):
            self.action.execute(self.stream)
            launch_kernel(
                self.euler_func, self.stream, self.euler_args, threads=32
            )
            launch_kernel(
                self.t_func, self.stream, self.t_args, threads=1
            )

        stream_sync(self.stream)

        output = np.empty(
            self.action.shape("noisy_actions"), dtype=self.action_dtype
        )
        d2h(output, self.action_ptr)

        if not np.all(np.isfinite(output)):
            raise RuntimeError(
                "Non-finite trajectory: "
                f"nan={np.isnan(output).sum()}, "
                f"inf={np.isinf(output).sum()}"
            )
        return output.astype(np.float32)

    def cleanup(self):
        self.action.cleanup()
        self.prefill.cleanup()

def compute_ade(prediction, gt):
    errors = np.linalg.norm(prediction - gt[None, :, :], axis=-1)
    return errors.mean(axis=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-engine", default=str(DEFAULT_PREFILL_ENGINE))
    parser.add_argument("--action-engine", default=str(DEFAULT_ACTION_ENGINE))
    parser.add_argument("--pickle", default=str(DEFAULT_PICKLE))
    parser.add_argument("--tokens-root", default=str(DEFAULT_TOKENS))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 1 <= args.num_trajectories <= 10:
        raise ValueError("--num-trajectories must be in [1, 10]")

    print("=" * 80)
    print("VaVAM-B Thor Accuracy Sanity Check")
    print("=" * 80)
    print("TensorRT:", trt.__version__)
    print("Sample:", args.index)
    print("Trajectories:", args.num_trajectories)
    print("Seed:", args.seed)

    check(cuda.cuInit(0), "cuInit")
    device = check(cuda.cuDeviceGet(0), "cuDeviceGet")[1]
    ctx = check(
        cuda.cuDevicePrimaryCtxRetain(device),
        "cuDevicePrimaryCtxRetain"
    )[1]
    check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    dataset = ThorEgoTrajectoryDataset(
        pickle_path=args.pickle,
        tokens_rootdir=args.tokens_root
    )
    print("Dataset size:", len(dataset))

    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"index {args.index} outside [0, {len(dataset)})"
        )

    sample = dataset[args.index]
    print("\nScene:", sample["scene_names"][0])
    print("Window:", sample["window_idx"])
    print("visual_tokens:", sample["visual_tokens"].shape)
    print("command:", sample["high_level_command"].tolist())
    print("positions:", sample["positions"].shape)

    commands = np.asarray(
        sample["high_level_command"][-1:], dtype=np.int64
    ).reshape(1, 1)
    gt = np.asarray(sample["positions"][-1], dtype=np.float32)
    visual_tokens = np.asarray(
        sample["visual_tokens"], dtype=np.int64
    )

    runner = ThorVaVAMRunner(
        args.prefill_engine,
        args.action_engine
    )

    try:
        runner.run_prefill(visual_tokens, commands)

        rng = np.random.default_rng(args.seed)
        trajectories = []
        ades = []

        print("\n=== 10-trajectory evaluation ===")
        for i in range(args.num_trajectories):
            initial_action = rng.standard_normal(
                ACTION_SHAPE
            ).astype(np.float32)

            trajectory = runner.run_trajectory(initial_action)[0]
            ade = float(compute_ade(
                trajectory[None, :, :], gt
            )[0])

            trajectories.append(trajectory)
            ades.append(ade)

            print(f"Trajectory {i:2d}: ADE={ade:.6f} m")

        ades = np.asarray(ades, dtype=np.float64)

        print("\n" + "=" * 80)
        print("Nested minADE")
        print("=" * 80)

        for m in (1, 4, 10):
            if m <= args.num_trajectories:
                idx = int(np.argmin(ades[:m]))
                print(
                    f"M={m:2d}: minADE={ades[idx]:.6f} m, "
                    f"best trajectory={idx}"
                )

        print("\nPrediction shape:", np.asarray(trajectories).shape)
        print("All finite:", bool(np.all(np.isfinite(trajectories))))
        print("\n[OK] Thor accuracy sanity check completed.")

    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()
