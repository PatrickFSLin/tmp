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


def event_create():
    return check(cuda.cuEventCreate(0), "cuEventCreate")[1]


def event_record(event, stream):
    check(cuda.cuEventRecord(event, stream), "cuEventRecord")


def event_sync(event):
    check(cuda.cuEventSynchronize(event), "cuEventSynchronize")


def event_elapsed_ms(start_event, end_event):
    result = check(cuda.cuEventElapsedTime(start_event, end_event), "cuEventElapsedTime")
    return float(result[1] if isinstance(result, tuple) else result)


def event_destroy(event):
    if event is not None:
        check(cuda.cuEventDestroy(event), "cuEventDestroy")


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

        self.prefill_start_event = event_create()
        self.prefill_end_event = event_create()
        self.action_start_event = event_create()
        self.action_end_event = event_create()
        self.e2e_start_event = event_create()
        self.e2e_end_event = event_create()

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

        event_record(self.e2e_start_event, self.stream)
        event_record(self.prefill_start_event, self.stream)
        self.prefill.execute(self.stream)
        event_record(self.prefill_end_event, self.stream)
        event_sync(self.prefill_end_event)

        self.prefill_done = True
        return event_elapsed_ms(self.prefill_start_event, self.prefill_end_event)

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

        event_record(self.action_start_event, self.stream)

        for _ in range(NUM_STEPS):
            self.action.execute(self.stream)
            launch_kernel(
                self.euler_func, self.stream, self.euler_args, threads=32
            )
            launch_kernel(
                self.t_func, self.stream, self.t_args, threads=1
            )

        event_record(self.action_end_event, self.stream)
        event_sync(self.action_end_event)
        action_ms = event_elapsed_ms(self.action_start_event, self.action_end_event)

        event_record(self.e2e_end_event, self.stream)
        event_sync(self.e2e_end_event)
        e2e_ms = event_elapsed_ms(self.e2e_start_event, self.e2e_end_event)

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
        return output.astype(np.float32), action_ms, e2e_ms

    def cleanup(self):
        for event in (
            self.prefill_start_event, self.prefill_end_event,
            self.action_start_event, self.action_end_event,
            self.e2e_start_event, self.e2e_end_event,
        ):
            event_destroy(event)
        self.action.cleanup()
        self.prefill.cleanup()

def compute_ade(prediction, gt):
    errors = np.linalg.norm(prediction - gt[None, :, :], axis=-1)
    return errors.mean(axis=-1)



def main():
    parser = argparse.ArgumentParser(
        description=(
            "VaVAM-B Thor full nuScenes-mini accuracy evaluation. "
            "Processes all valid 8-frame windows backed by tokens/*.npy "
            "and saves trajectory plots."
        )
    )
    parser.add_argument("--prefill-engine", default=str(DEFAULT_PREFILL_ENGINE))
    parser.add_argument("--action-engine", default=str(DEFAULT_ACTION_ENGINE))
    parser.add_argument("--pickle", default=str(DEFAULT_PICKLE))
    parser.add_argument("--tokens-root", default=str(DEFAULT_TOKENS))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="Exclusive dataset index; -1 means all valid samples.",
    )
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--summary-csv",
        default=str(ROOT / "thor_eval" / "vavamB_nuscenes_accuracy_summary.csv"),
    )
    parser.add_argument(
        "--results-npz",
        default=str(ROOT / "thor_eval" / "vavamB_nuscenes_accuracy_results.npz"),
    )
    args = parser.parse_args()

    if not 1 <= args.num_trajectories <= 10:
        raise ValueError("--num-trajectories must be in [1, 10]")

    summary_csv = Path(args.summary_csv).expanduser().resolve()
    results_npz = Path(args.results_npz).expanduser().resolve()
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    results_npz.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("VaVAM-B Thor — Full nuScenes-mini Accuracy + Trajectory Plots")
    print("=" * 90)
    print("TensorRT:", trt.__version__)
    print("Seed:", args.seed)
    print("Trajectories per sample:", args.num_trajectories)
    print("Tokens root:", args.tokens_root)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    check(cuda.cuInit(0), "cuInit")
    device = check(cuda.cuDeviceGet(0), "cuDeviceGet")[1]
    ctx = check(cuda.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain")[1]
    check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    dataset = ThorEgoTrajectoryDataset(
        pickle_path=args.pickle,
        tokens_rootdir=args.tokens_root,
    )
    dataset_size = len(dataset)

    token_files = sorted(Path(args.tokens_root).expanduser().glob("*.npy"))
    print("Token .npy files found:", len(token_files))
    if len(token_files) != 404:
        print(
            f"[WARN] Expected 404 token files, found {len(token_files)}. "
            "The evaluation will still use all valid dataset windows."
        )

    end_index = dataset_size if args.end_index < 0 else min(args.end_index, dataset_size)
    if not 0 <= args.start_index < end_index:
        raise ValueError(
            f"Invalid range [{args.start_index}, {end_index}) for dataset size {dataset_size}"
        )
    num_samples = end_index - args.start_index

    print("Valid evaluation windows:", dataset_size)
    print(f"Evaluation range: [{args.start_index}, {end_index}) = {num_samples} samples")
    print(
        "Note: the 404 .npy files are frame-level tokens. "
        "Each VaVAM-B sample consumes 8 consecutive token files, so the official "
        f"dataset produces {dataset_size} valid 8-frame evaluation windows."
    )

    # ------------------------------------------------------------------
    # Runner: load engines once; K/V stays GPU-resident between prefill/action.
    # ------------------------------------------------------------------
    runner = ThorVaVAMRunner(args.prefill_engine, args.action_engine)

    # Results are kept compact enough for 274 samples x 10 trajectories.
    indices = np.arange(args.start_index, end_index, dtype=np.int64)
    scenes = np.empty(num_samples, dtype="<U128")
    windows = np.empty(num_samples, dtype=np.int64)
    commands_all = np.empty(num_samples, dtype=np.int64)
    gt_all = np.empty((num_samples, 6, 2), dtype=np.float32)
    predictions_all = np.empty((num_samples, 10, 6, 2), dtype=np.float32)
    ade_all = np.empty((num_samples, 10), dtype=np.float32)
    minade_m1 = np.empty(num_samples, dtype=np.float32)
    minade_m4 = np.empty(num_samples, dtype=np.float32)
    minade_m10 = np.empty(num_samples, dtype=np.float32)
    best_m1 = np.empty(num_samples, dtype=np.int64)
    best_m4 = np.empty(num_samples, dtype=np.int64)
    best_m10 = np.empty(num_samples, dtype=np.int64)

    # GPU timing arrays.
    prefill_latency_ms = np.empty(num_samples, dtype=np.float32)
    action_latency_ms = np.empty(
        (num_samples, args.num_trajectories), dtype=np.float32
    )
    e2e_latency_ms = np.empty(
        (num_samples, args.num_trajectories), dtype=np.float32
    )

    referenced_token_paths = set()

    import csv
    csv_rows = []

    try:
        for local_idx, sample_index in enumerate(range(args.start_index, end_index)):
            sample = dataset[sample_index]

            visual_tokens = np.asarray(sample["visual_tokens"], dtype=np.int64)
            if visual_tokens.shape != (8, 18, 32):
                raise RuntimeError(
                    f"Unexpected visual token shape at sample {sample_index}: "
                    f"{visual_tokens.shape}"
                )
            visual_tokens = np.ascontiguousarray(visual_tokens)

            command = np.asarray(
                sample["high_level_command"][-1:], dtype=np.int64
            ).reshape(1, 1)
            gt = np.asarray(sample["positions"][-1], dtype=np.float32)
            if gt.shape != (6, 2):
                raise RuntimeError(
                    f"Unexpected GT shape at sample {sample_index}: {gt.shape}"
                )

            scene = str(sample["scene_names"][0])
            window_idx = int(sample["window_idx"])
            file_paths = list(sample.get("file_paths", []))
            for fp in file_paths:
                referenced_token_paths.add(
                    str((Path(args.tokens_root).expanduser() / Path(fp).with_suffix(".npy")).resolve())
                )
            first_frame = str(file_paths[0]) if file_paths else ""
            last_frame = str(file_paths[-1]) if file_paths else ""

            # Same semantics as the validated one-sample script:
            # one prefill, then the same seeded sequence of stochastic initial
            # actions, giving nested M=1/4/10 subsets.
            prefill_ms = runner.run_prefill(visual_tokens, command)
            prefill_latency_ms[local_idx] = prefill_ms
            rng = np.random.default_rng(args.seed)

            trajectories = []
            ades = []
            for traj_idx in range(args.num_trajectories):
                initial_action = rng.standard_normal(ACTION_SHAPE).astype(np.float32)
                trajectory, action_ms, e2e_ms = runner.run_trajectory(initial_action)
                action_latency_ms[local_idx, traj_idx] = action_ms
                e2e_latency_ms[local_idx, traj_idx] = e2e_ms

                # Normalize runner output to (6, 2).
                # Thor runner may return (1, 6, 2).
                trajectory = np.asarray(trajectory, dtype=np.float32)
                if trajectory.ndim == 3 and trajectory.shape[0] == 1:
                    trajectory = trajectory[0]
                if trajectory.shape != (6, 2):
                    raise RuntimeError(
                        f"Unexpected trajectory shape at sample {sample_index}, "
                        f"traj {traj_idx}: {trajectory.shape}"
                    )

                # Official VaVAM-B post-processing used by 12j:
                # normalized action -> meters.
                trajectory = trajectory * ACTION_SCALING

                ade = float(compute_ade(trajectory[None, :, :], gt)[0])
                trajectories.append(trajectory)
                ades.append(ade)

            trajectories = np.asarray(trajectories, dtype=np.float32)
            ades = np.asarray(ades, dtype=np.float32)

            if args.num_trajectories < 10:
                # Keep the fixed output shape for easy downstream plotting/NPZ.
                predictions_all[local_idx, :, :, :] = np.nan
                ade_all[local_idx, :] = np.nan
                predictions_all[local_idx, :args.num_trajectories] = trajectories
                ade_all[local_idx, :args.num_trajectories] = ades
            else:
                predictions_all[local_idx] = trajectories
                ade_all[local_idx] = ades

            scenes[local_idx] = scene
            windows[local_idx] = window_idx
            commands_all[local_idx] = int(command[0, 0])
            gt_all[local_idx] = gt

            b1 = int(np.argmin(ades[:1]))
            b4 = int(np.argmin(ades[:4])) if args.num_trajectories >= 4 else b1
            b10 = int(np.argmin(ades[:10])) if args.num_trajectories >= 10 else int(np.argmin(ades))
            m1 = float(ades[b1])
            m4 = float(ades[b4])
            m10 = float(ades[b10])

            minade_m1[local_idx] = m1
            minade_m4[local_idx] = m4
            minade_m10[local_idx] = m10
            best_m1[local_idx] = b1
            best_m4[local_idx] = b4
            best_m10[local_idx] = b10

            csv_rows.append(
                {
                    "sample_index": sample_index,
                    "scene": scene,
                    "window_idx": window_idx,
                    "command": int(command[0, 0]),
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "M1_minADE_m": m1,
                    "M4_minADE_m": m4,
                    "M10_minADE_m": m10,
                    "M1_best_idx": b1,
                    "M4_best_idx": b4,
                    "M10_best_idx": b10,
                    "VaViM_prefill_ms": float(prefill_ms),
                    "ActionExpert_mean_ms": float(np.mean(action_latency_ms[local_idx])),
                    "ActionExpert_min_ms": float(np.min(action_latency_ms[local_idx])),
                    "ActionExpert_max_ms": float(np.max(action_latency_ms[local_idx])),
                    "E2E_mean_ms": float(np.mean(e2e_latency_ms[local_idx])),
                    "E2E_min_ms": float(np.min(e2e_latency_ms[local_idx])),
                    "E2E_max_ms": float(np.max(e2e_latency_ms[local_idx])),
                }
            )

            if (
                local_idx < 5
                or (local_idx + 1) % 10 == 0
                or local_idx + 1 == num_samples
            ):
                action_mean = float(np.mean(action_latency_ms[local_idx]))
                e2e_mean = float(np.mean(e2e_latency_ms[local_idx]))
                print(
                    f"[{local_idx + 1:4d}/{num_samples}] "
                    f"index={sample_index:4d} scene={scene} "
                    f"M1={m1:.4f} m M4={m4:.4f} m M10={m10:.4f} m | "
                    f"VaViM={prefill_ms:.3f} ms | "
                    f"Action={action_mean:.3f} ms | "
                    f"E2E={e2e_mean:.3f} ms"
                )

    finally:
        runner.cleanup()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    np.savez_compressed(
        results_npz,
        indices=indices,
        scenes=scenes,
        windows=windows,
        commands=commands_all,
        ground_truth=gt_all,
        predictions=predictions_all,
        ADE=ade_all,
        minADE_M1=minade_m1,
        minADE_M4=minade_m4,
        minADE_M10=minade_m10,
        best_idx_M1=best_m1,
        best_idx_M4=best_m4,
        best_idx_M10=best_m10,
        VaViM_prefill_ms=prefill_latency_ms,
        ActionExpert_latency_ms=action_latency_ms,
        E2E_latency_ms=e2e_latency_ms,
        action_scaling=np.asarray(ACTION_SCALING, dtype=np.float32),
        seed=np.asarray(args.seed, dtype=np.int64),
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Samples evaluated : {num_samples}")
    print(f"Token .npy files  : {len(token_files)}")
    print(f"Unique token files referenced by evaluated windows: {len(referenced_token_paths)}")
    missing = [p for p in sorted(referenced_token_paths) if not Path(p).is_file()]
    if missing:
        print(f"[WARN] Missing referenced token files: {len(missing)}")
        for p in missing[:10]:
            print(f"       {p}")
    else:
        print("All referenced token .npy files exist.")
    print(f"CSV summary       : {summary_csv}")
    print(f"NPZ results       : {results_npz}")
    print()
    print("Dataset mean minADE:")
    print(f"  M=1  : {np.mean(minade_m1):.6f} m")
    print(f"  M=4  : {np.mean(minade_m4):.6f} m")
    print(f"  M=10 : {np.mean(minade_m10):.6f} m")
    print()
    print("Latency summary (CUDA GPU execution only):")
    print(f"  VaViM prefill : {np.mean(prefill_latency_ms):.3f} ms "
          f"(median {np.median(prefill_latency_ms):.3f} ms)")
    print(f"  Action Expert : {np.mean(action_latency_ms):.3f} ms "
          f"(10-step + Euler, per trajectory)")
    print(f"  E2E           : {np.mean(e2e_latency_ms):.3f} ms "
          f"(VaViM + one 10-step trajectory)")
    print(f"  E2E median    : {np.median(e2e_latency_ms):.3f} ms")
    print(f"  E2E p95       : {np.percentile(e2e_latency_ms, 95):.3f} ms")
    print(f"  E2E p99       : {np.percentile(e2e_latency_ms, 99):.3f} ms")
    print()
    print("[OK] Full Thor accuracy + latency evaluation completed (no plots).")


if __name__ == "__main__":
    main()
