#!/usr/bin/env python3
"""
VaVAM Thor: Visual Tokens -> Action Velocity / Euler Benchmark

Based on the previously working 12i-style Thor script.

Purpose
-------
Run the same Visual-Tokens -> TensorRT -> Euler validation with different
TensorRT engines and measure performance.

Typical engines:
    vavam_joint_inference_step_B_fp32.engine
    vavam_joint_inference_step_B_fp32_noTF32.engine
    vavam_joint_inference_step_B_fp16.engine
    vavam_joint_inference_step_B_bf16.engine
    vavam_joint_inference_step_L_fp16.engine
    vavam_joint_inference_step_L_bf16.engine
    ...

The script keeps the original 12i/12h behavior:
    CAM_FRONT visual tokens
        -> one TensorRT action-velocity step
        -> Euler update
        -> repeat 10 steps
        -> final trajectory/action
        -> compare with the PyTorch reference in the NPZ

In addition it reports:
    - H2D latency per step
    - TensorRT GPU execution latency per step
    - D2H latency per step
    - total per-step latency
    - total 10-step Euler latency
    - average/min/max per-step latency
    - numerical errors for velocity/action/final trajectory

IMPORTANT
---------
The reference NPZ supplies initial_action, command, diffusion step and the
PyTorch reference. The CAM_FRONT .npy files are independently loaded and
verified against the visual_tokens stored in the NPZ.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

from cuda import cuda


# ============================================================================
# Constants
# ============================================================================

CONTEXT_LENGTH = 8
ACTION_HORIZON = 6
ACTION_DIM = 2

NUM_EULER_STEPS = 10
DELTA_T = 0.1

ATOL = 1e-4


# ============================================================================
# CUDA helpers
# ============================================================================

def cuda_check(result, name):
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {err}")


def init_cuda():
    cuda_check(cuda.cuInit(0), "cuInit")

    result = cuda.cuDeviceGet(0)
    cuda_check(result, "cuDeviceGet")
    device = result[1]

    result = cuda.cuDevicePrimaryCtxRetain(device)
    cuda_check(result, "cuDevicePrimaryCtxRetain")
    context = result[1]

    cuda_check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")

    print("[OK] CUDA primary context created/set")


def cuda_malloc(nbytes):
    result = cuda.cuMemAlloc(int(nbytes))
    cuda_check(result, "cuMemAlloc")
    return result[1]


def cuda_free(ptr):
    if ptr is None:
        return

    result = cuda.cuMemFree(ptr)
    cuda_check(result, "cuMemFree")


def cuda_memcpy_htod(device_ptr, host_array):
    arr = np.ascontiguousarray(host_array)

    result = cuda.cuMemcpyHtoD(
        int(device_ptr),
        arr,
        int(arr.nbytes),
    )
    cuda_check(result, "cuMemcpyHtoD")


def cuda_memcpy_dtoh(host_array, device_ptr):
    arr = np.ascontiguousarray(host_array)

    result = cuda.cuMemcpyDtoH(
        arr,
        int(device_ptr),
        int(arr.nbytes),
    )
    cuda_check(result, "cuMemcpyDtoH")


# ============================================================================
# TensorRT dtype helpers
# ============================================================================

def is_bf16(dtype):
    return hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16


def trt_to_numpy_dtype(dtype):
    if dtype == trt.DataType.FLOAT:
        return np.float32

    if dtype == trt.DataType.HALF:
        return np.float16

    if is_bf16(dtype):
        # Native NumPy bfloat16 is not assumed.
        return None

    if dtype == trt.DataType.INT8:
        return np.int8

    if dtype == trt.DataType.INT32:
        return np.int32

    if hasattr(trt.DataType, "INT64") and dtype == trt.DataType.INT64:
        return np.int64

    if dtype == trt.DataType.BOOL:
        return np.bool_

    raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")


def make_bf16_storage(source, shape):
    """
    Return a contiguous 2-byte-per-element host buffer for TensorRT BF16.

    If NumPy has native bfloat16, use it.
    Otherwise store raw BF16 bits in uint16.
    """
    try:
        bf16_dtype = np.dtype("bfloat16")
        arr = np.asarray(source, dtype=bf16_dtype)
        if tuple(arr.shape) != tuple(shape):
            raise RuntimeError(
                f"BF16 shape mismatch: engine={shape}, input={arr.shape}"
            )
        return np.ascontiguousarray(arr)
    except TypeError:
        x = np.asarray(source, dtype=np.float32)

        if tuple(x.shape) != tuple(shape):
            raise RuntimeError(
                f"BF16 shape mismatch: engine={shape}, input={x.shape}"
            )

        x = np.ascontiguousarray(x)
        u32 = x.view(np.uint32)

        # Round-to-nearest-even conversion from FP32 to BF16.
        rounding = ((u32 >> 16) & 1) + 0x7FFF
        bits = ((u32 + rounding) >> 16).astype(np.uint16)

        return np.ascontiguousarray(bits)


def make_host_array(source, shape, dtype, tensor_name):
    source = np.asarray(source)

    if is_bf16(dtype):
        return make_bf16_storage(source, shape)

    np_dtype = trt_to_numpy_dtype(dtype)
    arr = np.asarray(source, dtype=np_dtype)
    arr = np.ascontiguousarray(arr)

    if tuple(arr.shape) != tuple(shape):
        raise RuntimeError(
            f"Shape mismatch for '{tensor_name}': "
            f"engine={shape}, input={arr.shape}"
        )

    return arr


def make_output_host(shape, dtype):
    if is_bf16(dtype):
        try:
            return np.empty(shape, dtype=np.dtype("bfloat16"))
        except TypeError:
            return np.empty(shape, dtype=np.uint16)

    return np.empty(
        shape,
        dtype=trt_to_numpy_dtype(dtype),
    )


def bf16_to_float32(arr):
    if arr.dtype == np.uint16:
        u32 = arr.astype(np.uint32) << 16
        return u32.view(np.float32)

    return np.asarray(arr, dtype=np.float32)


# ============================================================================
# Reference / CAM_FRONT tokens
# ============================================================================

def load_reference(reference_path):
    print()
    print("=" * 80)
    print("Loading 12h/12i reference")
    print("=" * 80)

    data = np.load(reference_path)

    required = [
        "scene_id",
        "window_index",
        "visual_tokens",
        "initial_action",
        "high_level_command",
        "initial_diffusion_step",
        "pytorch_velocity_history",
        "pytorch_action_history",
        "pytorch_final_trajectory",
    ]

    for name in required:
        if name not in data:
            raise RuntimeError(f"Missing NPZ field: {name}")

    visual_tokens = data["visual_tokens"].astype(np.int64)
    initial_action = data["initial_action"].astype(np.float32)
    high_level_command = data["high_level_command"].astype(np.int64)
    initial_diffusion_step = data["initial_diffusion_step"].astype(np.float32)

    pytorch_velocity = data["pytorch_velocity_history"].astype(np.float32)
    pytorch_action = data["pytorch_action_history"].astype(np.float32)
    pytorch_final = data["pytorch_final_trajectory"].astype(np.float32)

    print(f"  scene_id       : {data['scene_id']}")
    print(f"  window_index   : {data['window_index']}")

    print()
    print("[Reference inputs]")
    print(f"  visual_tokens      : {visual_tokens.shape} {visual_tokens.dtype}")
    print(f"  initial_action     : {initial_action.shape} {initial_action.dtype}")
    print(
        f"  high_level_command : "
        f"{high_level_command.shape} {high_level_command.dtype}"
    )
    print(
        f"  diffusion_step     : "
        f"{initial_diffusion_step.shape} {initial_diffusion_step.dtype}"
    )

    return {
        "scene_id": str(data["scene_id"]),
        "window_index": int(data["window_index"]),
        "visual_tokens": visual_tokens,
        "initial_action": initial_action,
        "high_level_command": high_level_command,
        "initial_diffusion_step": initial_diffusion_step,
        "pytorch_velocity": pytorch_velocity,
        "pytorch_action": pytorch_action,
        "pytorch_final": pytorch_final,
    }


def load_scene_tokens(token_dir, scene_id):
    token_dir = Path(token_dir)

    files = sorted(
        token_dir.glob(
            f"{scene_id}__CAM_FRONT__*.npy"
        )
    )

    if not files:
        raise RuntimeError(
            f"No token files found for scene: {scene_id}"
        )

    parsed = []

    for path in files:
        parts = path.stem.split("__")

        if len(parts) != 3:
            continue

        timestamp = parts[2]

        try:
            parsed.append((int(timestamp), path))
        except ValueError:
            continue

    parsed.sort(key=lambda x: x[0])
    return parsed


def build_visual_tokens(files, window_index):
    start = window_index
    end = start + CONTEXT_LENGTH

    if end > len(files):
        raise RuntimeError(
            f"Not enough frames for window {window_index}: "
            f"need {CONTEXT_LENGTH}, have {len(files) - start}"
        )

    selected = files[start:end]

    tokens = []

    print()
    print("[CAM_FRONT tokens]")

    for timestamp, path in selected:
        x = np.load(path)

        if x.shape != (18, 32):
            raise RuntimeError(
                f"Unexpected token shape: {path.name} {x.shape}"
            )

        print(
            f"  {timestamp} {x.shape} {x.dtype}"
        )

        tokens.append(x.astype(np.int64))

    return np.stack(tokens, axis=0)[None, ...]


# ============================================================================
# TensorRT Runner
# ============================================================================

class TRTRunner:
    def __init__(self, engine_path):
        self.engine_path = Path(engine_path)
        self.allocations = {}
        self.host_outputs = {}

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"Engine not found: {self.engine_path}"
            )

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: "
                f"{self.engine_path}"
            )

        print("[OK] TensorRT engine deserialized")

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError(
                "Failed to create execution context."
            )

        self.inspect_engine()
        self.allocate()

        result = cuda.cuStreamCreate(0)
        cuda_check(result, "cuStreamCreate")
        self.stream = result[1]

        self.set_addresses()

        print("[OK] CUDA stream created")
        print("[OK] Tensor addresses configured")

    def io_names(self):
        return [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]

    def inspect_engine(self):
        print()
        print("[TRT] Engine I/O")

        for name in self.io_names():
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))

            print(f"  {name}")
            print(f"    mode  : {mode}")
            print(f"    dtype : {dtype}")
            print(f"    shape : {shape}")

    def allocate(self):
        print()
        print("[TRT] Allocating buffers...")

        for name in self.io_names():
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))

            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Dynamic/unresolved tensor shape is not supported: "
                    f"{name} {shape}"
                )

            if is_bf16(dtype):
                itemsize = 2
            else:
                itemsize = np.dtype(
                    trt_to_numpy_dtype(dtype)
                ).itemsize

            nbytes = int(np.prod(shape)) * itemsize
            ptr = cuda_malloc(nbytes)

            self.allocations[name] = ptr

            print(
                f"  [OK] {name} "
                f"shape={shape} "
                f"dtype={dtype} "
                f"nbytes={nbytes}"
            )

            if mode == trt.TensorIOMode.OUTPUT:
                self.host_outputs[name] = make_output_host(
                    shape,
                    dtype,
                )

    def set_addresses(self):
        for name, ptr in self.allocations.items():
            ok = self.context.set_tensor_address(
                name,
                int(ptr),
            )

            if not ok:
                raise RuntimeError(
                    f"Failed to set tensor address: {name}"
                )

    def prepare_inputs(
        self,
        visual_tokens,
        noisy_actions,
        high_level_command,
        diffusion_step,
    ):
        sources = {
            "visual_tokens": visual_tokens,
            "noisy_actions": noisy_actions,
            "high_level_command": high_level_command,
            "diffusion_step": diffusion_step,
        }

        prepared = {}

        for name, source in sources.items():
            if name not in self.allocations:
                raise RuntimeError(
                    f"TensorRT input name '{name}' not found. "
                    f"Available tensors: {self.io_names()}"
                )

            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))

            prepared[name] = make_host_array(
                source,
                shape,
                dtype,
                name,
            )

        return prepared

    def infer_timed(
        self,
        visual_tokens,
        noisy_actions,
        high_level_command,
        diffusion_step,
    ):
        """
        One complete inference step.

        Timing boundaries:
            H2D  = host input copies + synchronization
            TRT  = execute_async_v3 + stream synchronization
            D2H  = output copy + synchronization
            TOTAL = full measured step, including Python bookkeeping
        """
        wall_start = time.perf_counter()

        prepared = self.prepare_inputs(
            visual_tokens,
            noisy_actions,
            high_level_command,
            diffusion_step,
        )

        # -------------------------
        # H2D
        # -------------------------
        h2d_start = time.perf_counter()

        for name, arr in prepared.items():
            cuda_memcpy_htod(
                self.allocations[name],
                arr,
            )

        cuda_check(
            cuda.cuStreamSynchronize(self.stream),
            "cuStreamSynchronize(H2D)",
        )

        h2d_end = time.perf_counter()

        # -------------------------
        # TensorRT GPU execution
        # -------------------------
        trt_start = time.perf_counter()

        ok = self.context.execute_async_v3(self.stream)

        if not ok:
            raise RuntimeError(
                "TensorRT execute_async_v3 failed"
            )

        cuda_check(
            cuda.cuStreamSynchronize(self.stream),
            "cuStreamSynchronize(TRT)",
        )

        trt_end = time.perf_counter()

        # -------------------------
        # D2H
        # -------------------------
        output_name = "action_velocity"

        if output_name not in self.host_outputs:
            raise RuntimeError(
                f"Expected output '{output_name}' not found. "
                f"Outputs: {list(self.host_outputs)}"
            )

        output = self.host_outputs[output_name]

        d2h_start = time.perf_counter()

        cuda_memcpy_dtoh(
            output,
            self.allocations[output_name],
        )

        cuda_check(
            cuda.cuStreamSynchronize(self.stream),
            "cuStreamSynchronize(D2H)",
        )

        d2h_end = time.perf_counter()

        wall_end = time.perf_counter()

        output_dtype = self.engine.get_tensor_dtype(output_name)

        if is_bf16(output_dtype):
            output_f32 = bf16_to_float32(
                np.array(output, copy=True)
            )
        else:
            output_f32 = np.asarray(
                np.array(output, copy=True),
                dtype=np.float32,
            )

        return {
            "output": output_f32,
            "h2d_ms": (h2d_end - h2d_start) * 1000.0,
            "trt_ms": (trt_end - trt_start) * 1000.0,
            "d2h_ms": (d2h_end - d2h_start) * 1000.0,
            "total_ms": (wall_end - wall_start) * 1000.0,
        }

    def cleanup(self):
        print()
        print("[TRT] Cleaning buffers...")

        for name, ptr in list(self.allocations.items()):
            try:
                cuda_free(ptr)
                print(f"  [OK] {name}")
            except Exception as exc:
                print(f"  [WARN] {name}: {exc}")

        self.allocations.clear()

        if hasattr(self, "stream"):
            try:
                cuda_check(
                    cuda.cuStreamDestroy(self.stream),
                    "cuStreamDestroy",
                )
                print("[OK] CUDA stream destroyed")
            except Exception as exc:
                print(f"[WARN] CUDA stream destroy: {exc}")

        self.host_outputs.clear()

        try:
            del self.context
        except Exception:
            pass

        try:
            del self.engine
        except Exception:
            pass

        try:
            del self.runtime
        except Exception:
            pass

        gc.collect()


# ============================================================================
# Euler
# ============================================================================

def run_trt_euler(
    runner,
    visual_tokens,
    initial_action,
    high_level_command,
    initial_diffusion_step,
    warmup_steps,
):
    print()
    print("=" * 80)
    print("Thor TensorRT + Euler")
    print("=" * 80)

    action = initial_action.copy()
    diffusion_step = initial_diffusion_step.copy()

    velocity_history = []
    action_history = [action.copy()]

    timings = []

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    if warmup_steps > 0:
        print()
        print(f"[Warmup] {warmup_steps} TensorRT steps")

        warmup_action = action.copy()
        warmup_t = diffusion_step.copy()

        for _ in range(warmup_steps):
            result = runner.infer_timed(
                visual_tokens,
                warmup_action,
                high_level_command,
                warmup_t,
            )

            warmup_action = (
                warmup_action
                + DELTA_T * result["output"]
            ).astype(np.float32)

            warmup_t = (
                warmup_t + DELTA_T
            ).astype(np.float32)

    # ------------------------------------------------------------------
    # Actual 10-step Euler loop
    # ------------------------------------------------------------------
    print()
    print(f"[Benchmark] Euler steps = {NUM_EULER_STEPS}")

    for step in range(NUM_EULER_STEPS):
        print()
        print(
            f"Step {step + 1}/{NUM_EULER_STEPS}"
        )

        print(
            f"  t = "
            f"{float(diffusion_step.reshape(-1)[0]):.6f}"
        )

        result = runner.infer_timed(
            visual_tokens,
            action,
            high_level_command,
            diffusion_step,
        )

        velocity = result["output"]

        velocity_history.append(velocity.copy())

        action = (
            action + DELTA_T * velocity
        ).astype(np.float32)

        action_history.append(action.copy())

        timings.append({
            "step": step + 1,
            "h2d_ms": result["h2d_ms"],
            "trt_ms": result["trt_ms"],
            "d2h_ms": result["d2h_ms"],
            "total_ms": result["total_ms"],
        })

        print(
            f"  velocity: "
            f"min={velocity.min():.6f} "
            f"max={velocity.max():.6f} "
            f"mean={velocity.mean():.8f}"
        )

        print(
            f"  action: "
            f"min={action.min():.6f} "
            f"max={action.max():.6f} "
            f"mean={action.mean():.8f}"
        )

        print(
            f"  timing: "
            f"H2D={result['h2d_ms']:.3f} ms, "
            f"TRT={result['trt_ms']:.3f} ms, "
            f"D2H={result['d2h_ms']:.3f} ms, "
            f"TOTAL={result['total_ms']:.3f} ms"
        )

        diffusion_step = (
            diffusion_step + DELTA_T
        ).astype(np.float32)

    return (
        np.stack(velocity_history, axis=0),
        np.stack(action_history, axis=0),
        action.copy(),
        timings,
    )


# ============================================================================
# Comparison
# ============================================================================

def compare(
    pt_velocity,
    thor_velocity,
    pt_action,
    thor_action,
    pt_final,
    thor_final,
):
    print()
    print("=" * 80)
    print("Numerical Comparison")
    print("=" * 80)

    velocity_error = np.abs(
        pt_velocity - thor_velocity
    )

    action_error = np.abs(
        pt_action - thor_action
    )

    final_error = np.abs(
        pt_final - thor_final
    )

    print()
    print("[Velocity history]")
    print(
        f"  max abs error : "
        f"{velocity_error.max():.10e}"
    )
    print(
        f"  mean abs error: "
        f"{velocity_error.mean():.10e}"
    )
    print(
        f"  RMSE          : "
        f"{np.sqrt(np.mean((pt_velocity - thor_velocity) ** 2)):.10e}"
    )

    print()
    print("[Action history]")
    print(
        f"  max abs error : "
        f"{action_error.max():.10e}"
    )
    print(
        f"  mean abs error: "
        f"{action_error.mean():.10e}"
    )
    print(
        f"  RMSE          : "
        f"{np.sqrt(np.mean((pt_action - thor_action) ** 2)):.10e}"
    )

    print()
    print("[Final trajectory]")
    print(
        f"  max abs error : "
        f"{final_error.max():.10e}"
    )
    print(
        f"  mean abs error: "
        f"{final_error.mean():.10e}"
    )
    print(
        f"  RMSE          : "
        f"{np.sqrt(np.mean((pt_final - thor_final) ** 2)):.10e}"
    )

    return {
        "velocity_max_abs": float(velocity_error.max()),
        "velocity_mean_abs": float(velocity_error.mean()),
        "velocity_rmse": float(
            np.sqrt(np.mean((pt_velocity - thor_velocity) ** 2))
        ),
        "action_max_abs": float(action_error.max()),
        "action_mean_abs": float(action_error.mean()),
        "action_rmse": float(
            np.sqrt(np.mean((pt_action - thor_action) ** 2))
        ),
        "final_max_abs": float(final_error.max()),
        "final_mean_abs": float(final_error.mean()),
        "final_rmse": float(
            np.sqrt(np.mean((pt_final - thor_final) ** 2))
        ),
    }


# ============================================================================
# Timing summary
# ============================================================================

def summarize_timing(timings):
    keys = [
        "h2d_ms",
        "trt_ms",
        "d2h_ms",
        "total_ms",
    ]

    print()
    print("=" * 80)
    print("Latency Summary")
    print("=" * 80)

    print()
    print(
        f"{'Metric':<14}"
        f"{'Mean':>12}"
        f"{'Median':>12}"
        f"{'Min':>12}"
        f"{'Max':>12}"
    )
    print("-" * 62)

    summary = {}

    for key in keys:
        values = np.array(
            [x[key] for x in timings],
            dtype=np.float64,
        )

        item = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "sum": float(np.sum(values)),
        }

        summary[key] = item

        print(
            f"{key:<14}"
            f"{item['mean']:>12.3f}"
            f"{item['median']:>12.3f}"
            f"{item['min']:>12.3f}"
            f"{item['max']:>12.3f}"
        )

    print()
    print(
        f"Total 10-step TRT GPU time : "
        f"{summary['trt_ms']['sum']:.3f} ms"
    )
    print(
        f"Total 10-step E2E time     : "
        f"{summary['total_ms']['sum']:.3f} ms"
    )

    print()
    print(
        f"Effective Euler frequency "
        f"(10-step E2E): "
        f"{1000.0 / summary['total_ms']['sum']:.3f} Hz"
    )

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run VaVAM Thor visual-token -> action-velocity "
            "-> Euler validation with a selected TensorRT engine."
        )
    )

    parser.add_argument(
        "engine",
        type=Path,
        help="TensorRT engine path",
    )

    parser.add_argument(
        "reference",
        type=Path,
        help="12h/12i reference NPZ",
    )

    parser.add_argument(
        "token_dir",
        type=Path,
        help="CAM_FRONT .npy token directory",
    )

    parser.add_argument(
        "--scene",
        required=True,
        help="nuScenes scene id",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="8-frame CAM_FRONT window index",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup TensorRT steps before the 10 measured steps",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "VaVAM Thor Visual Tokens -> "
        "Action Velocity / Euler Benchmark"
    )
    print("=" * 80)

    print()
    print(f"Engine    : {args.engine}")
    print(f"Reference : {args.reference}")
    print(f"Tokens    : {args.token_dir}")
    print(f"Scene     : {args.scene}")
    print(f"Window    : {args.window}")
    print(f"Warmup    : {args.warmup}")
    print(f"Euler     : {NUM_EULER_STEPS} steps")

    # ------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------
    init_cuda()

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------
    ref = load_reference(args.reference)

    if ref["scene_id"] != args.scene:
        raise RuntimeError(
            "Scene mismatch:\n"
            f"  NPZ    : {ref['scene_id']}\n"
            f"  CLI    : {args.scene}"
        )

    if ref["window_index"] != args.window:
        raise RuntimeError(
            "Window mismatch:\n"
            f"  NPZ    : {ref['window_index']}\n"
            f"  CLI    : {args.window}"
        )

    # ------------------------------------------------------------------
    # Actual CAM_FRONT .npy tokens
    # ------------------------------------------------------------------
    files = load_scene_tokens(
        args.token_dir,
        args.scene,
    )

    print()
    print(f"[OK] Scene frames: {len(files)}")

    visual_tokens_npy = build_visual_tokens(
        files,
        args.window,
    )

    print()
    print("[Visual Tokens from .npy]")
    print(f"  shape : {visual_tokens_npy.shape}")
    print(f"  dtype : {visual_tokens_npy.dtype}")
    print(f"  min   : {visual_tokens_npy.min()}")
    print(f"  max   : {visual_tokens_npy.max()}")

    # ------------------------------------------------------------------
    # Verify actual .npy tokens against reference NPZ
    # ------------------------------------------------------------------
    if visual_tokens_npy.shape != ref["visual_tokens"].shape:
        raise RuntimeError(
            "Visual token shape mismatch:\n"
            f"  .npy : {visual_tokens_npy.shape}\n"
            f"  .npz : {ref['visual_tokens'].shape}"
        )

    token_diff = np.abs(
        visual_tokens_npy.astype(np.int64)
        - ref["visual_tokens"].astype(np.int64)
    )

    print()
    print("[Visual Token Verification]")
    print(
        f"  max abs difference : {token_diff.max()}"
    )

    if token_diff.max() != 0:
        raise RuntimeError(
            "CAM_FRONT .npy tokens do not match "
            "12h NPZ visual_tokens."
        )

    print(
        "[PASS] .npy visual tokens match "
        ".npz reference exactly"
    )

    # ------------------------------------------------------------------
    # TensorRT
    # ------------------------------------------------------------------
    runner = None

    try:
        runner = TRTRunner(args.engine)

        (
            thor_velocity,
            thor_action,
            thor_final,
            timings,
        ) = run_trt_euler(
            runner,
            visual_tokens_npy,
            ref["initial_action"],
            ref["high_level_command"],
            ref["initial_diffusion_step"],
            args.warmup,
        )

    finally:
        if runner is not None:
            runner.cleanup()

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    timing_summary = summarize_timing(timings)

    # ------------------------------------------------------------------
    # Numerical comparison
    # ------------------------------------------------------------------
    accuracy = compare(
        ref["pytorch_velocity"],
        thor_velocity,
        ref["pytorch_action"],
        thor_action,
        ref["pytorch_final"],
        thor_final,
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print()
    print(f"Engine: {args.engine.name}")
    print()
    print(
        f"Per-step TRT GPU mean : "
        f"{timing_summary['trt_ms']['mean']:.3f} ms"
    )
    print(
        f"Per-step E2E mean     : "
        f"{timing_summary['total_ms']['mean']:.3f} ms"
    )
    print(
        f"10-step TRT GPU total : "
        f"{timing_summary['trt_ms']['sum']:.3f} ms"
    )
    print(
        f"10-step E2E total     : "
        f"{timing_summary['total_ms']['sum']:.3f} ms"
    )

    print()
    print(
        f"Final trajectory max abs error : "
        f"{accuracy['final_max_abs']:.10e}"
    )
    print(
        f"Final trajectory mean abs error: "
        f"{accuracy['final_mean_abs']:.10e}"
    )
    print(
        f"Final trajectory RMSE          : "
        f"{accuracy['final_rmse']:.10e}"
    )

    print()

    if accuracy["final_max_abs"] <= ATOL:
        print(
            "[PASS] Final trajectory matches "
            "the PC PyTorch reference within 1e-4."
        )
    elif accuracy["final_max_abs"] <= 1e-3:
        print(
            "[PASS-WARN] Final trajectory max error "
            "is within 1e-3."
        )
    else:
        print(
            "[INFO] Final trajectory differs from "
            "the PC reference by more than 1e-3."
        )

    print()
    print(
        "NOTE: The measured E2E here is PER DIFFUSION STEP "
        "plus the 10-step Euler total. It does not include "
        "Vision Encoder or Prefill."
    )


if __name__ == "__main__":
    main()
