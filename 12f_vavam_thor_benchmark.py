#!/usr/bin/env python3
"""
VaVAM Thor B/L Precision + Latency Benchmark

Benchmarks:
    VaVAM-B / VaVAM-L
    FP32 + TF32 allowed
    FP32 + TF32 disabled
    FP16
    BF16

Measures:
    - H2D latency
    - TensorRT GPU execution latency
    - D2H latency
    - End-to-end latency
    - Output accuracy vs PC PyTorch reference

Reference:
    thor_inference_step_reference.npz

Expected reference keys:
    visual_tokens
    noisy_actions
    high_level_command
    diffusion_step
    action_velocity

IMPORTANT:
    This script assumes the engine input/output tensor names contain
    the corresponding logical names. If your existing 12e script uses
    exact names that differ, adjust INPUT_ALIASES / OUTPUT_ALIASES below.
"""

import gc
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import driver


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path("/home/delta_drc/vblkdev2/VaVAM_Thor")

REFERENCE_PATH = ROOT / "thor_inference_step_reference.npz"

# Use the actual ONNX sources currently used on Thor.
# Existing FP32 engines from your current setup are retained.
ENGINES = {
    "B": {
        "FP32_TF32": ROOT / "vavam_joint_inference_step_B_fp32.engine",
        "FP32_noTF32": ROOT / "vavam_joint_inference_step_B_fp32_noTF32.engine",
        "FP16": ROOT / "vavam_joint_inference_step_B_fp16.engine",
        "BF16": ROOT / "vavam_joint_inference_step_B_bf16.engine",
    },
    "L": {
        "FP32_TF32": ROOT / "vavam_joint_inference_step_L_fp32.engine",
        "FP32_noTF32": ROOT / "vavam_joint_inference_step_L_fp32_noTF32.engine",
        "FP16": ROOT / "vavam_joint_inference_step_L_fp16.engine",
        "BF16": ROOT / "vavam_joint_inference_step_L_bf16.engine",
    },
}

WARMUP = 20
ITERATIONS = 100

# Ignore the first few measured samples when calculating stable statistics.
# Set to 0 if you want every iteration included.
DROP_FIRST = 5

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================================
# Logical tensor name aliases
# ============================================================================

INPUT_ALIASES = {
    "visual_tokens": [
        "visual_tokens",
        "visual_token",
    ],
    "noisy_actions": [
        "noisy_actions",
        "noisy_action",
    ],
    "high_level_command": [
        "high_level_command",
        "high_level_commands",
        "command",
    ],
    "diffusion_step": [
        "diffusion_step",
        "diffusion_steps",
        "timestep",
        "timesteps",
    ],
}

OUTPUT_ALIASES = [
    "action_velocity",
    "action_velocities",
    "action",
]


# ============================================================================
# CUDA helpers
# ============================================================================

def cuda_check(result, name):
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {err}")

    return result


def cuda_malloc(nbytes):
    result = driver.cuMemAlloc(int(nbytes))
    cuda_check(result, "cuMemAlloc")
    return result[1]


def cuda_free(ptr):
    if ptr is None:
        return
    result = driver.cuMemFree(ptr)
    cuda_check(result, "cuMemFree")


def cuda_memcpy_htod(ptr, host_array):
    arr = np.ascontiguousarray(host_array)
    result = driver.cuMemcpyHtoD(
        int(ptr),
        arr,
        int(arr.nbytes),
    )
    cuda_check(result, "cuMemcpyHtoD")


def cuda_memcpy_dtoh(host_array, ptr):
    arr = np.ascontiguousarray(host_array)
    result = driver.cuMemcpyDtoH(
        arr,
        int(ptr),
        int(arr.nbytes),
    )
    cuda_check(result, "cuMemcpyDtoH")


def cuda_sync():
    result = driver.cuCtxSynchronize()
    cuda_check(result, "cuCtxSynchronize")


# ============================================================================
# TensorRT dtype helpers
# ============================================================================

def is_bf16_dtype(dtype):
    return hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16


def trt_dtype_name(dtype):
    try:
        return str(dtype)
    except Exception:
        return repr(dtype)


def trt_dtype_to_numpy(dtype):
    if dtype == trt.DataType.FLOAT:
        return np.float32

    if dtype == trt.DataType.HALF:
        return np.float16

    if is_bf16_dtype(dtype):
        # Do NOT assume NumPy has native bfloat16.
        # BF16 is handled separately by make_host_array().
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


def make_host_array(source, shape, dtype, tensor_name):
    """
    Create a C-contiguous host array suitable for the TensorRT binding.

    BF16 handling:
      - If NumPy provides native bfloat16, use it.
      - Otherwise use uint16 storage containing the raw BF16 bits.

    CUDA copies bytes only, so the uint16 representation preserves the
    exact BF16 bit pattern without requiring NumPy bfloat16 support.
    """

    source = np.asarray(source)

    if is_bf16_dtype(dtype):
        # Native NumPy bfloat16, if available.
        try:
            bf16_dtype = np.dtype("bfloat16")
            arr = np.asarray(source, dtype=bf16_dtype)
            arr = np.ascontiguousarray(arr)

            if tuple(arr.shape) != tuple(shape):
                raise RuntimeError(
                    f"Shape mismatch for BF16 tensor '{tensor_name}': "
                    f"engine={shape}, input={arr.shape}"
                )

            return arr

        except TypeError:
            # No native NumPy BF16: create raw BF16 uint16 representation.
            f32 = np.asarray(source, dtype=np.float32)

            if tuple(f32.shape) != tuple(shape):
                raise RuntimeError(
                    f"Shape mismatch for BF16 tensor '{tensor_name}': "
                    f"engine={shape}, input={f32.shape}"
                )

            f32 = np.ascontiguousarray(f32)

            # BF16 is the upper 16 bits of IEEE FP32.
            # Round-to-nearest-even before truncation.
            u32 = f32.view(np.uint32)
            rounding = ((u32 >> 16) & 1) + 0x7FFF
            bf16_bits = ((u32 + rounding) >> 16).astype(
                np.uint16,
                copy=False,
            )

            return np.ascontiguousarray(bf16_bits)

    np_dtype = trt_dtype_to_numpy(dtype)

    arr = np.asarray(source, dtype=np_dtype)
    arr = np.ascontiguousarray(arr)

    if tuple(arr.shape) != tuple(shape):
        raise RuntimeError(
            f"Shape mismatch for tensor '{tensor_name}': "
            f"engine={shape}, input={arr.shape}"
        )

    return arr


def make_output_host_array(shape, dtype):
    """
    Allocate host storage for an output tensor.

    For BF16 without native NumPy support, use uint16 raw BF16 storage.
    """
    if is_bf16_dtype(dtype):
        try:
            return np.empty(shape, dtype=np.dtype("bfloat16"))
        except TypeError:
            return np.empty(shape, dtype=np.uint16)

    np_dtype = trt_dtype_to_numpy(dtype)
    return np.empty(shape, dtype=np_dtype)


def bf16_host_to_float32(arr):
    """
    Convert a BF16 host representation (native bfloat16 or uint16 raw bits)
    to float32 for accuracy comparison.
    """
    if arr.dtype == np.uint16:
        u32 = arr.astype(np.uint32) << 16
        return u32.view(np.float32)

    return np.asarray(arr, dtype=np.float32)


def output_to_float32(arr, dtype):
    if is_bf16_dtype(dtype):
        return bf16_host_to_float32(arr)

    return np.asarray(arr, dtype=np.float32)


# ============================================================================
# TensorRT engine utilities
# ============================================================================

def normalize_name(name):
    return (
        str(name)
        .lower()
        .replace("_", "")
        .replace(".", "")
        .replace("-", "")
    )


def get_io_tensor_names(engine):
    if hasattr(engine, "num_io_tensors"):
        return [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
        ]

    return [
        engine.get_binding_name(i)
        for i in range(engine.num_bindings)
    ]


def get_tensor_mode(engine, name):
    if hasattr(engine, "get_tensor_mode"):
        return engine.get_tensor_mode(name)

    index = engine.get_binding_index(name)
    return (
        trt.TensorIOMode.INPUT
        if engine.binding_is_input(index)
        else trt.TensorIOMode.OUTPUT
    )


def get_tensor_dtype(engine, name):
    if hasattr(engine, "get_tensor_dtype"):
        return engine.get_tensor_dtype(name)

    index = engine.get_binding_index(name)
    return engine.get_binding_dtype(index)


def get_tensor_shape(engine, name):
    if hasattr(engine, "get_tensor_shape"):
        return tuple(engine.get_tensor_shape(name))

    index = engine.get_binding_index(name)
    return tuple(engine.get_binding_shape(index))


def find_tensor_name(engine, aliases, is_input):
    candidates = []

    for name in get_io_tensor_names(engine):
        mode = get_tensor_mode(engine, name)

        if is_input and mode != trt.TensorIOMode.INPUT:
            continue

        if not is_input and mode != trt.TensorIOMode.OUTPUT:
            continue

        candidates.append(name)

    normalized_candidates = {
        name: normalize_name(name)
        for name in candidates
    }

    # Exact normalized match first.
    for alias in aliases:
        alias_norm = normalize_name(alias)

        for name, norm in normalized_candidates.items():
            if norm == alias_norm:
                return name

    # Then substring match.
    for alias in aliases:
        alias_norm = normalize_name(alias)

        for name, norm in normalized_candidates.items():
            if alias_norm in norm:
                return name

    return None


def print_engine_info(engine, engine_path):
    print()
    print("=" * 90)
    print(f"Engine: {engine_path.name}")
    print("=" * 90)

    for name in get_io_tensor_names(engine):
        mode = get_tensor_mode(engine, name)
        dtype = get_tensor_dtype(engine, name)
        shape = get_tensor_shape(engine, name)

        io = (
            "INPUT "
            if mode == trt.TensorIOMode.INPUT
            else "OUTPUT"
        )

        print(
            f"{io:7s} "
            f"{name:35s} "
            f"shape={str(shape):25s} "
            f"dtype={trt_dtype_name(dtype)}"
        )


# ============================================================================
# TensorRT wrapper
# ============================================================================

class TensorRTEngine:
    def __init__(self, engine_path):
        self.engine_path = Path(engine_path)

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"Engine not found: {self.engine_path}"
            )

        print()
        print(f"Loading engine: {self.engine_path}")

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine: {self.engine_path}"
            )

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError(
                f"Failed to create TensorRT execution context: "
                f"{self.engine_path}"
            )

        print_engine_info(self.engine, self.engine_path)

        self.input_names = {}
        for logical_name, aliases in INPUT_ALIASES.items():
            tensor_name = find_tensor_name(
                self.engine,
                aliases,
                is_input=True,
            )

            if tensor_name is None:
                available_inputs = [
                    n for n in get_io_tensor_names(self.engine)
                    if get_tensor_mode(self.engine, n) == trt.TensorIOMode.INPUT
                ]
                raise RuntimeError(
                    f"Cannot find TensorRT input for "
                    f"'{logical_name}'. "
                    f"Available inputs: {available_inputs}"
                )

            self.input_names[logical_name] = tensor_name

        self.output_name = find_tensor_name(
            self.engine,
            OUTPUT_ALIASES,
            is_input=False,
        )

        if self.output_name is None:
            available_outputs = [
                n for n in get_io_tensor_names(self.engine)
                if get_tensor_mode(self.engine, n) != trt.TensorIOMode.INPUT
            ]
            raise RuntimeError(
                "Cannot find TensorRT output 'action_velocity'. "
                f"Available outputs: {available_outputs}"
            )

        print()
        print("Logical tensor mapping:")
        for logical, actual in self.input_names.items():
            print(f"  {logical:22s} -> {actual}")
        print(f"  {'action_velocity':22s} -> {self.output_name}")

        self.device_buffers = {}
        self.output_host = None

        self._allocate_buffers()

    def _allocate_buffers(self):
        for name in get_io_tensor_names(self.engine):
            shape = get_tensor_shape(self.engine, name)
            dtype = get_tensor_dtype(self.engine, name)

            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Dynamic/unresolved shape for '{name}': {shape}. "
                    "This benchmark expects fixed shapes like the existing 12e."
                )

            # Size in bytes is determined from the TRT dtype.
            if is_bf16_dtype(dtype):
                itemsize = 2
            else:
                np_dtype = trt_dtype_to_numpy(dtype)
                itemsize = np.dtype(np_dtype).itemsize

            nbytes = int(np.prod(shape)) * itemsize

            ptr = cuda_malloc(nbytes)

            self.device_buffers[name] = {
                "ptr": ptr,
                "shape": shape,
                "dtype": dtype,
                "nbytes": nbytes,
            }

            if name == self.output_name:
                self.output_host = make_output_host_array(
                    shape,
                    dtype,
                )

    def prepare_inputs(self, inputs):
        prepared = {}

        for logical_name, tensor_name in self.input_names.items():
            info = self.device_buffers[tensor_name]

            prepared[logical_name] = make_host_array(
                inputs[logical_name],
                info["shape"],
                info["dtype"],
                tensor_name,
            )

        return prepared

    def set_tensor_addresses(self):
        if not hasattr(self.context, "set_tensor_address"):
            return

        for name, info in self.device_buffers.items():
            ok = self.context.set_tensor_address(
                name,
                int(info["ptr"]),
            )

            if ok is False:
                raise RuntimeError(
                    f"set_tensor_address failed for '{name}'"
                )

    def execute_trt(self):
        """
        Launch only TensorRT execution.

        Returns after GPU completion so the measured interval represents
        TensorRT GPU execution, not asynchronous enqueue time.
        """
        if hasattr(self.context, "set_tensor_address"):
            self.set_tensor_addresses()

            ok = self.context.execute_async_v3(0)

            if not ok:
                raise RuntimeError(
                    "TensorRT execute_async_v3 failed"
                )

        else:
            bindings = [0] * self.engine.num_bindings

            for name, info in self.device_buffers.items():
                index = self.engine.get_binding_index(name)
                bindings[index] = int(info["ptr"])

            ok = self.context.execute_async_v2(
                bindings=bindings,
                stream_handle=0,
            )

            if not ok:
                raise RuntimeError(
                    "TensorRT execute_async_v2 failed"
                )

        cuda_sync()

    def run_once(self, inputs):
        """
        One complete H2D -> TRT -> D2H execution.

        Returns:
            output_float32,
            h2d_ms,
            trt_ms,
            d2h_ms,
            e2e_ms
        """
        t0 = time.perf_counter()

        prepared = self.prepare_inputs(inputs)

        # H2D
        h2d_start = time.perf_counter()

        for logical_name, tensor_name in self.input_names.items():
            cuda_memcpy_htod(
                self.device_buffers[tensor_name]["ptr"],
                prepared[logical_name],
            )

        cuda_sync()

        h2d_end = time.perf_counter()

        # TensorRT
        trt_start = time.perf_counter()

        self.execute_trt()

        trt_end = time.perf_counter()

        # D2H
        d2h_start = time.perf_counter()

        cuda_memcpy_dtoh(
            self.output_host,
            self.device_buffers[self.output_name]["ptr"],
        )

        cuda_sync()

        d2h_end = time.perf_counter()

        t1 = time.perf_counter()

        output_dtype = get_tensor_dtype(
            self.engine,
            self.output_name,
        )

        output_float32 = output_to_float32(
            np.array(self.output_host, copy=True),
            output_dtype,
        )

        return (
            output_float32,
            (h2d_end - h2d_start) * 1000.0,
            (trt_end - trt_start) * 1000.0,
            (d2h_end - d2h_start) * 1000.0,
            (t1 - t0) * 1000.0,
        )

    def close(self):
        for info in self.device_buffers.values():
            try:
                cuda_free(info["ptr"])
            except Exception:
                pass

        self.device_buffers.clear()

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


# ============================================================================
# Accuracy
# ============================================================================

def compare_outputs(reference, test):
    reference = np.asarray(reference, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)

    if reference.shape != test.shape:
        raise RuntimeError(
            "Output shape mismatch:\n"
            f"  reference = {reference.shape}\n"
            f"  test      = {test.shape}"
        )

    diff = np.abs(reference - test)

    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))

    rmse = float(
        np.sqrt(
            np.mean(
                (reference - test) ** 2
            )
        )
    )

    denom = np.maximum(np.abs(reference), 1e-8)

    max_rel = float(
        np.max(diff / denom)
    )

    max_index = np.unravel_index(
        np.argmax(diff),
        diff.shape,
    )

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "max_rel": max_rel,
        "max_index": max_index,
        "reference_at_max": float(reference[max_index]),
        "test_at_max": float(test[max_index]),
    }


# ============================================================================
# Statistics
# ============================================================================

def stats(values):
    x = np.asarray(values, dtype=np.float64)

    if x.size == 0:
        raise RuntimeError("No benchmark samples")

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


# ============================================================================
# Reference
# ============================================================================

def load_reference():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_PATH}"
        )

    data = np.load(REFERENCE_PATH)

    required = [
        "visual_tokens",
        "noisy_actions",
        "high_level_command",
        "diffusion_step",
        "action_velocity",
    ]

    for key in required:
        if key not in data:
            raise KeyError(
                f"Missing reference key: {key}"
            )

    print()
    print("=" * 90)
    print("Reference")
    print("=" * 90)

    for key in required:
        x = data[key]
        print(
            f"{key:25s} "
            f"shape={str(x.shape):25s} "
            f"dtype={x.dtype}"
        )

    return data


# ============================================================================
# Benchmark one engine
# ============================================================================

def benchmark_engine(
    model_name,
    precision_name,
    engine_path,
    inputs,
    reference_output,
):
    print()
    print()
    print("#" * 90)
    print(f"# {model_name} / {precision_name}")
    print("#" * 90)

    if not engine_path.exists():
        print(f"SKIP: engine not found: {engine_path}")

        return {
            "model": model_name,
            "precision": precision_name,
            "status": "SKIP",
        }

    engine = None

    try:
        engine = TensorRTEngine(engine_path)

        # ------------------------------------------------------------
        # Warmup
        # ------------------------------------------------------------

        print()
        print(
            f"Warmup={WARMUP}, "
            f"Iterations={ITERATIONS}, "
            f"Drop first={DROP_FIRST}"
        )

        for i in range(WARMUP):
            engine.run_once(inputs)

        cuda_sync()

        # ------------------------------------------------------------
        # Benchmark
        # ------------------------------------------------------------

        h2d = []
        trt_gpu = []
        d2h = []
        e2e = []

        last_output = None

        for _ in range(ITERATIONS):
            (
                last_output,
                h2d_ms,
                trt_ms,
                d2h_ms,
                e2e_ms,
            ) = engine.run_once(inputs)

            h2d.append(h2d_ms)
            trt_gpu.append(trt_ms)
            d2h.append(d2h_ms)
            e2e.append(e2e_ms)

        # ------------------------------------------------------------
        # Remove startup samples if requested
        # ------------------------------------------------------------

        start = min(DROP_FIRST, len(h2d))

        h2d = h2d[start:]
        trt_gpu = trt_gpu[start:]
        d2h = d2h[start:]
        e2e = e2e[start:]

        h2d_s = stats(h2d)
        trt_s = stats(trt_gpu)
        d2h_s = stats(d2h)
        e2e_s = stats(e2e)

        # ------------------------------------------------------------
        # Accuracy
        # ------------------------------------------------------------

        accuracy = compare_outputs(
            reference_output,
            last_output,
        )

        # ------------------------------------------------------------
        # Print
        # ------------------------------------------------------------

        print()
        print("-" * 90)
        print(f"{model_name} / {precision_name}")
        print("-" * 90)

        print()
        print("Latency [ms]")
        print(
            f"  H2D          mean={h2d_s['mean']:.3f}  "
            f"median={h2d_s['median']:.3f}  "
            f"P95={h2d_s['p95']:.3f}"
        )
        print(
            f"  TensorRT GPU mean={trt_s['mean']:.3f}  "
            f"median={trt_s['median']:.3f}  "
            f"P95={trt_s['p95']:.3f}"
        )
        print(
            f"  D2H          mean={d2h_s['mean']:.3f}  "
            f"median={d2h_s['median']:.3f}  "
            f"P95={d2h_s['p95']:.3f}"
        )
        print(
            f"  E2E          mean={e2e_s['mean']:.3f}  "
            f"median={e2e_s['median']:.3f}  "
            f"P95={e2e_s['p95']:.3f}"
        )

        print()
        print("Accuracy")
        print(
            f"  max abs error : "
            f"{accuracy['max_abs']:.10e}"
        )
        print(
            f"  mean abs error: "
            f"{accuracy['mean_abs']:.10e}"
        )
        print(
            f"  RMSE          : "
            f"{accuracy['rmse']:.10e}"
        )
        print(
            f"  max rel error : "
            f"{accuracy['max_rel']:.10e}"
        )
        print(
            f"  max error idx : "
            f"{accuracy['max_index']}"
        )
        print(
            f"  reference     : "
            f"{accuracy['reference_at_max']:.10e}"
        )
        print(
            f"  TensorRT      : "
            f"{accuracy['test_at_max']:.10e}"
        )

        return {
            "model": model_name,
            "precision": precision_name,
            "status": "OK",

            "h2d_mean_ms": h2d_s["mean"],
            "h2d_median_ms": h2d_s["median"],
            "h2d_p95_ms": h2d_s["p95"],

            "trt_mean_ms": trt_s["mean"],
            "trt_median_ms": trt_s["median"],
            "trt_p95_ms": trt_s["p95"],

            "d2h_mean_ms": d2h_s["mean"],
            "d2h_median_ms": d2h_s["median"],
            "d2h_p95_ms": d2h_s["p95"],

            "e2e_mean_ms": e2e_s["mean"],
            "e2e_median_ms": e2e_s["median"],
            "e2e_p95_ms": e2e_s["p95"],

            "max_abs_error": accuracy["max_abs"],
            "mean_abs_error": accuracy["mean_abs"],
            "rmse": accuracy["rmse"],
            "max_rel_error": accuracy["max_rel"],
            "max_error_index": accuracy["max_index"],
        }

    except Exception as exc:
        print()
        print(
            f"ERROR: {model_name} / {precision_name}"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "model": model_name,
            "precision": precision_name,
            "status": "ERROR",
            "error": str(exc),
        }

    finally:
        if engine is not None:
            engine.close()

        gc.collect()


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 90)
    print("VaVAM Thor B/L Precision + Latency Benchmark")
    print("=" * 90)

    print()
    print(f"ROOT      : {ROOT}")
    print(f"REFERENCE : {REFERENCE_PATH}")
    print(f"WARMUP    : {WARMUP}")
    print(f"ITERATIONS: {ITERATIONS}")
    print(f"DROP_FIRST: {DROP_FIRST}")

    # ------------------------------------------------------------
    # CUDA initialization
    # ------------------------------------------------------------

    cuda_check(
        driver.cuInit(0),
        "cuInit",
    )

    result = driver.cuDeviceGet(0)
    cuda_check(result, "cuDeviceGet")
    device = result[1]

    result = driver.cuDeviceGetName(device)
    cuda_check(result, "cuDeviceGetName")

    device_name = result[1]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    print()
    print(f"CUDA Device: {device_name}")

    # ------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------

    reference = load_reference()

    inputs = {
        "visual_tokens": reference["visual_tokens"],
        "noisy_actions": reference["noisy_actions"],
        "high_level_command": reference["high_level_command"],
        "diffusion_step": reference["diffusion_step"],
    }

    reference_output = reference["action_velocity"]

    # ------------------------------------------------------------
    # Run all 8 configurations
    # ------------------------------------------------------------

    results = []

    for model_name in ("B", "L"):
        for precision_name in (
            "FP32_TF32",
            "FP32_noTF32",
            "FP16",
            "BF16",
        ):
            result = benchmark_engine(
                model_name=model_name,
                precision_name=precision_name,
                engine_path=ENGINES[model_name][precision_name],
                inputs=inputs,
                reference_output=reference_output,
            )

            results.append(result)

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print()
    print()
    print("=" * 150)
    print("FINAL SUMMARY")
    print("=" * 150)

    print()
    print(
        f"{'Model':<7}"
        f"{'Precision':<16}"
        f"{'Status':<8}"
        f"{'H2D':>10}"
        f"{'TRT GPU':>12}"
        f"{'D2H':>10}"
        f"{'E2E':>10}"
        f"{'MaxAbs':>15}"
        f"{'RMSE':>15}"
    )

    print("-" * 150)

    for r in results:
        if r["status"] != "OK":
            print(
                f"{r['model']:<7}"
                f"{r['precision']:<16}"
                f"{r['status']:<8}"
            )
            continue

        print(
            f"{r['model']:<7}"
            f"{r['precision']:<16}"
            f"{r['status']:<8}"
            f"{r['h2d_mean_ms']:>10.3f}"
            f"{r['trt_mean_ms']:>12.3f}"
            f"{r['d2h_mean_ms']:>10.3f}"
            f"{r['e2e_mean_ms']:>10.3f}"
            f"{r['max_abs_error']:>15.3e}"
            f"{r['rmse']:>15.3e}"
        )

    print()
    print("=" * 150)

    # ------------------------------------------------------------
    # Interpretation reminder
    # ------------------------------------------------------------

    print()
    print("Notes:")
    print("  FP32_TF32  = FP32 engine with TF32 allowed.")
    print("  FP32_noTF32= FP32 engine with TF32 disabled.")
    print("  FP16       = FP16-enabled TensorRT engine.")
    print("  BF16       = BF16-enabled TensorRT engine.")
    print()
    print(
        "  TensorRT GPU latency is measured from launch until "
        "GPU synchronization completes."
    )
    print(
        "  E2E latency includes host preparation + H2D + TensorRT + D2H "
        "and Python timing overhead."
    )


if __name__ == "__main__":
    main()
