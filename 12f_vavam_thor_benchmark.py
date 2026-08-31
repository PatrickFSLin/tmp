#!/usr/bin/env python3

"""
VaVAM Thor Benchmark
====================

Benchmark:
    VaVAM-B / VaVAM-L
    FP32 + TF32
    FP32 + no-TF32
    FP16
    BF16

Measures:
    1. TensorRT GPU latency
    2. End-to-end latency (H2D + TensorRT + D2H)
    3. Output accuracy vs PyTorch reference

Expected reference:
    thor_inference_step_reference.npz

Expected keys:
    visual_tokens
    noisy_actions
    high_level_command
    diffusion_step
    action_velocity
"""

import os
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

from cuda.bindings import driver


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/delta_drc/vblkdev2/VaVAM_Thor")

REFERENCE_PATH = ROOT / "thor_inference_step_reference.npz"


# ============================================================
# Engine configuration
# ============================================================

ENGINES = {
    "B": {
        "FP32_TF32": ROOT / "vavam_joint_inference_step_B_fp32.engine",
        "FP32":      ROOT / "vavam_joint_inference_step_B_fp32_noTF32.engine",
        "FP16":      ROOT / "vavam_joint_inference_step_B_fp16.engine",
        "BF16":      ROOT / "vavam_joint_inference_step_B_bf16.engine",
    },

    "L": {
        "FP32_TF32": ROOT / "vavam_joint_inference_step_L_fp32.engine",
        "FP32":      ROOT / "vavam_joint_inference_step_L_fp32_noTF32.engine",
        "FP16":      ROOT / "vavam_joint_inference_step_L_fp16.engine",
        "BF16":      ROOT / "vavam_joint_inference_step_L_bf16.engine",
    },
}


# ============================================================
# Benchmark parameters
# ============================================================

WARMUP = 20
ITERATIONS = 100


# ============================================================
# Logger
# ============================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# CUDA helpers
# ============================================================

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


# ============================================================
# TensorRT utilities
# ============================================================

def trt_dtype_to_numpy(dtype):
    if dtype == trt.DataType.FLOAT:
        return np.float32

    if dtype == trt.DataType.HALF:
        return np.float16

    if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
        return np.dtype("bfloat16")

    if dtype == trt.DataType.INT8:
        return np.int8

    if dtype == trt.DataType.INT32:
        return np.int32

    if hasattr(trt.DataType, "INT64") and dtype == trt.DataType.INT64:
        return np.int64

    if dtype == trt.DataType.BOOL:
        return np.bool_

    raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")


def print_engine_info(engine, name):
    print()
    print("=" * 80)
    print(f"Engine: {name}")
    print("=" * 80)

    # TensorRT 10.x
    if hasattr(engine, "num_io_tensors"):

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(tensor_name)
            dtype = engine.get_tensor_dtype(tensor_name)
            shape = engine.get_tensor_shape(tensor_name)

            print(
                f"{'INPUT ' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'} "
                f"{tensor_name:30s} "
                f"{str(shape):25s} "
                f"{dtype}"
            )

    # TensorRT 8.x
    else:

        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            is_input = engine.binding_is_input(i)

            print(
                f"{'INPUT ' if is_input else 'OUTPUT'} "
                f"{binding_name:30s} "
                f"{str(shape):25s} "
                f"{dtype}"
            )


# ============================================================
# Load reference
# ============================================================

def load_reference():

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference file not found:\n{REFERENCE_PATH}"
        )

    data = np.load(REFERENCE_PATH)

    required_keys = [
        "visual_tokens",
        "noisy_actions",
        "high_level_command",
        "diffusion_step",
        "action_velocity",
    ]

    print()
    print("=" * 80)
    print("Reference")
    print("=" * 80)

    for key in required_keys:

        if key not in data:
            raise KeyError(
                f"Missing reference key: {key}"
            )

        x = data[key]

        print(
            f"{key:25s} "
            f"shape={x.shape} "
            f"dtype={x.dtype}"
        )

    return data


# ============================================================
# Tensor name matching
# ============================================================

def normalize_name(name):
    return name.lower().replace("_", "").replace(".", "")


def find_tensor_name(engine, target, is_input=True):

    target_norm = normalize_name(target)

    candidates = []

    if hasattr(engine, "num_io_tensors"):

        for i in range(engine.num_io_tensors):

            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)

            if is_input and mode != trt.TensorIOMode.INPUT:
                continue

            if not is_input and mode != trt.TensorIOMode.OUTPUT:
                continue

            candidates.append(name)

    else:

        for i in range(engine.num_bindings):

            name = engine.get_binding_name(i)

            if engine.binding_is_input(i) != is_input:
                continue

            candidates.append(name)

    # Exact normalized match
    for name in candidates:
        if normalize_name(name) == target_norm:
            return name

    # Partial match
    for name in candidates:
        if target_norm in normalize_name(name):
            return name

    return None


# ============================================================
# TensorRT Engine wrapper
# ============================================================

class TensorRTEngine:

    def __init__(self, engine_path):

        self.engine_path = Path(engine_path)

        print()
        print(f"Loading engine:")
        print(f"  {self.engine_path}")

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"Engine not found: {self.engine_path}"
            )

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(TRT_LOGGER)

        self.engine = self.runtime.deserialize_cuda_engine(
            engine_data
        )

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine:\n"
                f"{self.engine_path}"
            )

        self.context = self.engine.create_execution_context()

        print_engine_info(
            self.engine,
            self.engine_path.name
        )

        self.input_names = {
            "visual_tokens":
                find_tensor_name(
                    self.engine,
                    "visual_tokens",
                    True
                ),

            "noisy_actions":
                find_tensor_name(
                    self.engine,
                    "noisy_actions",
                    True
                ),

            "high_level_command":
                find_tensor_name(
                    self.engine,
                    "high_level_command",
                    True
                ),

            "diffusion_step":
                find_tensor_name(
                    self.engine,
                    "diffusion_step",
                    True
                ),
        }

        self.output_name = find_tensor_name(
            self.engine,
            "action_velocity",
            False
        )

        print()
        print("Tensor mapping:")
        print(self.input_names)
        print("output:", self.output_name)

        for key, name in self.input_names.items():
            if name is None:
                raise RuntimeError(
                    f"Cannot find TensorRT input for: {key}"
                )

        if self.output_name is None:
            raise RuntimeError(
                "Cannot find TensorRT output: action_velocity"
            )

        self.allocate_buffers()

    # --------------------------------------------------------
    # Tensor metadata
    # --------------------------------------------------------

    def get_dtype(self, name):

        if hasattr(self.engine, "get_tensor_dtype"):
            return self.engine.get_tensor_dtype(name)

        index = self.engine.get_binding_index(name)
        return self.engine.get_binding_dtype(index)

    def get_shape(self, name):

        if hasattr(self.engine, "get_tensor_shape"):
            return tuple(self.engine.get_tensor_shape(name))

        index = self.engine.get_binding_index(name)
        return tuple(self.engine.get_binding_shape(index))

    # --------------------------------------------------------
    # Buffer allocation
    # --------------------------------------------------------

    def allocate_buffers(self):

        self.device_buffers = {}
        self.host_outputs = {}

        tensor_names = []

        if hasattr(self.engine, "num_io_tensors"):

            for i in range(self.engine.num_io_tensors):
                tensor_names.append(
                    self.engine.get_tensor_name(i)
                )

        else:

            for i in range(self.engine.num_bindings):
                tensor_names.append(
                    self.engine.get_binding_name(i)
                )

        for name in tensor_names:

            dtype = self.get_dtype(name)
            shape = self.get_shape(name)

            # Dynamic shape safety
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Dynamic shape not resolved for {name}: {shape}"
                )

            np_dtype = trt_dtype_to_numpy(dtype)

            size = int(np.prod(shape))

            dummy = np.empty(
                size,
                dtype=np_dtype
            )

            ptr = cuda_malloc(dummy.nbytes)

            self.device_buffers[name] = {
                "ptr": ptr,
                "shape": shape,
                "dtype": np_dtype,
                "nbytes": dummy.nbytes,
            }

            if name == self.output_name:

                self.host_outputs[name] = np.empty(
                    shape,
                    dtype=np_dtype
                )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    def prepare_input(self, tensor_name, array):

        info = self.device_buffers[tensor_name]

        expected_shape = tuple(info["shape"])
        expected_dtype = info["dtype"]

        array = np.asarray(array)

        if tuple(array.shape) != expected_shape:
            raise RuntimeError(
                f"Shape mismatch for {tensor_name}\n"
                f"Engine : {expected_shape}\n"
                f"Input  : {array.shape}"
            )

        array = np.ascontiguousarray(
            array.astype(
                expected_dtype,
                copy=False
            )
        )

        return array

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    def execute(self, inputs):

        # ----------------------------------------------------
        # TensorRT 10.x
        # ----------------------------------------------------

        if hasattr(self.context, "set_tensor_address"):

            for key, tensor_name in self.input_names.items():

                arr = self.prepare_input(
                    tensor_name,
                    inputs[key]
                )

                cuda_memcpy_htod(
                    self.device_buffers[tensor_name]["ptr"],
                    arr
                )

                self.context.set_tensor_address(
                    tensor_name,
                    int(self.device_buffers[tensor_name]["ptr"])
                )

            self.context.set_tensor_address(
                self.output_name,
                int(
                    self.device_buffers[
                        self.output_name
                    ]["ptr"]
                )
            )

            success = self.context.execute_async_v3(
                0
            )

            if not success:
                raise RuntimeError(
                    "TensorRT execute_async_v3 failed"
                )

        # ----------------------------------------------------
        # TensorRT 8.x
        # ----------------------------------------------------

        else:

            bindings = [0] * self.engine.num_bindings

            for key, tensor_name in self.input_names.items():

                arr = self.prepare_input(
                    tensor_name,
                    inputs[key]
                )

                cuda_memcpy_htod(
                    self.device_buffers[tensor_name]["ptr"],
                    arr
                )

                index = self.engine.get_binding_index(
                    tensor_name
                )

                bindings[index] = int(
                    self.device_buffers[tensor_name]["ptr"]
                )

            output_index = self.engine.get_binding_index(
                self.output_name
            )

            bindings[output_index] = int(
                self.device_buffers[
                    self.output_name
                ]["ptr"]
            )

            success = self.context.execute_async_v2(
                bindings=bindings,
                stream_handle=0
            )

            if not success:
                raise RuntimeError(
                    "TensorRT execute_async_v2 failed"
                )

        # Synchronize because this validation script
        # uses the default CUDA stream.
        result = driver.cuCtxSynchronize()

        cuda_check(
            result,
            "cuCtxSynchronize"
        )

        output = self.host_outputs[self.output_name]

        cuda_memcpy_dtoh(
            output,
            self.device_buffers[
                self.output_name
            ]["ptr"]
        )

        return np.array(
            output,
            copy=True
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def __del__(self):

        try:

            for info in self.device_buffers.values():
                cuda_free(info["ptr"])

        except Exception:
            pass


# ============================================================
# Accuracy comparison
# ============================================================

def compare_outputs(reference, test):

    reference = np.asarray(
        reference,
        dtype=np.float32
    )

    test = np.asarray(
        test,
        dtype=np.float32
    )

    if reference.shape != test.shape:

        raise RuntimeError(
            "Output shape mismatch:\n"
            f"reference = {reference.shape}\n"
            f"test      = {test.shape}"
        )

    diff = np.abs(
        reference - test
    )

    max_abs = float(
        diff.max()
    )

    mean_abs = float(
        diff.mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (reference - test) ** 2
            )
        )
    )

    denom = np.maximum(
        np.abs(reference),
        1e-8
    )

    max_rel = float(
        np.max(
            diff / denom
        )
    )

    max_index = np.unravel_index(
        np.argmax(diff),
        diff.shape
    )

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "max_rel": max_rel,
        "max_index": max_index,
        "reference_at_max": float(
            reference[max_index]
        ),
        "test_at_max": float(
            test[max_index]
        ),
    }


# ============================================================
# Latency benchmark
# ============================================================

def benchmark_engine(
    engine,
    inputs,
    reference_output,
    warmup=WARMUP,
    iterations=ITERATIONS,
):

    print()
    print(
        f"Warmup: {warmup}, "
        f"Iterations: {iterations}"
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(warmup):

        engine.execute(inputs)

    # --------------------------------------------------------
    # Synchronize before timing
    # --------------------------------------------------------

    cuda_check(
        driver.cuCtxSynchronize(),
        "cuCtxSynchronize"
    )

    # --------------------------------------------------------
    # End-to-end timing
    # --------------------------------------------------------

    e2e_times = []

    output = None

    for _ in range(iterations):

        t0 = time.perf_counter()

        output = engine.execute(inputs)

        t1 = time.perf_counter()

        e2e_times.append(
            (t1 - t0) * 1000.0
        )

    e2e_times = np.asarray(
        e2e_times,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = compare_outputs(
        reference_output,
        output
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    # Remove first few samples from latency statistics
    # to avoid residual startup effects.
    stable = e2e_times

    return {
        "e2e_mean": float(np.mean(stable)),
        "e2e_median": float(np.median(stable)),
        "e2e_p95": float(np.percentile(stable, 95)),
        "e2e_min": float(np.min(stable)),
        "e2e_max": float(np.max(stable)),
        "accuracy": accuracy,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("VaVAM Thor Benchmark")
    print("=" * 80)

    print()
    print(f"Reference:")
    print(f"  {REFERENCE_PATH}")

    # --------------------------------------------------------
    # CUDA init
    # --------------------------------------------------------

    cuda_check(
        driver.cuInit(0),
        "cuInit"
    )

    result = driver.cuDeviceGet(0)

    cuda_check(
        result,
        "cuDeviceGet"
    )

    device = result[1]

    result = driver.cuDeviceGetName(device)

    cuda_check(
        result,
        "cuDeviceGetName"
    )

    device_name = result[1]

    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    print()
    print(f"CUDA Device: {device_name}")

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = load_reference()

    inputs = {
        "visual_tokens":
            reference["visual_tokens"],

        "noisy_actions":
            reference["noisy_actions"],

        "high_level_command":
            reference["high_level_command"],

        "diffusion_step":
            reference["diffusion_step"],
    }

    reference_output = reference[
        "action_velocity"
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Benchmark all models
    # --------------------------------------------------------

    for model_name in ["B", "L"]:

        for precision in [
            "FP32_TF32",
            "FP32",
            "FP16",
            "BF16",
        ]:

            engine_path = ENGINES[
                model_name
            ][precision]

            print()
            print()
            print("#" * 80)
            print(
                f"# {model_name} / {precision}"
            )
            print("#" * 80)

            if not engine_path.exists():

                print(
                    f"SKIP: engine not found:\n"
                    f"  {engine_path}"
                )

                results.append({
                    "model": model_name,
                    "precision": precision,
                    "status": "SKIP",
                })

                continue

            try:

                engine = TensorRTEngine(
                    engine_path
                )

                stats = benchmark_engine(
                    engine,
                    inputs,
                    reference_output,
                )

                acc = stats["accuracy"]

                result = {
                    "model":
                        model_name,

                    "precision":
                        precision,

                    "status":
                        "OK",

                    "e2e_mean_ms":
                        stats["e2e_mean"],

                    "e2e_median_ms":
                        stats["e2e_median"],

                    "e2e_p95_ms":
                        stats["e2e_p95"],

                    "e2e_min_ms":
                        stats["e2e_min"],

                    "e2e_max_ms":
                        stats["e2e_max"],

                    "max_abs_error":
                        acc["max_abs"],

                    "mean_abs_error":
                        acc["mean_abs"],

                    "rmse":
                        acc["rmse"],

                    "max_rel_error":
                        acc["max_rel"],

                    "max_error_index":
                        acc["max_index"],
                }

                results.append(result)

                # ------------------------------------------------
                # Print result
                # ------------------------------------------------

                print()
                print("-" * 80)
                print(
                    f"{model_name} / {precision}"
                )
                print("-" * 80)

                print()
                print("Latency")
                print(
                    f"  E2E mean   : "
                    f"{stats['e2e_mean']:.3f} ms"
                )

                print(
                    f"  E2E median : "
                    f"{stats['e2e_median']:.3f} ms"
                )

                print(
                    f"  E2E P95    : "
                    f"{stats['e2e_p95']:.3f} ms"
                )

                print(
                    f"  E2E min    : "
                    f"{stats['e2e_min']:.3f} ms"
                )

                print(
                    f"  E2E max    : "
                    f"{stats['e2e_max']:.3f} ms"
                )

                print()
                print("Accuracy")
                print(
                    f"  max abs    : "
                    f"{acc['max_abs']:.10e}"
                )

                print(
                    f"  mean abs   : "
                    f"{acc['mean_abs']:.10e}"
                )

                print(
                    f"  RMSE       : "
                    f"{acc['rmse']:.10e}"
                )

                print(
                    f"  max rel    : "
                    f"{acc['max_rel']:.10e}"
                )

                print(
                    f"  max index  : "
                    f"{acc['max_index']}"
                )

                print(
                    f"  reference  : "
                    f"{acc['reference_at_max']:.10e}"
                )

                print(
                    f"  TensorRT   : "
                    f"{acc['test_at_max']:.10e}"
                )

                del engine

            except Exception as e:

                print()
                print(
                    f"ERROR: "
                    f"{model_name} / {precision}"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                results.append({
                    "model": model_name,
                    "precision": precision,
                    "status": "ERROR",
                    "error": str(e),
                })

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print()
    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)

    print()

    print(
        f"{'Model':<8}"
        f"{'Precision':<14}"
        f"{'Status':<8}"
        f"{'E2E Mean(ms)':>15}"
        f"{'Median(ms)':>15}"
        f"{'P95(ms)':>15}"
        f"{'Max Abs':>15}"
        f"{'RMSE':>15}"
    )

    print("-" * 110)

    for r in results:

        if r["status"] != "OK":

            print(
                f"{r['model']:<8}"
                f"{r['precision']:<14}"
                f"{r['status']:<8}"
            )

            continue

        print(
            f"{r['model']:<8}"
            f"{r['precision']:<14}"
            f"{r['status']:<8}"
            f"{r['e2e_mean_ms']:>15.3f}"
            f"{r['e2e_median_ms']:>15.3f}"
            f"{r['e2e_p95_ms']:>15.3f}"
            f"{r['max_abs_error']:>15.3e}"
            f"{r['rmse']:>15.3e}"
        )

    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
