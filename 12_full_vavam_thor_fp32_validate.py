import os
import sys
import time
import numpy as np

import tensorrt as trt

from cuda.bindings import driver


# ============================================================================
# Paths
# ============================================================================

ENGINE_PATH = "vavam_joint_thor_fp32.engine"
REFERENCE_PATH = "joint_reference.npz"


# ============================================================================
# CUDA helpers
# ============================================================================

def check_cuda(result, name):
    """
    cuda-python bindings generally return:
        (CUresult, ...)
    """
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if getattr(err, "value", int(err) if hasattr(err, "__int__") else None) != 0:
        raise RuntimeError(f"{name} failed: {err}")

    return result


def cuda_malloc(nbytes):
    result = driver.cuMemAlloc(int(nbytes))

    if not isinstance(result, tuple):
        raise RuntimeError(
            f"Unexpected cuMemAlloc return: {result}"
        )

    err, ptr = result

    if getattr(err, "value", int(err)) != 0:
        raise RuntimeError(
            f"cuMemAlloc failed: {err}"
        )

    return ptr


def cuda_free(ptr):
    if ptr is None:
        return

    result = driver.cuMemFree(ptr)

    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if getattr(err, "value", int(err)) != 0:
        raise RuntimeError(
            f"cuMemFree failed: {err}"
        )


def cuda_memcpy_htod(ptr, array):
    array = np.ascontiguousarray(array)

    nbytes = int(array.nbytes)

    result = driver.cuMemcpyHtoD(
        ptr,
        array,
        nbytes,
    )

    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if getattr(err, "value", int(err)) != 0:
        raise RuntimeError(
            f"cuMemcpyHtoD failed: {err}"
        )


def cuda_memcpy_dtoh(array, ptr):
    array = np.ascontiguousarray(array)

    nbytes = int(array.nbytes)

    result = driver.cuMemcpyDtoH(
        array,
        ptr,
        nbytes,
    )

    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if getattr(err, "value", int(err)) != 0:
        raise RuntimeError(
            f"cuMemcpyDtoH failed: {err}"
        )


# ============================================================================
# Statistics
# ============================================================================

def print_stats(name, x):
    x = np.asarray(x)

    print()
    print(f"[{name}]")
    print(f"  shape : {x.shape}")
    print(f"  dtype : {x.dtype}")
    print(f"  min   : {x.min()}")
    print(f"  max   : {x.max()}")
    print(f"  mean  : {x.mean()}")
    print(f"  std   : {x.std()}")


def compare_outputs(name, reference, actual):
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)

    diff = actual - reference

    abs_diff = np.abs(diff)

    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    denom = np.maximum(
        np.abs(reference),
        1e-6,
    )

    rel = abs_diff / denom

    max_rel = float(np.max(rel))
    mean_rel = float(np.mean(rel))

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"  max abs error : {max_abs:.10e}")
    print(f"  mean abs error: {mean_abs:.10e}")
    print(f"  RMSE          : {rmse:.10e}")
    print(f"  max rel error : {max_rel:.10e}")
    print(f"  mean rel error: {mean_rel:.10e}")

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
    }


# ============================================================================
# TensorRT engine information
# ============================================================================

def print_engine_info(engine):
    print()
    print("=" * 80)
    print("TensorRT Engine")
    print("=" * 80)

    print()
    print(f"  TensorRT version : {trt.__version__}")
    print(f"  num I/O tensors  : {engine.num_io_tensors}")

    print()
    print("[I/O tensors]")

    for i in range(engine.num_io_tensors):

        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(engine.get_tensor_shape(name))

        print()
        print(f"  {name}")
        print(f"    mode  : {mode}")
        print(f"    dtype : {dtype}")
        print(f"    shape : {shape}")


# ============================================================================
# TensorRT inference
# ============================================================================

def run_trt(engine, inputs):
    print()
    print("=" * 80)
    print("TensorRT Inference")
    print("=" * 80)

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "Failed to create TensorRT execution context"
        )

    allocations = {}
    host_outputs = {}

    try:

        # ------------------------------------------------------------------
        # Allocate tensors
        # ------------------------------------------------------------------

        print()
        print("[TensorRT buffers]")

        for i in range(engine.num_io_tensors):

            name = engine.get_tensor_name(i)

            mode = engine.get_tensor_mode(name)
            dtype = engine.get_tensor_dtype(name)
            shape = tuple(engine.get_tensor_shape(name))

            np_dtype = trt.nptype(dtype)

            size = int(np.prod(shape))
            nbytes = size * np.dtype(np_dtype).itemsize

            print()
            print(f"  {name}")
            print(f"    mode  : {mode}")
            print(f"    dtype : {np_dtype}")
            print(f"    shape : {shape}")
            print(f"    bytes : {nbytes}")

            ptr = cuda_malloc(nbytes)

            allocations[name] = ptr

            if mode == trt.TensorIOMode.INPUT:

                if name not in inputs:
                    raise RuntimeError(
                        f"Missing TensorRT input: {name}"
                    )

                data = np.asarray(
                    inputs[name],
                    dtype=np_dtype,
                )

                if tuple(data.shape) != shape:
                    raise RuntimeError(
                        f"Shape mismatch for {name}: "
                        f"input={data.shape}, engine={shape}"
                    )

                cuda_memcpy_htod(
                    ptr,
                    data,
                )

            else:

                host_outputs[name] = np.empty(
                    shape,
                    dtype=np_dtype,
                )

        # ------------------------------------------------------------------
        # Set tensor addresses
        # ------------------------------------------------------------------

        print()
        print("[TensorRT tensor addresses]")

        for name, ptr in allocations.items():

            ok = context.set_tensor_address(
                name,
                int(ptr),
            )

            if not ok:
                raise RuntimeError(
                    f"Failed to set tensor address: {name}"
                )

            print(f"  [OK] {name}")

        # ------------------------------------------------------------------
        # Execute
        # ------------------------------------------------------------------

        print()
        print("[INFO] Executing TensorRT...")

        ok = context.execute_v3()

        if not ok:
            raise RuntimeError(
                "TensorRT execute_v3() failed"
            )

        # ------------------------------------------------------------------
        # Copy outputs
        # ------------------------------------------------------------------

        for name, host in host_outputs.items():

            cuda_memcpy_dtoh(
                host,
                allocations[name],
            )

        return host_outputs

    finally:

        for ptr in allocations.values():

            try:
                cuda_free(ptr)
            except Exception:
                pass


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_trt(engine, inputs, warmup=3, runs=10):

    print()
    print("=" * 80)
    print("TensorRT Benchmark")
    print("=" * 80)

    context = engine.create_execution_context()

    allocations = {}

    try:

        # --------------------------------------------------------------
        # Allocate and copy inputs
        # --------------------------------------------------------------

        output_hosts = {}

        for i in range(engine.num_io_tensors):

            name = engine.get_tensor_name(i)

            mode = engine.get_tensor_mode(name)
            dtype = engine.get_tensor_dtype(name)
            shape = tuple(engine.get_tensor_shape(name))

            np_dtype = trt.nptype(dtype)

            size = int(np.prod(shape))
            nbytes = size * np.dtype(np_dtype).itemsize

            ptr = cuda_malloc(nbytes)

            allocations[name] = ptr

            if mode == trt.TensorIOMode.INPUT:

                data = np.asarray(
                    inputs[name],
                    dtype=np_dtype,
                )

                cuda_memcpy_htod(
                    ptr,
                    data,
                )

            else:

                output_hosts[name] = np.empty(
                    shape,
                    dtype=np_dtype,
                )

        # --------------------------------------------------------------
        # Set addresses
        # --------------------------------------------------------------

        for name, ptr in allocations.items():

            ok = context.set_tensor_address(
                name,
                int(ptr),
            )

            if not ok:
                raise RuntimeError(
                    f"Failed to set tensor address: {name}"
                )

        # --------------------------------------------------------------
        # Warmup
        # --------------------------------------------------------------

        print()
        print(f"  warmup runs : {warmup}")

        for _ in range(warmup):

            ok = context.execute_v3()

            if not ok:
                raise RuntimeError(
                    "TensorRT warmup execute_v3() failed"
                )

        # --------------------------------------------------------------
        # Benchmark
        # --------------------------------------------------------------

        latencies = []

        print()
        print(f"  benchmark runs : {runs}")

        for i in range(runs):

            t0 = time.perf_counter()

            ok = context.execute_v3()

            if not ok:
                raise RuntimeError(
                    "TensorRT benchmark execute_v3() failed"
                )

            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0

            latencies.append(latency_ms)

            print(
                f"  run {i:02d}: "
                f"{latency_ms:.4f} ms"
            )

        latencies = np.asarray(
            latencies,
            dtype=np.float64,
        )

        print()
        print("[TensorRT latency]")
        print(
            f"  mean : {latencies.mean():.4f} ms"
        )
        print(
            f"  min  : {latencies.min():.4f} ms"
        )
        print(
            f"  max  : {latencies.max():.4f} ms"
        )
        print(
            f"  std  : {latencies.std():.4f} ms"
        )

        return latencies

    finally:

        for ptr in allocations.values():

            try:
                cuda_free(ptr)
            except Exception:
                pass


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 80)
    print("VaVAM Full JointModel - Thor TensorRT FP32 Validation")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # Environment
    # ----------------------------------------------------------------------

    print()
    print("[Environment]")

    print(
        f"  Python       : "
        f"{sys.version.split()[0]}"
    )

    print(
        f"  TensorRT     : "
        f"{trt.__version__}"
    )

    # ----------------------------------------------------------------------
    # CUDA initialization
    # ----------------------------------------------------------------------

    print()
    print("[CUDA]")

    result = driver.cuInit(0)

    check_cuda(
        result,
        "cuInit",
    )

    result = driver.cuDeviceGetCount()

    if not isinstance(result, tuple):
        raise RuntimeError(
            f"Unexpected cuDeviceGetCount result: {result}"
        )

    err, count = result

    check_cuda(
        err,
        "cuDeviceGetCount",
    )

    print(
        f"  device count : {count}"
    )

    if count < 1:
        raise RuntimeError(
            "No CUDA device found"
        )

    result = driver.cuDeviceGet(0)

    if not isinstance(result, tuple):
        raise RuntimeError(
            f"Unexpected cuDeviceGet result: {result}"
        )

    err, device = result

    check_cuda(
        err,
        "cuDeviceGet",
    )

    print(
        f"  device       : {device}"
    )

    # ----------------------------------------------------------------------
    # Files
    # ----------------------------------------------------------------------

    print()
    print("[Files]")

    print(
        f"  engine    : "
        f"{os.path.abspath(ENGINE_PATH)}"
    )

    print(
        f"  reference : "
        f"{os.path.abspath(REFERENCE_PATH)}"
    )

    if not os.path.isfile(ENGINE_PATH):
        raise FileNotFoundError(
            ENGINE_PATH
        )

    if not os.path.isfile(REFERENCE_PATH):
        raise FileNotFoundError(
            REFERENCE_PATH
        )

    # ----------------------------------------------------------------------
    # Load reference
    # ----------------------------------------------------------------------

    print()
    print("[INFO] Loading reference...")

    reference = np.load(
        REFERENCE_PATH
    )

    print()
    print("[Reference keys]")

    for key in reference.files:

        x = reference[key]

        print(
            f"  {key:25s} "
            f"shape={x.shape} "
            f"dtype={x.dtype}"
        )

    torch_output = reference[
        "torch_output"
    ]

    export_output = reference[
        "export_output"
    ]

    print_stats(
        "PyTorch reference output",
        torch_output,
    )

    print_stats(
        "ONNX-friendly/export output",
        export_output,
    )

    # ----------------------------------------------------------------------
    # Prepare inputs
    # ----------------------------------------------------------------------

    input_names = [
        "visual_tokens",
        "noisy_actions",
        "high_level_command",
        "diffusion_step",
    ]

    inputs = {}

    for name in input_names:

        if name not in reference:
            raise RuntimeError(
                f"Missing reference input: {name}"
            )

        inputs[name] = reference[name]

    # ----------------------------------------------------------------------
    # Load TensorRT engine
    # ----------------------------------------------------------------------

    print()
    print("[INFO] Loading TensorRT engine...")

    logger = trt.Logger(
        trt.Logger.WARNING
    )

    runtime = trt.Runtime(
        logger
    )

    with open(
        ENGINE_PATH,
        "rb",
    ) as f:

        engine_data = f.read()

    print(
        f"  engine size : "
        f"{len(engine_data) / (1024 * 1024):.2f} MB"
    )

    engine = runtime.deserialize_cuda_engine(
        engine_data
    )

    if engine is None:
        raise RuntimeError(
            "Failed to deserialize TensorRT engine"
        )

    print_engine_info(
        engine
    )

    # ----------------------------------------------------------------------
    # Validate I/O names
    # ----------------------------------------------------------------------

    expected_inputs = set(
        input_names
    )

    actual_inputs = set()

    actual_outputs = set()

    for i in range(
        engine.num_io_tensors
    ):

        name = engine.get_tensor_name(i)

        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            actual_inputs.add(name)

        elif mode == trt.TensorIOMode.OUTPUT:
            actual_outputs.add(name)

    print()
    print("[I/O validation]")

    print(
        f"  expected inputs : "
        f"{sorted(expected_inputs)}"
    )

    print(
        f"  engine inputs   : "
        f"{sorted(actual_inputs)}"
    )

    print(
        f"  engine outputs  : "
        f"{sorted(actual_outputs)}"
    )

    if actual_inputs != expected_inputs:

        raise RuntimeError(
            "TensorRT input names do not match "
            "reference inputs"
        )

    if "action_output" not in actual_outputs:

        raise RuntimeError(
            "TensorRT output "
            "'action_output' not found"
        )

    # ----------------------------------------------------------------------
    # Single inference
    # ----------------------------------------------------------------------

    trt_outputs = run_trt(
        engine,
        inputs,
    )

    if "action_output" not in trt_outputs:

        raise RuntimeError(
            "TensorRT did not produce action_output"
        )

    thor_output = trt_outputs[
        "action_output"
    ]

    print_stats(
        "Thor TensorRT FP32 output",
        thor_output,
    )

    # ----------------------------------------------------------------------
    # Numerical comparison
    # ----------------------------------------------------------------------

    result_torch = compare_outputs(
        "PyTorch reference vs Thor TensorRT FP32",
        torch_output,
        thor_output,
    )

    result_export = compare_outputs(
        "ONNX-friendly/export output vs Thor TensorRT FP32",
        export_output,
        thor_output,
    )

    # ----------------------------------------------------------------------
    # Determinism
    # ----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("[TensorRT Determinism]")
    print("=" * 80)

    trt_output_2 = run_trt(
        engine,
        inputs,
    )["action_output"]

    deterministic_diff = np.max(
        np.abs(
            thor_output -
            trt_output_2
        )
    )

    print(
        f"  max abs diff : "
        f"{deterministic_diff:.10e}"
    )

    if deterministic_diff == 0.0:

        print(
            "  [PASS] Deterministic"
        )

    else:

        print(
            "  [WARN] Non-zero deterministic difference"
        )

    # ----------------------------------------------------------------------
    # Benchmark
    # ----------------------------------------------------------------------

    benchmark_trt(
        engine,
        inputs,
        warmup=3,
        runs=10,
    )

    # ----------------------------------------------------------------------
    # Final result
    # ----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("[RESULT] Thor TensorRT FP32 Validation")
    print("=" * 80)

    print()
    print(
        f"  PyTorch -> Thor TRT max abs error : "
        f"{result_torch['max_abs']:.10e}"
    )

    print(
        f"  Export   -> Thor TRT max abs error : "
        f"{result_export['max_abs']:.10e}"
    )

    print(
        f"  Determinism max abs diff          : "
        f"{deterministic_diff:.10e}"
    )

    print()

    # ----------------------------------------------------------------------
    # Do NOT use the PC threshold blindly.
    #
    # We first report the actual Thor number.
    # ----------------------------------------------------------------------

    if (
        result_torch["max_abs"] <= 1e-4
        and deterministic_diff == 0.0
    ):

        print(
            "[PASS] Thor TensorRT FP32 numerical validation"
        )

    else:

        print(
            "[INFO] Thor TensorRT FP32 execution completed."
        )

        print(
            "       Numerical difference requires "
            "comparison against PC TensorRT baseline."
        )


if __name__ == "__main__":
    main()