#!/usr/bin/env python3
"""
VaVAM 12e - Thor TensorRT Inference-Step Validation

Purpose
-------
Validate the TensorRT FP32 inference-step engine on the target platform
(e.g. NVIDIA DRIVE AGX Thor).

This script does NOT require:
  - PyTorch
  - ONNX Runtime
  - VaVAM checkpoint
  - ONNX model

It only requires:
  - TensorRT engine
  - deterministic reference NPZ
  - CUDA Driver API

Validation flow:

    visual_tokens
          +
    noisy_actions
          +
    high_level_command
          +
    diffusion_step
          |
          v
    TensorRT FP32 Engine
          |
          v
    action_velocity
          |
          v
    Compare with PC PyTorch reference

Expected reference file:
    thor_inference_step_reference.npz

Expected keys:
    visual_tokens
    noisy_actions
    high_level_command
    diffusion_step
    action_velocity
"""

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from cuda.bindings import driver


# ============================================================================
# Paths
# ============================================================================

ROOT = Path.home() / "VideoActionModel"

ENGINE_PATH = (
    ROOT
    / "artifacts"
    / "vavam_joint"
    / "vavam_joint_inference_step_fp32.engine"
)

REFERENCE_PATH = (
    ROOT
    / "artifacts"
    / "vavam_joint"
    / "thor_inference_step_reference.npz"
)


# ============================================================================
# CUDA helpers
# ============================================================================

def cuda_check(result, name):
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"{name} failed: {err}"
        )

    return result


def cuda_malloc(nbytes):
    result = driver.cuMemAlloc(
        int(nbytes)
    )

    cuda_check(
        result,
        "cuMemAlloc"
    )

    return result[1]


def cuda_free(ptr):
    if ptr is None:
        return

    result = driver.cuMemFree(
        ptr
    )

    cuda_check(
        result,
        "cuMemFree"
    )


def cuda_memcpy_htod(ptr, host_array):
    arr = np.ascontiguousarray(
        host_array
    )

    if not arr.flags["C_CONTIGUOUS"]:
        raise RuntimeError(
            "Host array is not C-contiguous"
        )

    result = driver.cuMemcpyHtoD(
        int(ptr),
        arr,
        int(arr.nbytes),
    )

    cuda_check(
        result,
        "cuMemcpyHtoD"
    )


def cuda_memcpy_dtoh(host_array, ptr):
    arr = np.ascontiguousarray(
        host_array
    )

    result = driver.cuMemcpyDtoH(
        arr,
        int(ptr),
        int(arr.nbytes),
    )

    cuda_check(
        result,
        "cuMemcpyDtoH"
    )


# ============================================================================
# Statistics
# ============================================================================

def print_stats(name, x):

    x = np.asarray(x)

    print()
    print(
        f"[{name}]"
    )

    print(
        f"  shape : {x.shape}"
    )

    print(
        f"  dtype : {x.dtype}"
    )

    print(
        f"  min   : {x.min()}"
    )

    print(
        f"  max   : {x.max()}"
    )

    print(
        f"  mean  : {x.mean()}"
    )

    print(
        f"  std   : {x.std()}"
    )


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
            "Output shape mismatch: "
            f"reference={reference.shape}, "
            f"test={test.shape}"
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
        1e-8,
    )

    max_rel = float(
        np.max(
            diff / denom
        )
    )

    max_index = np.unravel_index(
        np.argmax(diff),
        diff.shape,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "Thor TensorRT vs PC PyTorch Reference"
    )

    print(
        "=" * 80
    )

    print(
        f"  max abs error : {max_abs:.10e}"
    )

    print(
        f"  mean abs error: {mean_abs:.10e}"
    )

    print(
        f"  RMSE          : {rmse:.10e}"
    )

    print(
        f"  max rel error : {max_rel:.10e}"
    )

    print(
        f"  max error idx : {max_index}"
    )

    print(
        f"  reference     : "
        f"{reference[max_index]}"
    )

    print(
        f"  Thor TRT      : "
        f"{test[max_index]}"
    )

    return max_abs


# ============================================================================
# Load reference
# ============================================================================

def load_reference():

    print()
    print(
        "=" * 80
    )

    print(
        "Loading Thor validation reference"
    )

    print(
        "=" * 80
    )

    print()
    print(
        f"  reference : {REFERENCE_PATH}"
    )

    if not REFERENCE_PATH.exists():

        raise FileNotFoundError(
            f"Reference file not found:\n"
            f"{REFERENCE_PATH}"
        )

    data = np.load(
        REFERENCE_PATH
    )

    required_keys = [
        "visual_tokens",
        "noisy_actions",
        "high_level_command",
        "diffusion_step",
        "action_velocity",
    ]

    for key in required_keys:

        if key not in data:

            raise RuntimeError(
                f"Missing reference key: {key}"
            )

    visual_tokens = np.ascontiguousarray(
        data["visual_tokens"]
        .astype(np.int64)
    )

    noisy_actions = np.ascontiguousarray(
        data["noisy_actions"]
        .astype(np.float32)
    )

    high_level_command = np.ascontiguousarray(
        data["high_level_command"]
        .astype(np.int64)
    )

    diffusion_step = np.ascontiguousarray(
        data["diffusion_step"]
        .astype(np.float32)
    )

    reference_output = np.ascontiguousarray(
        data["action_velocity"]
        .astype(np.float32)
    )

    print()

    print(
        "  [Reference inputs]"
    )

    print(
        f"    visual_tokens      : "
        f"{visual_tokens.shape} "
        f"{visual_tokens.dtype}"
    )

    print(
        f"    noisy_actions      : "
        f"{noisy_actions.shape} "
        f"{noisy_actions.dtype}"
    )

    print(
        f"    high_level_command : "
        f"{high_level_command.shape} "
        f"{high_level_command.dtype}"
    )

    print(
        f"    diffusion_step     : "
        f"{diffusion_step.shape} "
        f"{diffusion_step.dtype}"
    )

    print()

    print(
        "  [PC PyTorch reference output]"
    )

    print(
        f"    action_velocity    : "
        f"{reference_output.shape} "
        f"{reference_output.dtype}"
    )

    return {
        "visual_tokens": visual_tokens,
        "noisy_actions": noisy_actions,
        "high_level_command": high_level_command,
        "diffusion_step": diffusion_step,
        "action_velocity": reference_output,
    }


# ============================================================================
# TensorRT validation
# ============================================================================

def run_tensorrt(reference):

    print()
    print(
        "=" * 80
    )

    print(
        "Loading TensorRT FP32 Engine"
    )

    print(
        "=" * 80
    )

    print()
    print(
        f"  engine : {ENGINE_PATH}"
    )

    if not ENGINE_PATH.exists():

        raise FileNotFoundError(
            f"TensorRT engine not found:\n"
            f"{ENGINE_PATH}"
        )

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

    engine = runtime.deserialize_cuda_engine(
        engine_data
    )

    if engine is None:

        raise RuntimeError(
            "Failed to deserialize TensorRT engine."
        )

    print()
    print(
        "[OK] TensorRT engine deserialized"
    )

    context = engine.create_execution_context()

    if context is None:

        raise RuntimeError(
            "Failed to create execution context."
        )

    # ------------------------------------------------------------------------
    # Expected input map
    # ------------------------------------------------------------------------

    input_map = {

        "visual_tokens":
            reference["visual_tokens"],

        "noisy_actions":
            reference["noisy_actions"],

        "high_level_command":
            reference["high_level_command"],

        "diffusion_step":
            reference["diffusion_step"],
    }

    allocations = {}
    host_outputs = {}

    # ------------------------------------------------------------------------
    # Inspect I/O
    # ------------------------------------------------------------------------

    print()
    print(
        "[TRT] Engine I/O"
    )

    for i in range(
        engine.num_io_tensors
    ):

        name = engine.get_tensor_name(
            i
        )

        mode = engine.get_tensor_mode(
            name
        )

        dtype = engine.get_tensor_dtype(
            name
        )

        shape = tuple(
            engine.get_tensor_shape(
                name
            )
        )

        print()
        print(
            f"  {name}"
        )

        print(
            f"    mode  : {mode}"
        )

        print(
            f"    dtype : {dtype}"
        )

        print(
            f"    shape : {shape}"
        )

        if any(
            d < 0
            for d in shape
        ):

            raise RuntimeError(
                f"Dynamic shape encountered: "
                f"{name} {shape}"
            )

        # ------------------------------------------------------------
        # Validate expected bindings
        # ------------------------------------------------------------

        if mode == trt.TensorIOMode.INPUT:

            if name not in input_map:

                raise RuntimeError(
                    f"Unexpected engine input: "
                    f"{name}"
                )

            host = input_map[
                name
            ]

            expected_dtype = np.dtype(
                trt.nptype(dtype)
            )

            if host.dtype != expected_dtype:

                raise RuntimeError(
                    f"Dtype mismatch for {name}: "
                    f"host={host.dtype}, "
                    f"engine={expected_dtype}"
                )

            if tuple(host.shape) != shape:

                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"host={host.shape}, "
                    f"engine={shape}"
                )

            allocations[name] = cuda_malloc(
                host.nbytes
            )

        else:

            np_dtype = np.dtype(
                trt.nptype(dtype)
            )

            host_outputs[name] = np.empty(
                shape,
                dtype=np_dtype,
            )

            allocations[name] = cuda_malloc(
                host_outputs[name].nbytes
            )

    # ------------------------------------------------------------------------
    # Validate exact expected I/O
    # ------------------------------------------------------------------------

    expected_inputs = {
        "visual_tokens",
        "noisy_actions",
        "high_level_command",
        "diffusion_step",
    }

    actual_inputs = set(
        name
        for name in allocations
        if engine.get_tensor_mode(name)
        == trt.TensorIOMode.INPUT
    )

    if actual_inputs != expected_inputs:

        raise RuntimeError(
            "Input binding mismatch:\n"
            f"expected={expected_inputs}\n"
            f"actual={actual_inputs}"
        )

    if "action_velocity" not in host_outputs:

        raise RuntimeError(
            "Expected output "
            "'action_velocity' not found."
        )

    # ------------------------------------------------------------------------
    # Copy inputs H -> D
    # ------------------------------------------------------------------------

    print()
    print(
        "[TRT] Copying inputs..."
    )

    for name in (
        "visual_tokens",
        "noisy_actions",
        "high_level_command",
        "diffusion_step",
    ):

        host = input_map[
            name
        ]

        print(
            f"  [COPY] {name}"
        )

        print(
            f"    dtype={host.dtype} "
            f"shape={host.shape} "
            f"nbytes={host.nbytes}"
        )

        cuda_memcpy_htod(
            allocations[name],
            host,
        )

        print(
            f"  [OK] {name}"
        )

    # ------------------------------------------------------------------------
    # Set tensor addresses
    # ------------------------------------------------------------------------

    print()
    print(
        "[TRT] Setting tensor addresses..."
    )

    for name, ptr in allocations.items():

        ok = context.set_tensor_address(
            name,
            int(ptr),
        )

        if not ok:

            raise RuntimeError(
                f"Failed to set tensor address: "
                f"{name}"
            )

        print(
            f"  [OK] {name}"
        )

    # ------------------------------------------------------------------------
    # CUDA stream
    # ------------------------------------------------------------------------

    print()
    print(
        "[TRT] Creating CUDA stream..."
    )

    result = driver.cuStreamCreate(
        0
    )

    cuda_check(
        result,
        "cuStreamCreate"
    )

    stream = result[1]

    print(
        "[OK] CUDA stream created"
    )

    # ------------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------------

    print()
    print(
        "[TRT] Executing engine..."
    )

    ok = context.execute_async_v3(
        stream_handle=int(stream)
    )

    if not ok:

        raise RuntimeError(
            "TensorRT execute_async_v3 failed."
        )

    # Synchronize stream

    result = driver.cuStreamSynchronize(
        stream
    )

    cuda_check(
        result,
        "cuStreamSynchronize"
    )

    print(
        "[OK] TensorRT inference completed"
    )

    # ------------------------------------------------------------------------
    # Copy output D -> H
    # ------------------------------------------------------------------------

    output = host_outputs[
        "action_velocity"
    ]

    cuda_memcpy_dtoh(
        output,
        allocations[
            "action_velocity"
        ],
    )

    print_stats(
        "TensorRT action_velocity",
        output,
    )

    # ------------------------------------------------------------------------
    # Cleanup CUDA allocations
    # ------------------------------------------------------------------------

    for ptr in allocations.values():

        cuda_free(
            ptr
        )

    result = driver.cuStreamDestroy(
        stream
    )

    cuda_check(
        result,
        "cuStreamDestroy"
    )

    return output


# ============================================================================
# Main
# ============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "VaVAM 12e - Thor TensorRT Inference-Step Validation"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "[Environment]"
    )

    print(
        f"  TensorRT : {trt.__version__}"
    )

    print()

    # ------------------------------------------------------------------------
    # CUDA init
    # ------------------------------------------------------------------------

    result = driver.cuInit(
        0
    )

    cuda_check(
        result,
        "cuInit"
    )

    print(
        "[OK] CUDA Driver initialized"
    )

    # ------------------------------------------------------------------------
    # Load reference
    # ------------------------------------------------------------------------

    reference = load_reference()

    # ------------------------------------------------------------------------
    # Run TRT
    # ------------------------------------------------------------------------

    trt_output = run_tensorrt(
        reference
    )

    # ------------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------------

    max_abs = compare_outputs(
        reference["action_velocity"],
        trt_output,
    )

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "12e RESULT"
    )

    print(
        "=" * 80
    )

    print()

    if max_abs <= 1e-4:

        print(
            "[PASS] Thor TensorRT FP32 output "
            "matches PC PyTorch reference "
            "within 1e-4 max absolute error."
        )

        print()

        print(
            "VaVAM inference-step validation "
            "on Thor is successful."
        )

        return 0

    elif max_abs <= 1e-3:

        print(
            "[WARN] Numerical difference is "
            "greater than 1e-4 but within 1e-3."
        )

        return 1

    else:

        print(
            "[FAIL] TensorRT output differs "
            "from PC PyTorch reference."
        )

        return 2


if __name__ == "__main__":

    sys.exit(
        main()
    )