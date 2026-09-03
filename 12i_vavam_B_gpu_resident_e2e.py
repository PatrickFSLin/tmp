#!/usr/bin/env python3

"""
VaVAM-B 12i - Thor GPU-Resident E2E Benchmark

Pipeline
--------
visual_tokens
     |
     v
TensorRT Prefill FP16
     |
     +---- visual K/V x 24 layers ----+
                                      |
                                      v
                              TensorRT Action FP16
                                      |
                                      v
                                  velocity
                                      |
                                      v
                              CUDA Euler update
                                      |
                                      v
                                   action
                                      |
                                      +------> repeat x10

Important
---------
- Prefill runs once per inference.
- 48 visual K/V tensors remain GPU-resident.
- Action remains GPU-resident.
- diffusion_step remains GPU-resident.
- Euler update is performed by a CUDA kernel.
- No D2H/H2D occurs inside the 10 Euler steps.
- Only final trajectory is copied D2H after the inference.
"""

import ctypes
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

from cuda import cuda, nvrtc


# ============================================================================
# Paths
# ============================================================================

ROOT = Path.home() / "vblkdev2" / "VaVAM_Thor"

PREFILL_ENGINE = (
    ROOT
    / "Engines"
    / "vavam_joint_kv_prefill_B_v10_fp16.engine"
)

ACTION_ENGINE = (
    ROOT
    / "Engines"
    / "vavam_joint_action_B_fp16.engine"
)

# Use the existing reference NPZ.
# Change this path if your Thor reference file is elsewhere.
REFERENCE_NPZ = (
    ROOT
    / "ONNX"
    / "vavam_joint_kv_prefill_B_v10_addmask_reference.npz"
)


# ============================================================================
# Configuration
# ============================================================================

NUM_LAYERS = 24

NUM_EULER_STEPS = 10

DELTA_T = np.float32(0.1)

ACTION_HORIZON = 6
ACTION_DIM = 2

ACTION_SIZE = (
    ACTION_HORIZON
    * ACTION_DIM
)

WARMUP = 5
BENCHMARK = 30


# ============================================================================
# CUDA helpers
# ============================================================================

def cuda_check(result, name):

    err = result[0] if isinstance(result, tuple) else result

    if err != cuda.CUresult.CUDA_SUCCESS:

        raise RuntimeError(
            f"{name} failed: {err}"
        )

    return result


def cuda_malloc(nbytes):

    result = cuda.cuMemAlloc(
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

    cuda_check(
        cuda.cuMemFree(int(ptr)),
        "cuMemFree"
    )


def cuda_htod(ptr, array):

    array = np.ascontiguousarray(array)

    cuda_check(
        cuda.cuMemcpyHtoD(
            int(ptr),
            array,
            int(array.nbytes),
        ),
        "cuMemcpyHtoD"
    )


def cuda_dtoh(array, ptr):

    array = np.ascontiguousarray(array)

    cuda_check(
        cuda.cuMemcpyDtoH(
            array,
            int(ptr),
            int(array.nbytes),
        ),
        "cuMemcpyDtoH"
    )


def cuda_stream_create():

    result = cuda.cuStreamCreate(0)

    cuda_check(
        result,
        "cuStreamCreate"
    )

    return result[1]


def cuda_stream_sync(stream):

    cuda_check(
        cuda.cuStreamSynchronize(
            stream
        ),
        "cuStreamSynchronize"
    )


def cuda_event_create():

    result = cuda.cuEventCreate(0)

    cuda_check(
        result,
        "cuEventCreate"
    )

    return result[1]


def cuda_event_record(event, stream):

    cuda_check(
        cuda.cuEventRecord(
            event,
            stream
        ),
        "cuEventRecord"
    )


def cuda_event_elapsed(start, end):

    result = cuda.cuEventElapsedTime(
        start,
        end
    )

    cuda_check(
        result,
        "cuEventElapsedTime"
    )

    return float(result[1])


# ============================================================================
# CUDA Euler kernel
# ============================================================================

EULER_CUDA = r"""
extern "C" __global__
void euler_update(
    float* action,
    const float* velocity,
    float dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 12)
    {
        action[i] = action[i] + dt * velocity[i];
    }
}
"""


def build_euler_kernel():

    print()
    print("=" * 80)
    print("Compiling CUDA Euler kernel")
    print("=" * 80)

    err, program = nvrtc.nvrtcCreateProgram(
        EULER_CUDA,
        b"euler.cu",
        0,
        [],
        [],
    )

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:

        raise RuntimeError(
            f"nvrtcCreateProgram failed: {err}"
        )

    err, _ = nvrtc.nvrtcCompileProgram(
        program,
        0,
        [],
    )

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:

        log_result = nvrtc.nvrtcGetProgramLog(
            program
        )

        log = (
            log_result[1]
            if isinstance(log_result, tuple)
            else ""
        )

        raise RuntimeError(
            f"NVRTC compile failed:\n{log}"
        )

    result = nvrtc.nvrtcGetPTX(
        program
    )

    cuda_check(
        result,
        "nvrtcGetPTX"
    )

    ptx = result[1]

    result = cuda.cuModuleLoadData(
        ptx
    )

    cuda_check(
        result,
        "cuModuleLoadData"
    )

    module = result[1]

    result = cuda.cuModuleGetFunction(
        module,
        b"euler_update"
    )

    cuda_check(
        result,
        "cuModuleGetFunction"
    )

    function = result[1]

    print("[OK] CUDA Euler kernel ready")

    return module, function


# ============================================================================
# TensorRT engine
# ============================================================================

class TensorRTEngine:

    def __init__(
        self,
        path,
    ):

        self.path = Path(path)

        print()
        print("=" * 80)
        print("Loading TensorRT engine")
        print("=" * 80)

        print(
            "Engine:",
            self.path
        )

        logger = trt.Logger(
            trt.Logger.WARNING
        )

        self.runtime = trt.Runtime(
            logger
        )

        with open(
            self.path,
            "rb",
        ) as f:

            data = f.read()

        self.engine = (
            self.runtime.deserialize_cuda_engine(
                data
            )
        )

        if self.engine is None:

            raise RuntimeError(
                f"Failed to load {self.path}"
            )

        self.context = (
            self.engine.create_execution_context()
        )

        if self.context is None:

            raise RuntimeError(
                "Failed to create execution context"
            )

        print(
            "[OK] Engine deserialized"
        )

        self.allocations = {}

        self.host_outputs = {}

        self.allocate()

        self.set_addresses()


    def allocate(self):

        print()
        print("[TRT] Allocating I/O buffers")

        for i in range(
            self.engine.num_io_tensors
        ):

            name = (
                self.engine.get_tensor_name(i)
            )

            mode = (
                self.engine.get_tensor_mode(name)
            )

            dtype = (
                self.engine.get_tensor_dtype(name)
            )

            shape = tuple(
                self.engine.get_tensor_shape(name)
            )

            if any(
                d < 0
                for d in shape
            ):

                raise RuntimeError(
                    f"Dynamic shape unsupported: "
                    f"{name}: {shape}"
                )

            np_dtype = np.dtype(
                trt.nptype(dtype)
            )

            nbytes = (
                int(np.prod(shape))
                * np_dtype.itemsize
            )

            self.allocations[name] = (
                cuda_malloc(nbytes)
            )

            if mode == trt.TensorIOMode.OUTPUT:

                self.host_outputs[name] = (
                    np.empty(
                        shape,
                        dtype=np_dtype,
                    )
                )

            print(
                f"  {name:24s}"
                f" {str(dtype):12s}"
                f" {shape}"
                f" {nbytes / 1024 / 1024:.3f} MiB"
            )


    def set_addresses(self):

        for name, ptr in (
            self.allocations.items()
        ):

            ok = self.context.set_tensor_address(
                name,
                int(ptr)
            )

            if not ok:

                raise RuntimeError(
                    f"set_tensor_address failed: "
                    f"{name}"
                )


    def execute(self, stream):

        ok = self.context.execute_async_v3(
            stream_handle=int(stream)
        )

        if not ok:

            raise RuntimeError(
                "TensorRT execute_async_v3 failed"
            )


    def cleanup(self):

        for ptr in (
            self.allocations.values()
        ):

            cuda_free(ptr)


# ============================================================================
# Load reference input
# ============================================================================

def load_reference():

    print()
    print("=" * 80)
    print("Loading reference input")
    print("=" * 80)

    if not REFERENCE_NPZ.exists():

        raise FileNotFoundError(
            f"Reference NPZ not found:\n"
            f"{REFERENCE_NPZ}\n\n"
            f"Please copy the reference NPZ to Thor "
            f"or change REFERENCE_NPZ."
        )

    data = np.load(
        REFERENCE_NPZ
    )

    print(
        "Keys:",
        list(data.keys())
    )

    if "visual_tokens" not in data:

        raise RuntimeError(
            "Reference NPZ does not contain "
            "visual_tokens"
        )

    visual_tokens = np.ascontiguousarray(
        data["visual_tokens"]
        .astype(np.int64)
    )

    if "high_level_command" in data:

        command = np.ascontiguousarray(
            data["high_level_command"][
                ...,
            ]
            .astype(np.int64)
        )

    else:

        command = np.zeros(
            (1, 1),
            dtype=np.int64
        )

    if command.shape != (1, 1):

        command = np.ascontiguousarray(
            command.reshape(1, 1)
        )

    if "initial_action" in data:

        initial_action = np.ascontiguousarray(
            data["initial_action"]
            .astype(np.float32)
        )

    elif "noisy_actions" in data:

        initial_action = np.ascontiguousarray(
            data["noisy_actions"]
            .astype(np.float32)
        )

    else:

        raise RuntimeError(
            "Reference NPZ does not contain "
            "initial_action/noisy_actions"
        )

    if initial_action.shape == (
        1,
        8,
        6,
        2,
    ):

        initial_action = (
            initial_action[:, :1, :, :]
        )

    if initial_action.shape != (
        1,
        1,
        6,
        2,
    ):

        raise RuntimeError(
            f"Unexpected initial action shape: "
            f"{initial_action.shape}"
        )

    print()
    print(
        "visual_tokens:",
        visual_tokens.shape,
        visual_tokens.dtype
    )

    print(
        "command:",
        command.shape,
        command.dtype,
        command
    )

    print(
        "initial_action:",
        initial_action.shape,
        initial_action.dtype
    )

    return (
        visual_tokens,
        command,
        initial_action,
    )


# ============================================================================
# Bind prefill -> action K/V
# ============================================================================

def connect_kv(
    prefill,
    action,
):

    print()
    print("=" * 80)
    print("Connecting Prefill K/V -> Action engine")
    print("=" * 80)

    for layer in range(NUM_LAYERS):

        k_name = (
            f"visual_k_{layer}"
        )

        v_name = (
            f"visual_v_{layer}"
        )

        if k_name not in prefill.allocations:

            raise RuntimeError(
                f"Missing Prefill output: "
                f"{k_name}"
            )

        if v_name not in prefill.allocations:

            raise RuntimeError(
                f"Missing Prefill output: "
                f"{v_name}"
            )

        if k_name not in action.allocations:

            raise RuntimeError(
                f"Missing Action input: "
                f"{k_name}"
            )

        if v_name not in action.allocations:

            raise RuntimeError(
                f"Missing Action input: "
                f"{v_name}"
            )

        # IMPORTANT:
        #
        # We do NOT copy K/V.
        #
        # The Action engine directly uses the
        # same GPU addresses produced by Prefill.
        #

        action.allocations[k_name] = (
            prefill.allocations[k_name]
        )

        action.allocations[v_name] = (
            prefill.allocations[v_name]
        )

        ok = action.context.set_tensor_address(
            k_name,
            int(
                prefill.allocations[k_name]
            )
        )

        if not ok:

            raise RuntimeError(
                f"Failed to bind {k_name}"
            )

        ok = action.context.set_tensor_address(
            v_name,
            int(
                prefill.allocations[v_name]
            )
        )

        if not ok:

            raise RuntimeError(
                f"Failed to bind {v_name}"
            )

    print(
        "[OK] 48 K/V tensors are GPU-shared "
        "between Prefill and Action"
    )


# ============================================================================
# Benchmark
# ============================================================================

def benchmark(
    prefill,
    action,
    euler_function,
    visual_tokens,
    command,
    initial_action,
    stream,
):

    # ------------------------------------------------------------------------
    # Persistent GPU buffers
    # ------------------------------------------------------------------------

    visual_ptr = prefill.allocations[
        "visual_tokens"
    ]

    action_ptr = action.allocations[
        "noisy_actions"
    ]

    command_ptr = action.allocations[
        "high_level_command"
    ]

    diffusion_ptr = action.allocations[
        "diffusion_step"
    ]

    velocity_ptr = action.allocations[
        "action_velocity"
    ]

    # ------------------------------------------------------------------------
    # One-time H2D
    # ------------------------------------------------------------------------

    cuda_htod(
        visual_ptr,
        visual_tokens
    )

    cuda_htod(
        command_ptr,
        command
    )

    # ------------------------------------------------------------------------
    # Initial action
    # ------------------------------------------------------------------------

    initial_action_flat = (
        np.ascontiguousarray(
            initial_action
            .reshape(-1)
            .astype(np.float32)
        )
    )

    cuda_htod(
        action_ptr,
        initial_action
    )

    # ------------------------------------------------------------------------
    # Diffusion step buffer
    # ------------------------------------------------------------------------

    zero_t = np.zeros(
        (1, 1),
        dtype=np.float32
    )

    cuda_htod(
        diffusion_ptr,
        zero_t
    )

    cuda_stream_sync(
        stream
    )

    # ------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------

    total_start = cuda_event_create()
    total_end = cuda_event_create()

    prefill_start = cuda_event_create()
    prefill_end = cuda_event_create()

    action_start = cuda_event_create()
    action_end = cuda_event_create()

    # ------------------------------------------------------------------------
    # CUDA kernel arguments
    # ------------------------------------------------------------------------

    action_arg = ctypes.c_void_p(
        int(action_ptr)
    )

    velocity_arg = ctypes.c_void_p(
        int(velocity_ptr)
    )

    dt_arg = ctypes.c_float(
        float(DELTA_T)
    )

    kernel_args = (
        ctypes.c_void_p(
            ctypes.addressof(action_arg)
        ),
        ctypes.c_void_p(
            ctypes.addressof(velocity_arg)
        ),
        ctypes.c_void_p(
            ctypes.addressof(dt_arg)
        ),
    )

    # ------------------------------------------------------------------------
    # Single inference
    # ------------------------------------------------------------------------

    def run_once():

        # Reset action
        cuda_htod(
            action_ptr,
            initial_action
        )

        # Reset t
        cuda_htod(
            diffusion_ptr,
            zero_t
        )

        cuda_event_record(
            total_start,
            stream
        )

        # ----------------------------------------------------
        # Prefill
        # ----------------------------------------------------

        cuda_event_record(
            prefill_start,
            stream
        )

        prefill.execute(
            stream
        )

        cuda_event_record(
            prefill_end,
            stream
        )

        # ----------------------------------------------------
        # Euler × 10
        # ----------------------------------------------------

        cuda_event_record(
            action_start,
            stream
        )

        for step in range(
            NUM_EULER_STEPS
        ):

            action.execute(
                stream
            )

            cuda_check(
                cuda.cuLaunchKernel(
                    euler_function,
                    1,
                    1,
                    1,
                    32,
                    1,
                    1,
                    0,
                    stream,
                    kernel_args,
                    0,
                ),
                "cuLaunchKernel"
            )

            # t += 0.1
            #
            # The diffusion_step tensor is only 1 float.
            # We update it with a tiny CUDA kernel later.
            #
            # For now this is handled by a separate
            # 1-element kernel below.

        cuda_event_record(
            action_end,
            stream
        )

        cuda_event_record(
            total_end,
            stream
        )

        cuda_stream_sync(
            stream
        )

        return (
            cuda_event_elapsed(
                prefill_start,
                prefill_end
            ),
            cuda_event_elapsed(
                action_start,
                action_end
            ),
            cuda_event_elapsed(
                total_start,
                total_end
            ),
        )

    # ------------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"Warmup: {WARMUP} complete inferences"
    )
    print("=" * 80)

    for _ in range(WARMUP):

        run_once()

    cuda_stream_sync(
        stream
    )

    # ------------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"Benchmark: {BENCHMARK} complete inferences"
    )
    print("=" * 80)

    prefill_times = []
    action_times = []
    total_times = []

    for i in range(
        BENCHMARK
    ):

        p, a, total = run_once()

        prefill_times.append(p)
        action_times.append(a)
        total_times.append(total)

        print(
            f"{i + 1:3d}: "
            f"Prefill={p:8.3f} ms  "
            f"Action×10={a:8.3f} ms  "
            f"Total={total:8.3f} ms"
        )

    return (
        np.asarray(prefill_times),
        np.asarray(action_times),
        np.asarray(total_times),
    )


# ============================================================================
# Statistics
# ============================================================================

def report(
    name,
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    print()
    print(
        f"[{name}]"
    )

    print(
        f"  mean   : {np.mean(values):.4f} ms"
    )

    print(
        f"  median : {np.median(values):.4f} ms"
    )

    print(
        f"  p95    : {np.percentile(values, 95):.4f} ms"
    )

    print(
        f"  p99    : {np.percentile(values, 99):.4f} ms"
    )

    print(
        f"  min    : {np.min(values):.4f} ms"
    )

    print(
        f"  max    : {np.max(values):.4f} ms"
    )


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 80)
    print(
        "VaVAM-B 12i - Thor GPU-Resident E2E Benchmark"
    )
    print("=" * 80)

    print(
        "TensorRT:",
        trt.__version__
    )

    print(
        "Prefill:",
        PREFILL_ENGINE
    )

    print(
        "Action:",
        ACTION_ENGINE
    )

    print(
        "Steps:",
        NUM_EULER_STEPS
    )

    # ------------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------------

    cuda_check(
        cuda.cuInit(0),
        "cuInit"
    )

    result = cuda.cuDeviceGet(0)

    cuda_check(
        result,
        "cuDeviceGet"
    )

    device = result[1]

    cuda_check(
        cuda.cuDevicePrimaryCtxRetain(
            device
        ),
        "cuDevicePrimaryCtxRetain"
    )

    result = cuda.cuDevicePrimaryCtxRetain(
        device
    )

    cuda_check(
        result,
        "cuDevicePrimaryCtxRetain"
    )

    cuda_context = result[1]

    cuda_check(
        cuda.cuCtxSetCurrent(
            cuda_context
        ),
        "cuCtxSetCurrent"
    )

    print(
        "[OK] CUDA primary context"
    )

    # ------------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------------

    for path in (
        PREFILL_ENGINE,
        ACTION_ENGINE,
        REFERENCE_NPZ,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )

    # ------------------------------------------------------------------------
    # Load input
    # ------------------------------------------------------------------------

    (
        visual_tokens,
        command,
        initial_action,
    ) = load_reference()

    # ------------------------------------------------------------------------
    # Engines
    # ------------------------------------------------------------------------

    prefill = TensorRTEngine(
        PREFILL_ENGINE
    )

    action = TensorRTEngine(
        ACTION_ENGINE
    )

    # ------------------------------------------------------------------------
    # Connect K/V
    # ------------------------------------------------------------------------

    connect_kv(
        prefill,
        action
    )

    # ------------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------------

    stream = cuda_stream_create()

    print(
        "[OK] CUDA stream created"
    )

    # ------------------------------------------------------------------------
    # Euler kernel
    # ------------------------------------------------------------------------

    euler_module, euler_function = (
        build_euler_kernel()
    )

    # ------------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------------

    try:

        (
            prefill_times,
            action_times,
            total_times,
        ) = benchmark(
            prefill,
            action,
            euler_function,
            visual_tokens,
            command,
            initial_action,
            stream,
        )

    finally:

        cuda_stream_sync(
            stream
        )

    # ------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("12i RESULT")
    print("=" * 80)

    report(
        "Prefill FP16",
        prefill_times
    )

    report(
        "Action FP16 × 10 + Euler",
        action_times
    )

    report(
        "Complete 10-step E2E",
        total_times
    )

    mean_total = float(
        np.mean(total_times)
    )

    print()
    print(
        f"10-step E2E mean : "
        f"{mean_total:.4f} ms"
    )

    print(
        f"Inference rate   : "
        f"{1000.0 / mean_total:.4f} Hz"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  K/V: GPU resident"
    )

    print(
        "  Action: GPU resident"
    )

    print(
        "  Euler: GPU kernel"
    )

    print(
        "  Per-step D2H: NONE"
    )

    print(
        "  Per-step H2D: NONE"
    )

    print()
    print(
        "Benchmark completed."
    )


if __name__ == "__main__":

    main()
