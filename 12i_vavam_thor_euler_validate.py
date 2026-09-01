import os
import argparse
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
        raise RuntimeError(
            f"{name} failed: {err}"
        )


def init_cuda():

    cuda_check(
        cuda.cuInit(0),
        "cuInit",
    )

    result = cuda.cuDeviceGet(0)

    cuda_check(
        result,
        "cuDeviceGet",
    )

    device = result[1]

    result = cuda.cuDevicePrimaryCtxRetain(
        device
    )

    cuda_check(
        result,
        "cuDevicePrimaryCtxRetain",
    )

    context = result[1]

    cuda_check(
        cuda.cuCtxSetCurrent(context),
        "cuCtxSetCurrent",
    )

    print(
        "[OK] CUDA primary context created/set"
    )


def cuda_malloc(nbytes):

    result = cuda.cuMemAlloc(
        int(nbytes)
    )

    cuda_check(
        result,
        "cuMemAlloc",
    )

    return result[1]


def cuda_free(ptr):

    if ptr is None:
        return

    result = cuda.cuMemFree(
        ptr
    )

    cuda_check(
        result,
        "cuMemFree",
    )


def cuda_memcpy_htod(
    device_ptr,
    host_array,
):

    arr = np.ascontiguousarray(
        host_array
    )

    nbytes = int(
        arr.nbytes
    )

    result = cuda.cuMemcpyHtoD(
        int(device_ptr),
        arr,
        nbytes,
    )

    cuda_check(
        result,
        "cuMemcpyHtoD",
    )


def cuda_memcpy_dtoh(
    host_array,
    device_ptr,
):

    arr = np.ascontiguousarray(
        host_array
    )

    nbytes = int(
        arr.nbytes
    )

    result = cuda.cuMemcpyDtoH(
        arr,
        int(device_ptr),
        nbytes,
    )

    cuda_check(
        result,
        "cuMemcpyDtoH",
    )

    return arr


# ============================================================================
# Load NPZ
# ============================================================================

def load_reference(
    reference_path,
):

    print()
    print("=" * 80)
    print("Loading 12h reference")
    print("=" * 80)

    data = np.load(
        reference_path
    )

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

            raise RuntimeError(
                f"Missing NPZ field: {name}"
            )

    print()
    print(
        f"  scene_id       : "
        f"{data['scene_id']}"
    )

    print(
        f"  window_index   : "
        f"{data['window_index']}"
    )

    visual_tokens = (
        data["visual_tokens"]
        .astype(np.int64)
    )

    initial_action = (
        data["initial_action"]
        .astype(np.float32)
    )

    high_level_command = (
        data["high_level_command"]
        .astype(np.int64)
    )

    initial_diffusion_step = (
        data["initial_diffusion_step"]
        .astype(np.float32)
    )

    pytorch_velocity = (
        data["pytorch_velocity_history"]
        .astype(np.float32)
    )

    pytorch_action = (
        data["pytorch_action_history"]
        .astype(np.float32)
    )

    pytorch_final = (
        data["pytorch_final_trajectory"]
        .astype(np.float32)
    )

    print()
    print(
        "[Reference inputs]"
    )

    print(
        f"  visual_tokens      : "
        f"{visual_tokens.shape} "
        f"{visual_tokens.dtype}"
    )

    print(
        f"  initial_action     : "
        f"{initial_action.shape} "
        f"{initial_action.dtype}"
    )

    print(
        f"  high_level_command : "
        f"{high_level_command.shape} "
        f"{high_level_command.dtype}"
    )

    print(
        f"  diffusion_step     : "
        f"{initial_diffusion_step.shape} "
        f"{initial_diffusion_step.dtype}"
    )

    return {
        "scene_id":
            str(data["scene_id"]),

        "window_index":
            int(data["window_index"]),

        "visual_tokens":
            visual_tokens,

        "initial_action":
            initial_action,

        "high_level_command":
            high_level_command,

        "initial_diffusion_step":
            initial_diffusion_step,

        "pytorch_velocity":
            pytorch_velocity,

        "pytorch_action":
            pytorch_action,

        "pytorch_final":
            pytorch_final,
    }


# ============================================================================
# Load scene .npy
# ============================================================================

def load_scene_tokens(
    token_dir,
    scene_id,
):

    token_dir = Path(
        token_dir
    )

    files = sorted(
        token_dir.glob(
            f"{scene_id}__CAM_FRONT__*.npy"
        )
    )

    if not files:

        raise RuntimeError(
            f"No token files found for scene: "
            f"{scene_id}"
        )

    parsed = []

    for path in files:

        name = path.stem

        parts = name.split(
            "__"
        )

        if len(parts) != 3:
            continue

        timestamp = parts[2]

        parsed.append(
            (
                int(timestamp),
                path,
            )
        )

    parsed.sort(
        key=lambda x: x[0]
    )

    return parsed


def build_visual_tokens(
    files,
    window_index,
):

    start = window_index

    end = (
        start
        + CONTEXT_LENGTH
    )

    if end > len(files):

        raise RuntimeError(
            f"Not enough frames for "
            f"window {window_index}"
        )

    selected = files[
        start:end
    ]

    tokens = []

    print()
    print(
        "[CAM_FRONT tokens]"
    )

    for timestamp, path in selected:

        x = np.load(
            path
        )

        if x.shape != (
            18,
            32,
        ):

            raise RuntimeError(
                f"Unexpected token shape: "
                f"{path.name} "
                f"{x.shape}"
            )

        print(
            f"  {timestamp} "
            f"{x.shape} "
            f"{x.dtype}"
        )

        tokens.append(
            x.astype(np.int64)
        )

    visual_tokens = np.stack(
        tokens,
        axis=0,
    )

    visual_tokens = (
        visual_tokens[
            None,
            ...
        ]
    )

    return visual_tokens


# ============================================================================
# TensorRT Runner
# ============================================================================

class TRTRunner:

    def __init__(
        self,
        engine_path,
    ):

        self.allocations = {}

        self.host_outputs = {}

        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        self.runtime = trt.Runtime(
            self.logger
        )

        with open(
            engine_path,
            "rb",
        ) as f:

            engine_data = f.read()

        self.engine = (
            self.runtime
            .deserialize_cuda_engine(
                engine_data
            )
        )

        if self.engine is None:

            raise RuntimeError(
                "Failed to deserialize TensorRT engine."
            )

        print(
            "[OK] TensorRT engine deserialized"
        )

        self.context = (
            self.engine
            .create_execution_context()
        )

        if self.context is None:

            raise RuntimeError(
                "Failed to create execution context."
            )

        self.inspect_engine()

        self.allocate()

        result = cuda.cuStreamCreate(0)

        cuda_check(
            result,
            "cuStreamCreate",
        )

        self.stream = result[1]

        print(
            "[OK] CUDA stream created"
        )

    def inspect_engine(self):

        print()
        print(
            "[TRT] Engine I/O"
        )

        for i in range(
            self.engine.num_io_tensors
        ):

            name = (
                self.engine
                .get_tensor_name(i)
            )

            mode = (
                self.engine
                .get_tensor_mode(name)
            )

            dtype = (
                self.engine
                .get_tensor_dtype(name)
            )

            shape = tuple(
                self.engine
                .get_tensor_shape(name)
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

    def allocate(self):

        print()
        print(
            "[TRT] Allocating buffers..."
        )

        for i in range(
            self.engine.num_io_tensors
        ):

            name = (
                self.engine
                .get_tensor_name(i)
            )

            mode = (
                self.engine
                .get_tensor_mode(name)
            )

            dtype = (
                self.engine
                .get_tensor_dtype(name)
            )

            shape = tuple(
                self.engine
                .get_tensor_shape(name)
            )

            np_dtype = np.dtype(
                trt.nptype(dtype)
            )

            nbytes = (
                int(np.prod(shape))
                * np_dtype.itemsize
            )

            ptr = cuda_malloc(
                nbytes
            )

            self.allocations[
                name
            ] = ptr

            print()
            print("[TRT] Setting tensor addresses...")
            self.set_addresses()
            print("[OK] Tensor addresses configured")

            print(
                f"  [OK] {name} "
                f"shape={shape} "
                f"dtype={dtype} "
                f"nbytes={nbytes}"
            )

            if mode == trt.TensorIOMode.OUTPUT:

                self.host_outputs[
                    name
                ] = np.empty(
                    shape,
                    dtype=np_dtype,
                )

    def set_addresses(self):

        for name, ptr in (
            self.allocations.items()
        ):

            ok = (
                self.context
                .set_tensor_address(
                    name,
                    int(ptr),
                )
            )

            if not ok:

                raise RuntimeError(
                    f"Failed to set tensor address: "
                    f"{name}"
                )

    def infer(
        self,
        visual_tokens,
        noisy_actions,
        high_level_command,
        diffusion_step,
    ):

        inputs = {
            "visual_tokens":
                np.ascontiguousarray(
                    visual_tokens,
                    dtype=np.int64,
                ),

            "noisy_actions":
                np.ascontiguousarray(
                    noisy_actions,
                    dtype=np.float32,
                ),

            "high_level_command":
                np.ascontiguousarray(
                    high_level_command,
                    dtype=np.int64,
                ),

            "diffusion_step":
                np.ascontiguousarray(
                    diffusion_step,
                    dtype=np.float32,
                ),
        }

        for name, arr in (
            inputs.items()
        ):

            cuda_memcpy_htod(
                self.allocations[name],
                arr,
            )

        ok = (
            self.context
            .execute_async_v3(
                self.stream
            )
        )

        if not ok:

            raise RuntimeError(
                "TensorRT execute_async_v3 failed"
            )

        cuda_check(
            cuda.cuStreamSynchronize(
                self.stream
            ),
            "cuStreamSynchronize",
        )

        output_name = (
            "action_velocity"
        )

        output = self.host_outputs[
            output_name
        ]

        cuda_memcpy_dtoh(
            output,
            self.allocations[
                output_name
            ],
        )

        return output.copy()

    def cleanup(self):

        print()
        print(
            "[TRT] Cleaning buffers..."
        )

        for name, ptr in (
            self.allocations.items()
        ):

            cuda_free(
                ptr
            )

            print(
                f"  [OK] {name}"
            )

        if hasattr(
            self,
            "stream",
        ):

            cuda_check(
                cuda.cuStreamDestroy(
                    self.stream
                ),
                "cuStreamDestroy",
            )

            print(
                "[OK] CUDA stream destroyed"
            )


# ============================================================================
# Euler
# ============================================================================

def run_trt_euler(
    runner,
    visual_tokens,
    initial_action,
    high_level_command,
    initial_diffusion_step,
):

    print()
    print(
        "=" * 80
    )
    print(
        "Thor TensorRT + Euler"
    )
    print(
        "=" * 80
    )

    action = (
        initial_action.copy()
    )

    velocity_history = []

    action_history = [
        action.copy()
    ]

    diffusion_step = (
        initial_diffusion_step.copy()
    )

    for step in range(
        NUM_EULER_STEPS
    ):

        print()
        print(
            f"Step {step + 1}/"
            f"{NUM_EULER_STEPS}"
        )

        print(
            f"  t = "
            f"{float(diffusion_step.reshape(-1)[0]):.6f}"
        )

        velocity = runner.infer(
            visual_tokens,
            action,
            high_level_command,
            diffusion_step,
        )

        velocity_history.append(
            velocity.copy()
        )

        action = (
            action
            + DELTA_T * velocity
        ).astype(
            np.float32
        )

        action_history.append(
            action.copy()
        )

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

        diffusion_step = (
            diffusion_step
            + DELTA_T
        ).astype(
            np.float32
        )

    return (
        np.stack(
            velocity_history,
            axis=0,
        ),
        np.stack(
            action_history,
            axis=0,
        ),
        action.copy(),
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
    print(
        "=" * 80
    )
    print(
        "Numerical Comparison"
    )
    print(
        "=" * 80
    )

    velocity_error = np.abs(
        pt_velocity
        - thor_velocity
    )

    action_error = np.abs(
        pt_action
        - thor_action
    )

    final_error = np.abs(
        pt_final
        - thor_final
    )

    print()
    print(
        "[Velocity history]"
    )

    print(
        f"  max abs error : "
        f"{velocity_error.max():.10e}"
    )

    print(
        f"  mean abs error: "
        f"{velocity_error.mean():.10e}"
    )

    print()
    print(
        "[Action history]"
    )

    print(
        f"  max abs error : "
        f"{action_error.max():.10e}"
    )

    print(
        f"  mean abs error: "
        f"{action_error.mean():.10e}"
    )

    print()
    print(
        "[Final trajectory]"
    )

    print(
        f"  max abs error : "
        f"{final_error.max():.10e}"
    )

    print(
        f"  mean abs error: "
        f"{final_error.mean():.10e}"
    )

    return max(
        velocity_error.max(),
        action_error.max(),
        final_error.max(),
    )


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "engine",
        type=Path,
    )

    parser.add_argument(
        "reference",
        type=Path,
    )

    parser.add_argument(
        "token_dir",
        type=Path,
    )

    parser.add_argument(
        "--scene",
        required=True,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    print(
        "=" * 80
    )

    print(
        "VaVAM 12i - Thor "
        "CAM_FRONT .npy + 12h .npz "
        "Euler Validation"
    )

    print(
        "=" * 80
    )

    print()
    print(
        f"Engine    : {args.engine}"
    )

    print(
        f"Reference : {args.reference}"
    )

    print(
        f"Tokens    : {args.token_dir}"
    )

    print(
        f"Scene     : {args.scene}"
    )

    print(
        f"Window    : {args.window}"
    )

    # ------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------

    init_cuda()

    # ------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------

    ref = load_reference(
        args.reference
    )

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

    # ------------------------------------------------------------
    # Load actual .npy tokens
    # ------------------------------------------------------------

    files = load_scene_tokens(
        args.token_dir,
        args.scene,
    )

    print()
    print(
        f"[OK] Scene frames: "
        f"{len(files)}"
    )

    visual_tokens_npy = (
        build_visual_tokens(
            files,
            args.window,
        )
    )

    print()
    print(
        "[Visual Tokens from .npy]"
    )

    print(
        f"  shape : "
        f"{visual_tokens_npy.shape}"
    )

    print(
        f"  dtype : "
        f"{visual_tokens_npy.dtype}"
    )

    print(
        f"  min   : "
        f"{visual_tokens_npy.min()}"
    )

    print(
        f"  max   : "
        f"{visual_tokens_npy.max()}"
    )

    # ------------------------------------------------------------
    # Verify .npy == .npz visual tokens
    # ------------------------------------------------------------

    token_diff = np.abs(
        visual_tokens_npy.astype(
            np.int64
        )
        -
        ref["visual_tokens"]
    )

    print()
    print(
        "[Visual Token Verification]"
    )

    print(
        f"  max abs difference : "
        f"{token_diff.max()}"
    )

    if token_diff.max() != 0:

        raise RuntimeError(
            "CAM_FRONT .npy tokens do not "
            "match 12h NPZ visual_tokens."
        )

    print(
        "[PASS] .npy visual tokens "
        "match .npz reference exactly"
    )

    # ------------------------------------------------------------
    # TensorRT
    # ------------------------------------------------------------

    runner = TRTRunner(
        args.engine
    )

    try:

        (
            thor_velocity,
            thor_action,
            thor_final,
        ) = run_trt_euler(
            runner,
            visual_tokens_npy,
            ref["initial_action"],
            ref["high_level_command"],
            ref["initial_diffusion_step"],
        )

    finally:

        runner.cleanup()

    # ------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------

    max_error = compare(
        ref["pytorch_velocity"],
        thor_velocity,
        ref["pytorch_action"],
        thor_action,
        ref["pytorch_final"],
        thor_final,
    )

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    print()
    print(
        "=" * 80
    )
    print(
        "12i RESULT"
    )
    print(
        "=" * 80
    )

    print()

    if max_error <= ATOL:

        print(
            "[PASS] Thor TensorRT + Euler "
            "matches PC PyTorch reference "
            "within 1e-4."
        )

    elif max_error <= 1e-3:

        print(
            "[PASS-WARN] Thor result matches "
            "within 1e-3."
        )

    else:

        print(
            "[FAIL] Thor result differs "
            "from PC reference by more than 1e-3."
        )


if __name__ == "__main__":

    main()
