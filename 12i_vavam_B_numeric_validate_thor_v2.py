#!/usr/bin/env python3
"""
VaVAM-B v10 Thor numerical validation v2.

Purpose:
  1) Locate Prefill K/V numerical discrepancy layer-by-layer.
  2) Locate Action/Euler discrepancy step-by-step.
  3) Compare final trajectory.
  4) Print absolute + relative diagnostics without declaring failure solely
     from the K/V discrepancy.

Reference:
  PC FP32 ONNX + FP32 NumPy Euler
Deployment:
  Thor FP16 TensorRT + GPU-resident CUDA Euler

Expected files:
  ONNX/thor_v10_numeric_reference.npz
  Engines/vavam_joint_kv_prefill_B_v10_fp16.engine
  Engines/vavam_joint_action_B_fp16.engine
"""

import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cuda, nvrtc


ROOT = Path.home() / "vblkdev2" / "VaVAM_Thor"
PREFILL_ENGINE = ROOT / "Engines" / "vavam_joint_kv_prefill_B_v10_fp16.engine"
ACTION_ENGINE = ROOT / "Engines" / "vavam_joint_action_B_fp16.engine"
REF = ROOT / "NPZ" / "thor_v10_numeric_reference.npz"

NL = 24
STEPS = 10
DT = 0.1
ACTION_SHAPE = (1, 1, 6, 2)
KV_SHAPE = (1, 8, 4608, 128)


def ck(result, name):
    err = result[0] if isinstance(result, tuple) else result
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {err}")
    return result


def malloc(nbytes):
    return ck(cuda.cuMemAlloc(int(nbytes)), "cuMemAlloc")[1]


def h2d(ptr, arr):
    arr = np.ascontiguousarray(arr)
    ck(cuda.cuMemcpyHtoD(int(ptr), arr, int(arr.nbytes)), "H2D")


def d2h(arr, ptr):
    arr = np.ascontiguousarray(arr)
    ck(cuda.cuMemcpyDtoH(arr, int(ptr), int(arr.nbytes)), "D2H")


def sync(stream):
    ck(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize")


class Engine:
    def __init__(self, path, skip_names=()):
        self.owned = {}
        self.skip = set(skip_names)

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as f:
            blob = f.read()

        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize: {path}")

        self.context = self.engine.create_execution_context()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)

            if name in self.skip:
                continue

            shape = tuple(self.engine.get_tensor_shape(name))
            if any(x < 0 for x in shape):
                raise RuntimeError(f"Dynamic shape not supported here: {name} {shape}")

            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            self.owned[name] = malloc(nbytes)

        for name, ptr in self.owned.items():
            if not self.context.set_tensor_address(name, int(ptr)):
                raise RuntimeError(f"set_tensor_address failed: {name}")

    def set_address(self, name, ptr):
        if not self.context.set_tensor_address(name, int(ptr)):
            raise RuntimeError(f"set_tensor_address failed: {name}")

    def execute(self, stream):
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")


CUDA_SRC = r"""
#include <cuda_fp16.h>

extern "C" __global__
void euler_f32_f32(float* action, const float* velocity, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12)
        action[i] = action[i] + dt * velocity[i];
}

extern "C" __global__
void add_t_f32(float* t, float dt)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        t[0] = t[0] + dt;
}
"""


def compile_cuda_helpers():
    print("\n" + "=" * 80)
    print("Compiling CUDA helper kernels with NVRTC")
    print("=" * 80)

    program = nvrtc.nvrtcCreateProgram(
        CUDA_SRC.encode("utf-8"),
        b"vavam_12i_v2.cu",
        0,
        [],
        [],
    )[1]

    options = [b"--include-path=/usr/local/cuda/include"]

    result = nvrtc.nvrtcCompileProgram(
        program,
        len(options),
        options,
    )
    err = result[0] if isinstance(result, tuple) else result

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        size_result = nvrtc.nvrtcGetProgramLogSize(program)
        size_err = size_result[0] if isinstance(size_result, tuple) else size_result

        log_text = "<unable to retrieve NVRTC log>"
        if size_err == nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log_size = size_result[1]
            log_buffer = bytearray(log_size)
            log_result = nvrtc.nvrtcGetProgramLog(program, log_buffer)
            log_err = log_result[0] if isinstance(log_result, tuple) else log_result
            if log_err == nvrtc.nvrtcResult.NVRTC_SUCCESS:
                log_text = bytes(log_buffer).rstrip(b"\x00").decode(
                    "utf-8", errors="replace"
                )

        raise RuntimeError(
            f"NVRTC compilation failed: {err}\nCompiler log:\n{log_text}"
        )

    ptx_size = nvrtc.nvrtcGetPTXSize(program)[1]
    ptx_buffer = bytearray(ptx_size)
    nvrtc.nvrtcGetPTX(program, ptx_buffer)

    module = ck(
        cuda.cuModuleLoadData(bytes(ptx_buffer)),
        "cuModuleLoadData",
    )[1]

    euler_func = ck(
        cuda.cuModuleGetFunction(module, b"euler_f32_f32"),
        "cuModuleGetFunction(euler)",
    )[1]

    t_func = ck(
        cuda.cuModuleGetFunction(module, b"add_t_f32"),
        "cuModuleGetFunction(add_t)",
    )[1]

    print("[OK] CUDA helper kernels ready")
    return euler_func, t_func


def launch_kernel(func, stream, args, threads):
    ck(
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


def make_euler_args(action_ptr, velocity_ptr):
    return (
        (int(action_ptr), int(velocity_ptr), float(DT)),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float),
    )


def make_t_args(t_ptr):
    return (
        (int(t_ptr), float(DT)),
        (ctypes.c_void_p, ctypes.c_float),
    )


def metrics(a, b):
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    diff = np.abs(a64 - b64)

    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rmse = float(np.sqrt(np.mean(diff * diff)))

    denom = np.maximum(np.abs(b64), 1e-8)
    rel = diff / denom
    max_rel = float(rel.max())

    return max_abs, mean_abs, rmse, max_rel


def print_matrix_diff(pc, thor):
    diff = thor.astype(np.float64) - pc.astype(np.float64)

    print("\nPC FP32 ONNX final:")
    print(pc.reshape(6, 2))

    print("\nThor FP16 final:")
    print(thor.reshape(6, 2))

    print("\nThor - PC:")
    print(diff.reshape(6, 2))

    print("\nPer trajectory point |dx| / |dy|:")
    for i in range(6):
        print(
            f"  point {i}: "
            f"dx={abs(diff.reshape(6,2)[i,0]):.8e}, "
            f"dy={abs(diff.reshape(6,2)[i,1]):.8e}"
        )


def main():
    if not REF.exists():
        raise FileNotFoundError(f"Missing reference: {REF}")
    if not PREFILL_ENGINE.exists():
        raise FileNotFoundError(f"Missing engine: {PREFILL_ENGINE}")
    if not ACTION_ENGINE.exists():
        raise FileNotFoundError(f"Missing engine: {ACTION_ENGINE}")

    d = np.load(REF, allow_pickle=False)

    visual = d["visual_tokens"].astype(np.int64)
    initial_action = d["initial_action"].astype(np.float32)
    command = d["high_level_command"].astype(np.int64)
    t0 = d["initial_diffusion_step"].astype(np.float32)

    reference_velocity = d["reference_velocity_history"].astype(np.float32)
    reference_action = d["reference_action_history"].astype(np.float32)
    reference_final = d["reference_final_action"].astype(np.float32)

    # ------------------------------------------------------------------
    # CUDA / TensorRT initialization
    # ------------------------------------------------------------------
    ck(cuda.cuInit(0), "cuInit")
    device = ck(cuda.cuDeviceGet(0), "cuDeviceGet")[1]
    primary_ctx = ck(
        cuda.cuDevicePrimaryCtxRetain(device),
        "cuDevicePrimaryCtxRetain",
    )[1]
    ck(cuda.cuCtxSetCurrent(primary_ctx), "cuCtxSetCurrent")
    stream = ck(cuda.cuStreamCreate(0), "cuStreamCreate")[1]

    kv_names = (
        [f"visual_k_{i}" for i in range(NL)]
        + [f"visual_v_{i}" for i in range(NL)]
    )

    prefill = Engine(PREFILL_ENGINE)
    action_engine = Engine(ACTION_ENGINE, skip_names=kv_names)

    # Share exactly the Prefill output addresses with Action inputs.
    for name in kv_names:
        action_engine.set_address(name, prefill.owned[name])

    print("\n[OK] 48 visual K/V tensors are GPU-resident and shared")

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------
    h2d(prefill.owned["visual_tokens"], visual)
    prefill.execute(stream)
    sync(stream)

    print("\n" + "=" * 80)
    print("PREFILL K/V PER-LAYER VALIDATION")
    print("=" * 80)

    layer_rows = []

    for layer in range(NL):
        row = {"layer": layer}

        for kind in ("k", "v"):
            name = f"visual_{kind}_{layer}"
            got = np.empty(KV_SHAPE, dtype=np.float32)
            d2h(got, prefill.owned[name])

            ref = d[f"reference_visual_{kind}_{layer}"].astype(np.float32)
            mx, mean, rmse, rel = metrics(got, ref)

            row[f"{kind}_max"] = mx
            row[f"{kind}_mean"] = mean
            row[f"{kind}_rmse"] = rmse
            row[f"{kind}_rel"] = rel

            # Also print the actual ranges. This helps identify whether
            # a large absolute error comes from a large-magnitude tensor.
            row[f"{kind}_pc_min"] = float(ref.min())
            row[f"{kind}_pc_max"] = float(ref.max())
            row[f"{kind}_thor_min"] = float(got.min())
            row[f"{kind}_thor_max"] = float(got.max())

        layer_rows.append(row)

        print(
            f"L{layer:02d} | "
            f"K max={row['k_max']:.6e}, mean={row['k_mean']:.6e}, "
            f"RMSE={row['k_rmse']:.6e}, rel={row['k_rel']:.6e} | "
            f"V max={row['v_max']:.6e}, mean={row['v_mean']:.6e}, "
            f"RMSE={row['v_rmse']:.6e}, rel={row['v_rel']:.6e}"
        )

    worst_k = max(layer_rows, key=lambda x: x["k_max"])
    worst_v = max(layer_rows, key=lambda x: x["v_max"])

    print("\nWorst K layer:")
    print(
        f"  L{worst_k['layer']:02d}: "
        f"max_abs={worst_k['k_max']:.8e}, "
        f"mean_abs={worst_k['k_mean']:.8e}, "
        f"RMSE={worst_k['k_rmse']:.8e}"
    )
    print(
        f"  PC range  = [{worst_k['k_pc_min']:.8e}, "
        f"{worst_k['k_pc_max']:.8e}]"
    )
    print(
        f"  Thor range= [{worst_k['k_thor_min']:.8e}, "
        f"{worst_k['k_thor_max']:.8e}]"
    )

    print("\nWorst V layer:")
    print(
        f"  L{worst_v['layer']:02d}: "
        f"max_abs={worst_v['v_max']:.8e}, "
        f"mean_abs={worst_v['v_mean']:.8e}, "
        f"RMSE={worst_v['v_rmse']:.8e}"
    )
    print(
        f"  PC range  = [{worst_v['v_pc_min']:.8e}, "
        f"{worst_v['v_pc_max']:.8e}]"
    )
    print(
        f"  Thor range= [{worst_v['v_thor_min']:.8e}, "
        f"{worst_v['v_thor_max']:.8e}]"
    )

    # ------------------------------------------------------------------
    # Action + Euler
    # ------------------------------------------------------------------
    action_ptr = action_engine.owned["noisy_actions"]
    command_ptr = action_engine.owned["high_level_command"]
    t_ptr = action_engine.owned["diffusion_step"]
    velocity_ptr = action_engine.owned["actions"]

    h2d(action_ptr, initial_action)
    h2d(command_ptr, command)
    h2d(t_ptr, t0)
    sync(stream)

    euler_func, t_func = compile_cuda_helpers()

    euler_args = make_euler_args(action_ptr, velocity_ptr)
    t_args = make_t_args(t_ptr)

    thor_velocity = []
    thor_action = []

    print("\n" + "=" * 80)
    print("ACTION / EULER PER-STEP VALIDATION")
    print("=" * 80)

    for step in range(STEPS):
        # Action TRT inference
        action_engine.execute(stream)
        sync(stream)

        velocity = np.empty(ACTION_SHAPE, dtype=np.float32)
        d2h(velocity, velocity_ptr)

        # GPU-resident Euler update and t += DT
        launch_kernel(euler_func, stream, euler_args, threads=32)
        launch_kernel(t_func, stream, t_args, threads=1)
        sync(stream)

        action = np.empty(ACTION_SHAPE, dtype=np.float32)
        d2h(action, action_ptr)

        thor_velocity.append(velocity.copy())
        thor_action.append(action.copy())

        vm = metrics(velocity, reference_velocity[step])
        am = metrics(action, reference_action[step])

        print(
            f"Step {step:02d} | "
            f"VEL max={vm[0]:.6e}, mean={vm[1]:.6e}, "
            f"RMSE={vm[2]:.6e}, rel={vm[3]:.6e} | "
            f"ACTION max={am[0]:.6e}, mean={am[1]:.6e}, "
            f"RMSE={am[2]:.6e}, rel={am[3]:.6e}"
        )

    thor_velocity = np.stack(thor_velocity)
    thor_action = np.stack(thor_action)
    thor_final = thor_action[-1]

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    velocity_all = metrics(thor_velocity, reference_velocity)
    action_all = metrics(thor_action, reference_action)
    final_m = metrics(thor_final, reference_final)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"All velocity steps | max_abs={velocity_all[0]:.8e}, "
        f"mean_abs={velocity_all[1]:.8e}, RMSE={velocity_all[2]:.8e}, "
        f"max_rel={velocity_all[3]:.8e}"
    )
    print(
        f"All action steps   | max_abs={action_all[0]:.8e}, "
        f"mean_abs={action_all[1]:.8e}, RMSE={action_all[2]:.8e}, "
        f"max_rel={action_all[3]:.8e}"
    )
    print(
        f"Final action       | max_abs={final_m[0]:.8e}, "
        f"mean_abs={final_m[1]:.8e}, RMSE={final_m[2]:.8e}, "
        f"max_rel={final_m[3]:.8e}"
    )

    print_matrix_diff(reference_final, thor_final)

    print("\nFinal ranges:")
    print(
        f"  PC   : [{reference_final.min():.8e}, "
        f"{reference_final.max():.8e}]"
    )
    print(
        f"  Thor : [{thor_final.min():.8e}, "
        f"{thor_final.max():.8e}]"
    )

    # Do NOT use the K/V discrepancy as an automatic deployment failure.
    # The v2 script intentionally separates:
    #   - Prefill discrepancy
    #   - Action discrepancy
    #   - Final trajectory discrepancy
    #
    # Thresholds:
    #   final max_abs <= 1e-3 -> trajectory PASS
    #   final max_abs <= 2e-3 -> trajectory PASS (relaxed)
    #   otherwise -> investigate
    final_pass = np.isfinite(thor_final).all() and final_m[0] <= 1e-3

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if final_pass:
        print("[PASS] Final trajectory max_abs <= 1e-3")
        print(
            "[INFO] Prefill K/V discrepancies are reported separately and "
            "must be investigated by layer."
        )
    else:
        print("[WARN] Final trajectory exceeds 1e-3; investigate further.")

    print("\nNumerical validation v2 completed.")


if __name__ == "__main__":
    main()
