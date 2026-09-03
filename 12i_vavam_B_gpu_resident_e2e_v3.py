#!/usr/bin/env python3
# VaVAM-B 12i v3 — Thor integrated GPU-resident benchmark
#
# Measures:
#   1) Prefill FP16
#   2) Action FP16 x10 + GPU Euler
#   3) Complete Prefill + Action x10 + Euler
#
# visual_tokens is uploaded once before each measured inference.
# Prefill K/V outputs are directly reused as Action K/V inputs on GPU.
# No K/V copy occurs.
# noisy_actions and diffusion_step stay on GPU during all 10 steps.
# Euler and t += 0.1 are CUDA kernels.
# Final D2H is outside measured GPU timing.

import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cuda, nvrtc


ROOT = Path.home() / "vblkdev2" / "VaVAM_Thor"

PREFILL_ENGINE = ROOT / "Engines" / "vavam_joint_kv_prefill_B_v10_fp16.engine"
ACTION_ENGINE = ROOT / "Engines" / "vavam_joint_action_B_fp16.engine"
REFERENCE_NPZ = ROOT / "NPZ" / "vavam_joint_kv_prefill_B_v10_addmask_reference.npz"

NUM_LAYERS = 24
NUM_STEPS = 10
DT = 0.1

WARMUP = 5
BENCHMARK = 30

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


def elapsed_ms(start, end):
    return float(check(cuda.cuEventElapsedTime(start, end), "cuEventElapsedTime")[1])


class Engine:
    def __init__(self, path, skip_names=None):
        self.path = Path(path)
        self.skip_names = set(skip_names or [])
        self.owned = {}
        self.host_outputs = {}

        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)

        with open(self.path, "rb") as f:
            blob = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {self.path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create context for {self.path}")

        print()
        print("=" * 80)
        print(f"Loaded: {self.path.name}")
        print("=" * 80)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))

            print(
                f"{'IN ' if mode == trt.TensorIOMode.INPUT else 'OUT'} "
                f"{name:24s} dtype={dtype} shape={shape}"
            )

            if any(d < 0 for d in shape):
                raise RuntimeError(f"Dynamic shape unsupported: {name} {shape}")

            if name in self.skip_names:
                continue

            np_dtype = np.dtype(trt.nptype(dtype))
            nbytes = int(np.prod(shape)) * np_dtype.itemsize
            self.owned[name] = malloc(nbytes)

            if mode == trt.TensorIOMode.OUTPUT:
                self.host_outputs[name] = np.empty(shape, dtype=np_dtype)

        self.set_owned_addresses()

    def set_owned_addresses(self):
        for name, ptr in self.owned.items():
            ok = self.context.set_tensor_address(name, int(ptr))
            if not ok:
                raise RuntimeError(f"set_tensor_address failed: {name}")

    def set_address(self, name, ptr):
        ok = self.context.set_tensor_address(name, int(ptr))
        if not ok:
            raise RuntimeError(f"set_tensor_address failed: {name}")

    def execute(self, stream):
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
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
void euler_f32_f32(float* action, const float* vel, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) action[i] = action[i] + dt * vel[i];
}

extern "C" __global__
void euler_f32_f16(float* action, const __half* vel, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) action[i] = action[i] + dt * __half2float(vel[i]);
}

extern "C" __global__
void euler_f16_f32(__half* action, const float* vel, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) {
        float a = __half2float(action[i]);
        action[i] = __float2half(a + dt * vel[i]);
    }
}

extern "C" __global__
void euler_f16_f16(__half* action, const __half* vel, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 12) {
        float a = __half2float(action[i]);
        float v = __half2float(vel[i]);
        action[i] = __float2half(a + dt * v);
    }
}

extern "C" __global__
void add_t_f32(float* t, float dt)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        t[0] = t[0] + dt;
}

extern "C" __global__
void add_t_f16(__half* t, float dt)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        t[0] = __float2half(__half2float(t[0]) + dt);
}
'''


def compile_cuda_module():
    print()
    print("=" * 80)
    print("Compiling CUDA helper kernels with NVRTC")
    print("=" * 80)

    result = nvrtc.nvrtcCreateProgram(
        KERNEL_SRC.encode("utf-8"),
        b"vavam_12i.cu",
        0,
        [],
        [],
    )
    err, program = result if isinstance(result, tuple) else (result, None)

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram failed: {err}")

    # Thor's NVRTC does not automatically search CUDA headers.
    # Explicitly provide the CUDA include directory so <cuda_fp16.h> resolves.
    compile_options = [
        b"--include-path=/usr/local/cuda/include",
    ]

    result = nvrtc.nvrtcCompileProgram(
        program,
        len(compile_options),
        compile_options,
    )
    err = result[0] if isinstance(result, tuple) else result

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        # cuda.bindings NVRTC API requires an output buffer for the log.
        size_result = nvrtc.nvrtcGetProgramLogSize(program)
        size_err = (
            size_result[0]
            if isinstance(size_result, tuple)
            else size_result
        )

        if size_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(
                f"NVRTC compilation failed: {err}; "
                f"also failed to get compiler log: {size_err}"
            )

        log_size = size_result[1]
        log_buffer = bytearray(log_size)

        log_result = nvrtc.nvrtcGetProgramLog(
            program,
            log_buffer,
        )
        log_err = (
            log_result[0]
            if isinstance(log_result, tuple)
            else log_result
        )

        if log_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log_text = f"<unable to retrieve NVRTC log: {log_err}>"
        else:
            log_text = bytes(log_buffer).rstrip(b"\\x00").decode(
                "utf-8",
                errors="replace",
            )

        raise RuntimeError(
            f"NVRTC compilation failed: {err}\\n"
            f"Compiler log:\\n{log_text}"
        )

    # cuda.bindings NVRTC API requires an output buffer for PTX as well.
    ptx_size_result = nvrtc.nvrtcGetPTXSize(program)
    ptx_size_err = (
        ptx_size_result[0]
        if isinstance(ptx_size_result, tuple)
        else ptx_size_result
    )

    if ptx_size_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(
            f"nvrtcGetPTXSize failed: {ptx_size_result}"
        )

    ptx_size = ptx_size_result[1]
    ptx_buffer = bytearray(ptx_size)

    ptx_result = nvrtc.nvrtcGetPTX(
        program,
        ptx_buffer,
    )
    ptx_err = (
        ptx_result[0]
        if isinstance(ptx_result, tuple)
        else ptx_result
    )

    if ptx_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(
            f"nvrtcGetPTX failed: {ptx_result}"
        )

    ptx = bytes(ptx_buffer)
    module = check(cuda.cuModuleLoadData(ptx), "cuModuleLoadData")[1]

    functions = {}
    for name in [
        "euler_f32_f32",
        "euler_f32_f16",
        "euler_f16_f32",
        "euler_f16_f16",
        "add_t_f32",
        "add_t_f16",
    ]:
        functions[name] = check(
            cuda.cuModuleGetFunction(module, name.encode()),
            f"cuModuleGetFunction({name})",
        )[1]

    print("[OK] CUDA helper kernels ready")
    return module, functions


def launch_kernel(func, stream, args, threads=32):
    check(
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


def make_euler_args(action_ptr, velocity_ptr, dt):
    # cuda.bindings' ctypes launch interface expects:
    #   kernelParams = (kernel_values, kernel_types)
    # rather than a pre-built tuple of void* addresses.
    kernel_values = (
        int(action_ptr),
        int(velocity_ptr),
        float(dt),
    )
    kernel_types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
    )
    return kernel_values, kernel_types


def make_t_args(t_ptr, dt):
    kernel_values = (
        int(t_ptr),
        float(dt),
    )
    kernel_types = (
        ctypes.c_void_p,
        ctypes.c_float,
    )
    return kernel_values, kernel_types


def load_inputs():
    if not REFERENCE_NPZ.exists():
        raise FileNotFoundError(f"Missing reference NPZ:\n{REFERENCE_NPZ}")

    data = np.load(REFERENCE_NPZ, allow_pickle=False)

    if "visual_tokens" not in data:
        raise RuntimeError("Reference NPZ has no visual_tokens")

    visual_tokens = np.ascontiguousarray(
        data["visual_tokens"].astype(np.int64)
    )

    if visual_tokens.shape != (1, 8, 18, 32):
        raise RuntimeError(
            f"Unexpected visual_tokens shape: {visual_tokens.shape}"
        )

    command = np.zeros((1, 1), dtype=np.int64)

    # Latency benchmark only:
    # deterministic input is sufficient. Exact PC random initialization
    # will be used later for numerical equivalence validation.
    rng = np.random.default_rng(0)
    initial_action = np.ascontiguousarray(
        rng.standard_normal(ACTION_SHAPE).astype(np.float32)
    )

    print()
    print("=" * 80)
    print("Input")
    print("=" * 80)
    print("visual_tokens :", visual_tokens.shape, visual_tokens.dtype)
    print("command       :", command.shape, command.dtype, command)
    print("initial_action:", initial_action.shape, initial_action.dtype)
    print(
        "initial range :",
        float(initial_action.min()),
        float(initial_action.max()),
    )

    return visual_tokens, command, initial_action


def main():
    print("=" * 80)
    print("VaVAM-B 12i v3 — Thor GPU-Resident E2E Benchmark")
    print("=" * 80)
    print("TensorRT:", trt.__version__)
    print("Steps   :", NUM_STEPS)
    print("dt      :", DT)
    print("Warmup  :", WARMUP)
    print("Runs    :", BENCHMARK)

    check(cuda.cuInit(0), "cuInit")
    device = check(cuda.cuDeviceGet(0), "cuDeviceGet")[1]
    ctx = check(
        cuda.cuDevicePrimaryCtxRetain(device),
        "cuDevicePrimaryCtxRetain",
    )[1]
    check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    stream = stream_create()

    visual_tokens, command, initial_action = load_inputs()

    prefill = Engine(PREFILL_ENGINE)

    kv_names = (
        [f"visual_k_{i}" for i in range(NUM_LAYERS)]
        + [f"visual_v_{i}" for i in range(NUM_LAYERS)]
    )

    action = Engine(
        ACTION_ENGINE,
        skip_names=kv_names,
    )

    for name in ["noisy_actions", "high_level_command", "diffusion_step"]:
        if action.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Missing Action input: {name}")

    if action.engine.get_tensor_mode("actions") != trt.TensorIOMode.OUTPUT:
        raise RuntimeError("Expected Action output named 'actions'")

    # Direct GPU address sharing: Prefill K/V -> Action K/V.
    for name in kv_names:
        if name not in prefill.owned:
            raise RuntimeError(f"Prefill output missing: {name}")
        action.set_address(name, prefill.owned[name])

    print()
    print("[OK] 48 visual K/V tensors are GPU-resident and shared")

    _, funcs = compile_cuda_module()

    action_dtype = action.dtype("noisy_actions")
    velocity_dtype = action.dtype("actions")
    t_dtype = action.dtype("diffusion_step")

    print()
    print("=" * 80)
    print("Action IO dtypes")
    print("=" * 80)
    print("noisy_actions :", action_dtype)
    print("actions       :", velocity_dtype)
    print("diffusion_step:", t_dtype)

    if action.shape("noisy_actions") != ACTION_SHAPE:
        raise RuntimeError(
            f"Unexpected noisy_actions shape: {action.shape('noisy_actions')}"
        )

    visual_ptr = prefill.owned["visual_tokens"]
    action_ptr = action.owned["noisy_actions"]
    command_ptr = action.owned["high_level_command"]
    t_ptr = action.owned["diffusion_step"]
    velocity_ptr = action.owned["actions"]

    action_host = initial_action.astype(action_dtype, copy=False)
    command_host = command.astype(
        action.dtype("high_level_command"),
        copy=False,
    )
    t_host = np.zeros(
        action.shape("diffusion_step"),
        dtype=t_dtype,
    )

    # Initial state upload. These are outside the measured GPU region.
    h2d(visual_ptr, visual_tokens)
    h2d(command_ptr, command_host)
    h2d(action_ptr, action_host)
    h2d(t_ptr, t_host)
    stream_sync(stream)

    e_total_start = event_create()
    e_total_end = event_create()
    e_prefill_start = event_create()
    e_prefill_end = event_create()
    e_action_start = event_create()
    e_action_end = event_create()

    action_is_f16 = action_dtype == np.dtype(np.float16)
    velocity_is_f16 = velocity_dtype == np.dtype(np.float16)

    if action_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise RuntimeError(f"Unsupported action dtype: {action_dtype}")

    if velocity_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise RuntimeError(f"Unsupported velocity dtype: {velocity_dtype}")

    euler_name = (
        "euler_f16_f16" if action_is_f16 and velocity_is_f16 else
        "euler_f16_f32" if action_is_f16 and not velocity_is_f16 else
        "euler_f32_f16" if not action_is_f16 and velocity_is_f16 else
        "euler_f32_f32"
    )
    euler_func = funcs[euler_name]

    if t_dtype == np.dtype(np.float16):
        t_func = funcs["add_t_f16"]
    elif t_dtype == np.dtype(np.float32):
        t_func = funcs["add_t_f32"]
    else:
        raise RuntimeError(f"Unsupported diffusion_step dtype: {t_dtype}")

    euler_args = make_euler_args(action_ptr, velocity_ptr, DT)
    t_args = make_t_args(t_ptr, DT)

    def reset_state():
        h2d(action_ptr, action_host)
        h2d(t_ptr, t_host)
        stream_sync(stream)

    def run_once():
        reset_state()

        event_record(e_total_start, stream)

        event_record(e_prefill_start, stream)
        prefill.execute(stream)
        event_record(e_prefill_end, stream)

        event_record(e_action_start, stream)

        for _ in range(NUM_STEPS):
            action.execute(stream)

            launch_kernel(
                euler_func,
                stream,
                euler_args,
                threads=32,
            )

            launch_kernel(
                t_func,
                stream,
                t_args,
                threads=1,
            )

        event_record(e_action_end, stream)
        event_record(e_total_end, stream)

        stream_sync(stream)

        return (
            elapsed_ms(e_prefill_start, e_prefill_end),
            elapsed_ms(e_action_start, e_action_end),
            elapsed_ms(e_total_start, e_total_end),
        )

    print()
    print("=" * 80)
    print(f"Warmup: {WARMUP}")
    print("=" * 80)

    for _ in range(WARMUP):
        run_once()

    print()
    print("=" * 80)
    print(f"Benchmark: {BENCHMARK} complete inferences")
    print("=" * 80)

    ps, ass, ts = [], [], []

    for i in range(BENCHMARK):
        p, a, total = run_once()
        ps.append(p)
        ass.append(a)
        ts.append(total)

        print(
            f"{i+1:3d}: "
            f"Prefill={p:8.3f} ms  "
            f"Action×10+Euler={a:8.3f} ms  "
            f"E2E={total:8.3f} ms"
        )

    ps = np.asarray(ps, dtype=np.float64)
    ass = np.asarray(ass, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    def report(name, x):
        print()
        print(name)
        print(f"  mean   : {np.mean(x):.4f} ms")
        print(f"  median : {np.median(x):.4f} ms")
        print(f"  p95    : {np.percentile(x, 95):.4f} ms")
        print(f"  p99    : {np.percentile(x, 99):.4f} ms")
        print(f"  min    : {np.min(x):.4f} ms")
        print(f"  max    : {np.max(x):.4f} ms")

    print()
    print("=" * 80)
    print("12i RESULT")
    print("=" * 80)

    report("Prefill FP16", ps)
    report("Action FP16 ×10 + Euler", ass)
    report("Complete E2E", ts)

    mean_total = float(np.mean(ts))

    print()
    print(f"E2E mean       : {mean_total:.4f} ms")
    print(f"E2E throughput : {1000.0 / mean_total:.4f} Hz")

    print()
    print("GPU residency:")
    print("  visual K/V   : YES")
    print("  action       : YES")
    print("  diffusion t  : YES")
    print("  Euler        : CUDA kernel")
    print("  per-step H2D : NONE")
    print("  per-step D2H : NONE")
    print("  final D2H    : outside GPU timing")

    # Final trajectory copy is deliberately outside timing.
    final_host = np.empty(
        action.shape("noisy_actions"),
        dtype=action_dtype,
    )
    d2h(final_host, action_ptr)

    print()
    print("Final action:")
    print("  shape:", final_host.shape)
    print("  dtype:", final_host.dtype)
    print("  min  :", float(final_host.min()))
    print("  max  :", float(final_host.max()))
    print("  mean :", float(final_host.mean()))
    print("  std  :", float(final_host.std()))

    action.cleanup()
    prefill.cleanup()

    print()
    print("[OK] Benchmark completed.")


if __name__ == "__main__":
    main()
