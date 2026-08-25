#!/usr/bin/env python3

import argparse
import ctypes
import ctypes.util
import os
import time

import numpy as np
import tensorrt as trt


# ============================================================
# CUDA Runtime wrapper
# Avoid dependency on cuda-python
# ============================================================

class CudaRuntime:
    CUDA_SUCCESS = 0

    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self):
        lib_path = ctypes.util.find_library("cudart")

        candidates = []

        if lib_path:
            candidates.append(lib_path)

        candidates.extend([
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda-12.8/lib64/libcudart.so",
            "/usr/lib/aarch64-linux-gnu/libcudart.so",
            "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
        ])

        self.lib = None

        for path in candidates:
            try:
                self.lib = ctypes.CDLL(path)
                print(f"[INFO] Loaded CUDA Runtime: {path}")
                break
            except OSError:
                continue

        if self.lib is None:
            raise RuntimeError(
                "Could not load libcudart.so. "
                "Please check CUDA installation."
            )

        self._setup_signatures()

    def _setup_signatures(self):

        # cudaMalloc(void **devPtr, size_t size)
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int

        # cudaFree(void *devPtr)
        self.lib.cudaFree.argtypes = [
            ctypes.c_void_p,
        ]
        self.lib.cudaFree.restype = ctypes.c_int

        # cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int

        # cudaStreamCreate(cudaStream_t *pStream)
        self.lib.cudaStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cudaStreamCreate.restype = ctypes.c_int

        # cudaStreamDestroy(cudaStream_t stream)
        self.lib.cudaStreamDestroy.argtypes = [
            ctypes.c_void_p,
        ]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

        # cudaStreamSynchronize(cudaStream_t stream)
        self.lib.cudaStreamSynchronize.argtypes = [
            ctypes.c_void_p,
        ]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

        # cudaMemcpyAsync(...)
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int

        # cudaGetErrorString(cudaError_t)
        self.lib.cudaGetErrorString.argtypes = [
            ctypes.c_int,
        ]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, status, what):

        if status != self.CUDA_SUCCESS:

            try:
                msg = self.lib.cudaGetErrorString(status)
                if msg:
                    msg = msg.decode()
                else:
                    msg = f"CUDA error code {status}"
            except Exception:
                msg = f"CUDA error code {status}"

            raise RuntimeError(
                f"{what} failed: {msg}"
            )

    def malloc(self, nbytes):

        ptr = ctypes.c_void_p()

        status = self.lib.cudaMalloc(
            ctypes.byref(ptr),
            nbytes,
        )

        self.check(status, "cudaMalloc")

        return ptr

    def free(self, ptr):

        if ptr is not None:
            status = self.lib.cudaFree(ptr)
            self.check(status, "cudaFree")

    def memcpy_htod(self, dst, src):

        src_ptr = ctypes.c_void_p(src.ctypes.data)

        status = self.lib.cudaMemcpy(
            dst,
            src_ptr,
            src.nbytes,
            self.CUDA_MEMCPY_HOST_TO_DEVICE,
        )

        self.check(status, "cudaMemcpy H2D")

    def memcpy_dtoh(self, dst, src):

        dst_ptr = ctypes.c_void_p(dst.ctypes.data)

        status = self.lib.cudaMemcpy(
            dst_ptr,
            src,
            dst.nbytes,
            self.CUDA_MEMCPY_DEVICE_TO_HOST,
        )

        self.check(status, "cudaMemcpy D2H")

    def stream_create(self):

        stream = ctypes.c_void_p()

        status = self.lib.cudaStreamCreate(
            ctypes.byref(stream)
        )

        self.check(status, "cudaStreamCreate")

        return stream

    def stream_destroy(self, stream):

        status = self.lib.cudaStreamDestroy(stream)
        self.check(status, "cudaStreamDestroy")

    def stream_synchronize(self, stream):

        status = self.lib.cudaStreamSynchronize(stream)
        self.check(status, "cudaStreamSynchronize")


# ============================================================
# TensorRT helpers
# ============================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):

    print(f"\n[INFO] Loading engine:")
    print(f"  {engine_path}")

    if not os.path.isfile(engine_path):
        raise FileNotFoundError(engine_path)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)

    print(f"  Engine size: {size_mb:.2f} MB")

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)

    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize TensorRT engine: {engine_path}"
        )

    return runtime, engine


def print_engine_io(engine):

    print("\n[TensorRT Engine IO]")

    for i in range(engine.num_io_tensors):

        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)

        if mode == trt.TensorIOMode.INPUT:
            io_type = "INPUT "
        else:
            io_type = "OUTPUT"

        print(
            f"  {io_type} {name:<25}"
            f" shape={tuple(shape)}"
            f" dtype={dtype}"
        )


# ============================================================
# TensorRT inference
# ============================================================

class TensorRTEngine:

    def __init__(self, engine_path, cuda):

        self.cuda = cuda
        self.runtime, self.engine = load_engine(engine_path)

        print_engine_io(self.engine)

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError(
                "Failed to create TensorRT execution context."
            )

        self.stream = cuda.stream_create()

        self.device_buffers = {}
        self.host_outputs = {}
        self.host_inputs = {}

        self._allocate_buffers()

    def _allocate_buffers(self):

        print("\n[TensorRT Buffers]")

        for i in range(self.engine.num_io_tensors):

            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))

            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Dynamic shape is not supported by this script: "
                    f"{name} {shape}"
                )

            np_dtype = trt.nptype(dtype)
            nbytes = int(
                np.prod(shape) * np.dtype(np_dtype).itemsize
            )

            device_ptr = self.cuda.malloc(nbytes)

            self.device_buffers[name] = device_ptr

            self.context.set_tensor_address(
                name,
                device_ptr.value,
            )

            mode = self.engine.get_tensor_mode(name)

            print(
                f"  {name:<25}"
                f" shape={shape}"
                f" dtype={np_dtype}"
                f" bytes={nbytes}"
            )

            if mode == trt.TensorIOMode.OUTPUT:

                self.host_outputs[name] = np.empty(
                    shape,
                    dtype=np_dtype,
                )

            else:

                self.host_inputs[name] = np.empty(
                    shape,
                    dtype=np_dtype,
                )

    def validate_inputs(self, reference_inputs):

        print("\n[Input validation]")

        for name, ref in reference_inputs.items():

            if name not in self.device_buffers:
                raise RuntimeError(
                    f"Reference input '{name}' "
                    f"not found in engine."
                )

            engine_shape = tuple(
                self.engine.get_tensor_shape(name)
            )

            engine_dtype = np.dtype(
                trt.nptype(
                    self.engine.get_tensor_dtype(name)
                )
            )

            print(
                f"  {name:<25}"
                f"reference={ref.shape}/{ref.dtype}"
                f" engine={engine_shape}/{engine_dtype}"
            )

            if tuple(ref.shape) != engine_shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"{ref.shape} vs {engine_shape}"
                )

            if np.dtype(ref.dtype) != engine_dtype:
                raise RuntimeError(
                    f"Dtype mismatch for {name}: "
                    f"{ref.dtype} vs {engine_dtype}"
                )

    def inference(self, inputs):

        # H2D
        for name, array in inputs.items():

            self.cuda.memcpy_htod(
                self.device_buffers[name],
                np.ascontiguousarray(array),
            )

        # TensorRT execution
        ok = self.context.execute_async_v3(
            stream_handle=self.stream.value
        )

        if not ok:
            raise RuntimeError(
                "TensorRT execute_async_v3() failed."
            )

        # Synchronize
        self.cuda.stream_synchronize(self.stream)

        # D2H
        outputs = {}

        for name, host_array in self.host_outputs.items():

            self.cuda.memcpy_dtoh(
                host_array,
                self.device_buffers[name],
            )

            outputs[name] = host_array.copy()

        return outputs

    def warmup(self, inputs, count=10):

        print("\n[INFO] Warmup...")

        for _ in range(count):
            self.inference(inputs)

        print("[OK] Warmup completed")

    def benchmark(self, inputs, runs=20):

        print("\n[INFO] TensorRT inference...")

        latencies = []

        for i in range(runs):

            start = time.perf_counter()

            self.inference(inputs)

            end = time.perf_counter()

            latency_ms = (end - start) * 1000.0

            latencies.append(latency_ms)

            print(
                f"  run {i:02d}: "
                f"{latency_ms:.4f} ms"
            )

        latencies = np.asarray(latencies)

        print("\n[Latency]")

        print(
            f"  mean : {latencies.mean():.4f} ms"
        )

        print(
            f"  min  : {latencies.min():.4f} ms"
        )

        print(
            f"  max  : {latencies.max():.4f} ms"
        )

        return latencies

    def close(self):

        for ptr in self.device_buffers.values():

            self.cuda.free(ptr)

        self.device_buffers.clear()

        if self.stream is not None:

            self.cuda.stream_destroy(
                self.stream
            )

            self.stream = None


# ============================================================
# Statistics
# ============================================================

def print_stats(label, output):

    print(f"\n[{label}]")

    print(f"  shape : {output.shape}")
    print(f"  dtype : {output.dtype}")
    print(f"  min   : {output.min()}")
    print(f"  max   : {output.max()}")
    print(f"  mean  : {output.mean()}")
    print(f"  std   : {output.std()}")


def compare_outputs(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    diff = np.abs(a - b)

    denom = np.maximum(
        np.abs(a),
        1e-8,
    )

    rel = diff / denom

    rmse = np.sqrt(
        np.mean(
            (a - b) ** 2
        )
    )

    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(rmse),
        "max_rel": float(rel.max()),
        "mean_rel": float(rel.mean()),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare TensorRT FP32 and FP16 "
            "Action Expert engines on NVIDIA DRIVE Thor."
        )
    )

    parser.add_argument(
        "--fp32-engine",
        default="./action_expert_fp32.engine",
        help="Path to TensorRT FP32 engine",
    )

    parser.add_argument(
        "--fp16-engine",
        default="./action_expert_fp16.engine",
        help="Path to TensorRT FP16 engine",
    )

    parser.add_argument(
        "--reference",
        default="./onnx_reference.npz",
        help="Path to PC reference NPZ",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "VaVAM Action Expert - "
        "Thor TensorRT FP32 vs FP16"
    )
    print("=" * 72)

    print("\n[INFO] TensorRT:")
    print(f"  {trt.__version__}")

    # --------------------------------------------------------
    # Load reference
    # --------------------------------------------------------

    print("\n[INFO] Loading reference:")

    print(f"  {args.reference}")

    if not os.path.isfile(args.reference):
        raise FileNotFoundError(args.reference)

    reference = np.load(args.reference)

    print("\n[Reference keys]")

    for key in reference.files:

        print(
            f"  {key:<22}"
            f"shape={reference[key].shape}"
            f" dtype={reference[key].dtype}"
        )

    required_inputs = [
        "noisy_actions",
        "high_level_command",
        "diffusion_step",
    ]

    for key in required_inputs:

        if key not in reference:
            raise RuntimeError(
                f"Missing reference input: {key}"
            )

    inputs = {
        "noisy_actions":
            np.ascontiguousarray(
                reference["noisy_actions"]
            ),

        "high_level_command":
            np.ascontiguousarray(
                reference["high_level_command"]
            ),

        "diffusion_step":
            np.ascontiguousarray(
                reference["diffusion_step"]
            ),
    }

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    print("\n[INFO] Initializing CUDA Runtime...")

    cuda = CudaRuntime()

    print("[OK] CUDA Runtime initialized")

    # --------------------------------------------------------
    # FP32
    # --------------------------------------------------------

    fp32 = TensorRTEngine(
        args.fp32_engine,
        cuda,
    )

    fp32.validate_inputs(inputs)

    fp32.warmup(
        inputs,
        count=args.warmup,
    )

    fp32_latencies = fp32.benchmark(
        inputs,
        runs=args.runs,
    )

    fp32_output_dict = fp32.inference(inputs)

    fp32_output = fp32_output_dict[
        "action_output"
    ]

    print_stats(
        "TensorRT FP32 output",
        fp32_output,
    )

    # Determinism
    fp32_output_2 = fp32.inference(inputs)[
        "action_output"
    ]

    fp32_det = compare_outputs(
        fp32_output,
        fp32_output_2,
    )

    print("\n[FP32 determinism]")

    print(
        f"  max abs diff : "
        f"{fp32_det['max_abs']:.8e}"
    )

    print(
        f"  mean abs diff: "
        f"{fp32_det['mean_abs']:.8e}"
    )

    fp32.close()

    # --------------------------------------------------------
    # FP16
    # --------------------------------------------------------

    fp16 = TensorRTEngine(
        args.fp16_engine,
        cuda,
    )

    fp16.validate_inputs(inputs)

    fp16.warmup(
        inputs,
        count=args.warmup,
    )

    fp16_latencies = fp16.benchmark(
        inputs,
        runs=args.runs,
    )

    fp16_output_dict = fp16.inference(inputs)

    fp16_output = fp16_output_dict[
        "action_output"
    ]

    print_stats(
        "TensorRT FP16 output",
        fp16_output,
    )

    # Determinism
    fp16_output_2 = fp16.inference(inputs)[
        "action_output"
    ]

    fp16_det = compare_outputs(
        fp16_output,
        fp16_output_2,
    )

    print("\n[FP16 determinism]")

    print(
        f"  max abs diff : "
        f"{fp16_det['max_abs']:.8e}"
    )

    print(
        f"  mean abs diff: "
        f"{fp16_det['mean_abs']:.8e}"
    )

    fp16.close()

    # --------------------------------------------------------
    # Numerical comparison
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("Numerical Comparison")
    print("=" * 72)

    fp32_vs_fp16 = compare_outputs(
        fp32_output,
        fp16_output,
    )

    print("\n[Thor TensorRT FP32 vs FP16]")

    print(
        f"  max abs error : "
        f"{fp32_vs_fp16['max_abs']:.8e}"
    )

    print(
        f"  mean abs error: "
        f"{fp32_vs_fp16['mean_abs']:.8e}"
    )

    print(
        f"  RMSE          : "
        f"{fp32_vs_fp16['rmse']:.8e}"
    )

    print(
        f"  max rel error : "
        f"{fp32_vs_fp16['max_rel']:.8e}"
    )

    print(
        f"  mean rel error: "
        f"{fp32_vs_fp16['mean_rel']:.8e}"
    )

    # --------------------------------------------------------
    # Compare against PC PyTorch
    # --------------------------------------------------------

    if "torch_output" in reference:

        torch_output = reference[
            "torch_output"
        ]

        print("\n[Thor FP32 vs PC PyTorch]")

        m = compare_outputs(
            torch_output,
            fp32_output,
        )

        print(
            f"  max abs error : "
            f"{m['max_abs']:.8e}"
        )

        print(
            f"  mean abs error: "
            f"{m['mean_abs']:.8e}"
        )

        print(
            f"  RMSE          : "
            f"{m['rmse']:.8e}"
        )

        print(
            f"  max rel error : "
            f"{m['max_rel']:.8e}"
        )

        print(
            f"  mean rel error: "
            f"{m['mean_rel']:.8e}"
        )

        print("\n[Thor FP16 vs PC PyTorch]")

        m = compare_outputs(
            torch_output,
            fp16_output,
        )

        print(
            f"  max abs error : "
            f"{m['max_abs']:.8e}"
        )

        print(
            f"  mean abs error: "
            f"{m['mean_abs']:.8e}"
        )

        print(
            f"  RMSE          : "
            f"{m['rmse']:.8e}"
        )

        print(
            f"  max rel error : "
            f"{m['max_rel']:.8e}"
        )

        print(
            f"  mean rel error: "
            f"{m['mean_rel']:.8e}"
        )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    fp32_mean = fp32_latencies.mean()
    fp16_mean = fp16_latencies.mean()

    speedup = fp32_mean / fp16_mean

    print("\n" + "=" * 72)
    print("Performance")
    print("=" * 72)

    print(
        f"\nFP32 latency : {fp32_mean:.4f} ms"
    )

    print(
        f"FP16 latency : {fp16_mean:.4f} ms"
    )

    print(
        f"FP16 speedup : {speedup:.3f}x"
    )

    fp32_size = (
        os.path.getsize(args.fp32_engine)
        / (1024 * 1024)
    )

    fp16_size = (
        os.path.getsize(args.fp16_engine)
        / (1024 * 1024)
    )

    print("\n[Engine Size]")

    print(
        f"  FP32 : {fp32_size:.2f} MB"
    )

    print(
        f"  FP16 : {fp16_size:.2f} MB"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("[RESULT] FP32 vs FP16 comparison complete")
    print("=" * 72)


if __name__ == "__main__":
    main()