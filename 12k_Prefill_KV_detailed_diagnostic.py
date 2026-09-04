#!/usr/bin/env python3
"""
12k VaVAM-B — Prefill K/V detailed diagnostic (PC Official vs Thor TRT)

Purpose
-------
Diagnose the already-observed PC <-> Thor 24-layer K/V differences in more
detail. This script runs ONLY the Thor TensorRT PREFILL engine once, captures
all 24 K/V outputs, and compares them against the PC Official Forward K/V
reference files.

It reports for every layer and for K/V:
  - max / mean absolute error
  - RMSE
  - PC and Thor min/max/mean/std
  - relative L2 error
  - fraction of Thor values exactly representable as FP16
  - worst absolute-error element:
      flattened index + multi-dimensional index
      PC value, Thor value, difference

It also reports global worst K/V elements and saves a JSON report.

This is a numerical diagnostic, NOT a latency benchmark.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cuda


ROOT = Path.home() / "vblkdev2" / "VaVAM_Thor"
DEFAULT_ENGINE = ROOT / "Engines" / "vavam_joint_kv_prefill_B_v10_fp16.engine"
DEFAULT_REFERENCE = ROOT / "pc_reference"
NUM_LAYERS = 24


def check(result, name):
    err = result[0] if isinstance(result, tuple) else result
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {err}")
    return result


def stats_detail(pc, thor):
    pc = np.asarray(pc, dtype=np.float32)
    thor = np.asarray(thor, dtype=np.float32)

    if pc.shape != thor.shape:
        raise RuntimeError(
            f"Shape mismatch: PC={pc.shape}, Thor={thor.shape}"
        )

    diff = thor - pc
    absdiff = np.abs(diff)

    flat = absdiff.reshape(-1)
    wi = int(np.argmax(flat))
    idx = np.unravel_index(wi, absdiff.shape)

    # Relative L2 error is more useful than pointwise relative error because
    # individual K/V values can be close to zero.
    pc_l2 = float(np.linalg.norm(pc.reshape(-1)))
    diff_l2 = float(np.linalg.norm(diff.reshape(-1)))

    # Check whether values are exactly representable by IEEE FP16.
    # This is a diagnostic only; it does not imply how the TensorRT internal
    # computation was performed.
    thor_fp16_roundtrip = thor.astype(np.float16).astype(np.float32)
    pc_fp16_roundtrip = pc.astype(np.float16).astype(np.float32)

    thor_fp16_exact_fraction = float(
        np.mean(thor_fp16_roundtrip == thor)
    )
    pc_fp16_exact_fraction = float(
        np.mean(pc_fp16_roundtrip == pc)
    )

    return {
        "shape": list(pc.shape),
        "num_elements": int(pc.size),

        "max_abs": float(absdiff.max()),
        "mean_abs": float(absdiff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),

        "pc_min": float(pc.min()),
        "pc_max": float(pc.max()),
        "pc_mean": float(pc.mean()),
        "pc_std": float(pc.std()),

        "thor_min": float(thor.min()),
        "thor_max": float(thor.max()),
        "thor_mean": float(thor.mean()),
        "thor_std": float(thor.std()),

        "relative_l2_error": (
            diff_l2 / pc_l2 if pc_l2 > 0 else None
        ),
        "pc_l2": pc_l2,
        "diff_l2": diff_l2,

        "pc_fp16_exact_fraction": pc_fp16_exact_fraction,
        "thor_fp16_exact_fraction": thor_fp16_exact_fraction,

        "worst_index": list(idx),
        "worst_flat_index": wi,
        "worst_pc_value": float(pc[idx]),
        "worst_thor_value": float(thor[idx]),
        "worst_difference_thor_minus_pc": float(diff[idx]),
        "worst_abs_difference": float(absdiff[idx]),
    }


def top_k_errors(pc, thor, k=10):
    pc = np.asarray(pc, dtype=np.float32)
    thor = np.asarray(thor, dtype=np.float32)

    diff = thor - pc
    absdiff = np.abs(diff)
    flat = absdiff.reshape(-1)

    k = min(k, flat.size)
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(flat[indices])[::-1]]

    result = []
    for wi in indices:
        idx = np.unravel_index(int(wi), absdiff.shape)
        result.append({
            "index": list(idx),
            "flat_index": int(wi),
            "pc": float(pc[idx]),
            "thor": float(thor[idx]),
            "difference_thor_minus_pc": float(diff[idx]),
            "abs_difference": float(absdiff[idx]),
        })
    return result


class Engine:
    def __init__(self, path):
        self.path = Path(path)
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)

        with open(self.path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {self.path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.ptrs = {}

        print("=" * 80)
        print(f"Loaded engine: {self.path}")
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
                raise RuntimeError(
                    f"Dynamic shape unsupported: {name} {shape}"
                )

            np_dtype = np.dtype(trt.nptype(dtype))
            nbytes = int(np.prod(shape)) * np_dtype.itemsize

            ptr = check(
                cuda.cuMemAlloc(nbytes),
                f"cuMemAlloc({name})",
            )[1]

            self.ptrs[name] = ptr

            if not self.context.set_tensor_address(name, int(ptr)):
                raise RuntimeError(
                    f"set_tensor_address failed: {name}"
                )

    def input_names(self):
        return [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(
                self.engine.get_tensor_name(i)
            ) == trt.TensorIOMode.INPUT
        ]

    def output_names(self):
        return [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(
                self.engine.get_tensor_name(i)
            ) == trt.TensorIOMode.OUTPUT
        ]

    def dtype(self, name):
        return np.dtype(
            trt.nptype(self.engine.get_tensor_dtype(name))
        )

    def shape(self, name):
        return tuple(self.engine.get_tensor_shape(name))

    def execute(self, stream):
        if not self.context.execute_async_v3(
            stream_handle=int(stream)
        ):
            raise RuntimeError(
                "TensorRT execute_async_v3 failed"
            )

    def cleanup(self):
        for ptr in self.ptrs.values():
            check(
                cuda.cuMemFree(int(ptr)),
                "cuMemFree",
            )
        self.ptrs.clear()


def h2d(ptr, arr):
    arr = np.ascontiguousarray(arr)
    check(
        cuda.cuMemcpyHtoD(
            int(ptr),
            arr,
            int(arr.nbytes),
        ),
        "cuMemcpyHtoD",
    )


def d2h(ptr, arr):
    arr = np.ascontiguousarray(arr)
    check(
        cuda.cuMemcpyDtoH(
            arr,
            int(ptr),
            int(arr.nbytes),
        ),
        "cuMemcpyDtoH",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        default=str(DEFAULT_ENGINE),
    )
    parser.add_argument(
        "--reference-dir",
        default=str(DEFAULT_REFERENCE),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    ref = Path(args.reference_dir).expanduser().resolve()

    visual_path = ref / "visual_tokens.npy"
    if not visual_path.exists():
        raise FileNotFoundError(visual_path)

    for layer in range(NUM_LAYERS):
        for kind in ("k", "v"):
            p = ref / f"pc_official_kv_{kind}_{layer:02d}.npy"
            if not p.exists():
                raise FileNotFoundError(p)

    visual = np.load(visual_path)

    if visual.shape == (8, 18, 32):
        visual = visual[None, ...]

    if visual.shape != (1, 8, 18, 32):
        raise RuntimeError(
            f"Unexpected visual_tokens shape: {visual.shape}"
        )

    print()
    print("=" * 80)
    print("12k Prefill K/V Detailed Diagnostic")
    print("=" * 80)
    print(f"Reference dir : {ref}")
    print(f"Visual shape  : {visual.shape}")
    print(f"Top-K errors  : {args.top_k}")
    print()
    print("This run executes ONLY TensorRT Prefill once.")
    print("No Action Expert and no Euler are executed.")

    check(cuda.cuInit(0), "cuInit")
    device = check(
        cuda.cuDeviceGet(0),
        "cuDeviceGet",
    )[1]
    ctx = check(
        cuda.cuDevicePrimaryCtxRetain(device),
        "cuDevicePrimaryCtxRetain",
    )[1]
    check(
        cuda.cuCtxSetCurrent(ctx),
        "cuCtxSetCurrent",
    )
    stream = check(
        cuda.cuStreamCreate(0),
        "cuStreamCreate",
    )[1]

    eng = Engine(args.engine)

    inputs = eng.input_names()
    outputs = eng.output_names()

    if len(inputs) != 1:
        raise RuntimeError(
            f"Expected exactly one input, found: {inputs}"
        )

    visual_name = inputs[0]
    print(f"\nUsing input: {visual_name}")

    visual_host = visual.astype(
        eng.dtype(visual_name),
        copy=False,
    )

    h2d(eng.ptrs[visual_name], visual_host)

    check(
        cuda.cuStreamSynchronize(stream),
        "sync before prefill",
    )

    eng.execute(stream)

    check(
        cuda.cuStreamSynchronize(stream),
        "sync after prefill",
    )

    k_names = sorted(
        [n for n in outputs if "visual_k" in n],
        key=lambda x: int(x.rsplit("_", 1)[-1]),
    )

    v_names = sorted(
        [n for n in outputs if "visual_v" in n],
        key=lambda x: int(x.rsplit("_", 1)[-1]),
    )

    if len(k_names) != NUM_LAYERS or len(v_names) != NUM_LAYERS:
        raise RuntimeError(
            f"Expected 24 K + 24 V outputs, got "
            f"K={len(k_names)} V={len(v_names)}"
        )

    per_layer = []
    all_k = []
    all_v = []

    print()
    print("=" * 180)
    print("Per-layer detailed PC Official ↔ Thor Prefill K/V")
    print("=" * 180)
    print(
        "layer | "
        "K max      K mean     K RMSE     K relL2   "
        "| V max      V mean     V RMSE     V relL2"
    )
    print("-" * 180)

    for layer in range(NUM_LAYERS):
        kname = k_names[layer]
        vname = v_names[layer]

        k = np.empty(
            eng.shape(kname),
            dtype=eng.dtype(kname),
        )
        v = np.empty(
            eng.shape(vname),
            dtype=eng.dtype(vname),
        )

        d2h(eng.ptrs[kname], k)
        d2h(eng.ptrs[vname], v)

        thor_k = k.astype(np.float32)
        thor_v = v.astype(np.float32)

        pc_k = np.load(
            ref / f"pc_official_kv_k_{layer:02d}.npy"
        ).astype(np.float32)

        pc_v = np.load(
            ref / f"pc_official_kv_v_{layer:02d}.npy"
        ).astype(np.float32)

        ks = stats_detail(pc_k, thor_k)
        vs = stats_detail(pc_v, thor_v)

        ks["top_errors"] = top_k_errors(
            pc_k, thor_k, args.top_k
        )
        vs["top_errors"] = top_k_errors(
            pc_v, thor_v, args.top_k
        )

        per_layer.append({
            "layer": layer,
            "k": ks,
            "v": vs,
        })

        all_k.append((layer, pc_k, thor_k))
        all_v.append((layer, pc_v, thor_v))

        print(
            f"{layer:5d} | "
            f"{ks['max_abs']:9.3e} "
            f"{ks['mean_abs']:9.3e} "
            f"{ks['rmse']:9.3e} "
            f"{ks['relative_l2_error']:9.3e} "
            f"| "
            f"{vs['max_abs']:9.3e} "
            f"{vs['mean_abs']:9.3e} "
            f"{vs['rmse']:9.3e} "
            f"{vs['relative_l2_error']:9.3e}"
        )

    # ---------------------------------------------------------------
    # Global worst elements.
    # ---------------------------------------------------------------
    worst_k = None
    worst_v = None

    for layer, pc, thor in all_k:
        s = stats_detail(pc, thor)
        candidate = {
            "layer": layer,
            **s,
        }
        if (
            worst_k is None
            or candidate["max_abs"] > worst_k["max_abs"]
        ):
            worst_k = candidate

    for layer, pc, thor in all_v:
        s = stats_detail(pc, thor)
        candidate = {
            "layer": layer,
            **s,
        }
        if (
            worst_v is None
            or candidate["max_abs"] > worst_v["max_abs"]
        ):
            worst_v = candidate

    print()
    print("=" * 100)
    print("GLOBAL WORST ELEMENTS")
    print("=" * 100)

    print("\nK:")
    print(f"  layer       : {worst_k['layer']}")
    print(f"  index       : {worst_k['worst_index']}")
    print(f"  flat index  : {worst_k['worst_flat_index']}")
    print(f"  PC value    : {worst_k['worst_pc_value']:.9e}")
    print(f"  Thor value  : {worst_k['worst_thor_value']:.9e}")
    print(
        f"  Thor-PC     : "
        f"{worst_k['worst_difference_thor_minus_pc']:.9e}"
    )
    print(f"  abs diff    : {worst_k['worst_abs_difference']:.9e}")

    print("\nV:")
    print(f"  layer       : {worst_v['layer']}")
    print(f"  index       : {worst_v['worst_index']}")
    print(f"  flat index  : {worst_v['worst_flat_index']}")
    print(f"  PC value    : {worst_v['worst_pc_value']:.9e}")
    print(f"  Thor value  : {worst_v['worst_thor_value']:.9e}")
    print(
        f"  Thor-PC     : "
        f"{worst_v['worst_difference_thor_minus_pc']:.9e}"
    )
    print(f"  abs diff    : {worst_v['worst_abs_difference']:.9e}")

    # ---------------------------------------------------------------
    # Detailed top errors for global worst layers.
    # ---------------------------------------------------------------
    wk_layer = worst_k["layer"]
    wv_layer = worst_v["layer"]

    wk_pc = np.load(
        ref / f"pc_official_kv_k_{wk_layer:02d}.npy"
    ).astype(np.float32)
    wk_thor = all_k[wk_layer][2]

    wv_pc = np.load(
        ref / f"pc_official_kv_v_{wv_layer:02d}.npy"
    ).astype(np.float32)
    wv_thor = all_v[wv_layer][2]

    print()
    print("=" * 120)
    print(f"TOP {args.top_k} K ERRORS — layer {wk_layer}")
    print("=" * 120)
    for x in top_k_errors(wk_pc, wk_thor, args.top_k):
        print(
            f"idx={x['index']} "
            f"PC={x['pc']:.9e} "
            f"Thor={x['thor']:.9e} "
            f"diff={x['difference_thor_minus_pc']:.9e}"
        )

    print()
    print("=" * 120)
    print(f"TOP {args.top_k} V ERRORS — layer {wv_layer}")
    print("=" * 120)
    for x in top_k_errors(wv_pc, wv_thor, args.top_k):
        print(
            f"idx={x['index']} "
            f"PC={x['pc']:.9e} "
            f"Thor={x['thor']:.9e} "
            f"diff={x['difference_thor_minus_pc']:.9e}"
        )

    # ---------------------------------------------------------------
    # Compact global interpretation hints.
    # ---------------------------------------------------------------
    k_maxes = np.array(
        [x["k"]["max_abs"] for x in per_layer],
        dtype=np.float64,
    )
    v_maxes = np.array(
        [x["v"]["max_abs"] for x in per_layer],
        dtype=np.float64,
    )

    result = {
        "type": "12k_prefill_kv_detailed_diagnostic",
        "reference": "PC Official Forward K/V capture",
        "engine": str(Path(args.engine).resolve()),
        "num_layers": NUM_LAYERS,
        "visual_shape": list(visual.shape),
        "per_layer": per_layer,
        "global_worst_k": worst_k,
        "global_worst_v": worst_v,
        "layer_max_summary": {
            "k_max_abs_min": float(k_maxes.min()),
            "k_max_abs_max": float(k_maxes.max()),
            "k_max_abs_mean": float(k_maxes.mean()),
            "v_max_abs_min": float(v_maxes.min()),
            "v_max_abs_max": float(v_maxes.max()),
            "v_max_abs_mean": float(v_maxes.mean()),
        },
    }

    out = ref / "metrics_12k_prefill_kv_detailed.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    eng.cleanup()

    print()
    print("=" * 100)
    print("Saved:")
    print(f"  {out}")
    print()
    print("[OK] Prefill K/V detailed diagnostic completed.")
    print()
    print("Next decision should be based on:")
    print("  1) whether Thor values are mostly FP16-representable,")
    print("  2) whether error is concentrated in specific layers/channels,")
    print("  3) worst-element PC vs Thor values and indices,")
    print("  4) relative L2 error across layers.")


if __name__ == "__main__":
    main()
