#!/usr/bin/env python3
"""
12k-5 — Prefill K/V Head-level + FP16-pattern Diagnostic

Purpose
-------
Run the Thor TensorRT Prefill engine once and compare its 24-layer K/V cache
against the PC Official Forward K/V reference.

This is NOT a latency benchmark.
It does NOT run the Action Expert or Euler.

New diagnostics:
  1. Per-layer / per-head K and V error statistics
  2. Identify the worst head for K and V
  3. Compare PC/Thor min/max/mean/std per head
  4. Check whether captured PC/Thor values are exactly FP16-representable
  5. Estimate error in units of FP16 spacing (ULP-like diagnostic)
  6. Error-distribution buckets: <=1, <=2, <=4, <=8, <=16, >16 FP16 spacings
  7. Global worst K/V element with layer/head/token/channel index
  8. Save JSON + CSV summaries

Expected reference:
  ~/vblkdev2/VaVAM_Thor/pc_reference/
    visual_tokens.npy
    pc_official_kv_k_00.npy ... pc_official_kv_k_23.npy
    pc_official_kv_v_00.npy ... pc_official_kv_v_23.npy

Default engine:
  ~/vblkdev2/VaVAM_Thor/Engines/vavam_joint_kv_prefill_B_v10_fp16.engine
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cuda


ROOT = Path.home() / "vblkdev2" / "VaVAM_Thor"
DEFAULT_ENGINE = ROOT / "Engines" / "vavam_joint_kv_prefill_B_v10_fp16.engine"
DEFAULT_REFERENCE = ROOT / "pc_reference"
NUM_LAYERS = 24
EXPECTED_HEADS = 8


def check(result, name):
    err = result[0] if isinstance(result, tuple) else result
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {err}")
    return result


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def fp16_exact_fraction(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.mean(x.astype(np.float16).astype(np.float32) == x))


def fp16_spacing(x):
    """
    Return the local distance between adjacent FP16 representable values,
    evaluated at x. This is used only as an ULP-like scale indicator.

    For zero, use the smallest positive normal/subnormal spacing represented
    by np.nextafter(np.float16(0), np.float16(1)).
    """
    x = np.asarray(x, dtype=np.float32)
    xf = x.astype(np.float16)

    next_up = np.nextafter(
        xf,
        np.float16(np.inf),
    ).astype(np.float32)

    spacing = np.abs(next_up - xf)

    # At +inf/-inf or unusual overflow cases, fall back to the distance to
    # the previous representable value.
    bad = ~np.isfinite(spacing) | (spacing == 0)
    if np.any(bad):
        next_down = np.nextafter(
            xf,
            np.float16(-np.inf),
        ).astype(np.float32)
        spacing_down = np.abs(xf - next_down)
        spacing = np.where(
            bad,
            spacing_down,
            spacing,
        )

    # For zero, nextafter toward +1 gives the smallest subnormal step.
    zero = (xf == np.float16(0))
    if np.any(zero):
        min_sub = float(
            np.nextafter(
                np.float16(0),
                np.float16(1),
            ).astype(np.float32)
        )
        spacing = np.where(zero, min_sub, spacing)

    spacing = np.maximum(spacing, np.finfo(np.float32).tiny)
    return spacing


def basic_stats(pc, thor):
    pc = np.asarray(pc, dtype=np.float32)
    thor = np.asarray(thor, dtype=np.float32)

    diff = thor - pc
    ad = np.abs(diff)

    pc_l2 = float(np.linalg.norm(pc.reshape(-1)))
    diff_l2 = float(np.linalg.norm(diff.reshape(-1)))

    return {
        "num_elements": int(pc.size),
        "max_abs": float(ad.max()),
        "mean_abs": float(ad.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_l2_error": float(
            diff_l2 / pc_l2 if pc_l2 > 0 else 0.0
        ),
        "pc_min": float(pc.min()),
        "pc_max": float(pc.max()),
        "pc_mean": float(pc.mean()),
        "pc_std": float(pc.std()),
        "thor_min": float(thor.min()),
        "thor_max": float(thor.max()),
        "thor_mean": float(thor.mean()),
        "thor_std": float(thor.std()),
        "pc_fp16_exact_fraction": fp16_exact_fraction(pc),
        "thor_fp16_exact_fraction": fp16_exact_fraction(thor),
    }


def ulp_stats(pc, thor):
    pc = np.asarray(pc, dtype=np.float32)
    thor = np.asarray(thor, dtype=np.float32)

    ad = np.abs(thor - pc)
    spacing = fp16_spacing(pc)
    ratio = ad / spacing

    finite = np.isfinite(ratio)
    ratio = ratio[finite]

    if ratio.size == 0:
        return {
            "mean_spacing": None,
            "mean_abs_error_in_fp16_spacing": None,
            "median_abs_error_in_fp16_spacing": None,
            "max_abs_error_in_fp16_spacing": None,
            "fraction_le_1_spacing": None,
            "fraction_le_2_spacing": None,
            "fraction_le_4_spacing": None,
            "fraction_le_8_spacing": None,
            "fraction_le_16_spacing": None,
            "fraction_gt_16_spacing": None,
        }

    return {
        "mean_spacing": float(np.mean(spacing[finite])),
        "mean_abs_error_in_fp16_spacing": float(np.mean(ratio)),
        "median_abs_error_in_fp16_spacing": float(np.median(ratio)),
        "max_abs_error_in_fp16_spacing": float(np.max(ratio)),
        "fraction_le_1_spacing": float(np.mean(ratio <= 1.0)),
        "fraction_le_2_spacing": float(np.mean(ratio <= 2.0)),
        "fraction_le_4_spacing": float(np.mean(ratio <= 4.0)),
        "fraction_le_8_spacing": float(np.mean(ratio <= 8.0)),
        "fraction_le_16_spacing": float(np.mean(ratio <= 16.0)),
        "fraction_gt_16_spacing": float(np.mean(ratio > 16.0)),
    }


def worst_element(pc, thor):
    pc = np.asarray(pc, dtype=np.float32)
    thor = np.asarray(thor, dtype=np.float32)
    diff = thor - pc
    ad = np.abs(diff)

    flat_i = int(np.argmax(ad.reshape(-1)))
    idx = np.unravel_index(flat_i, ad.shape)

    return {
        "index": list(idx),
        "flat_index": flat_i,
        "pc_value": float(pc[idx]),
        "thor_value": float(thor[idx]),
        "difference_thor_minus_pc": float(diff[idx]),
        "abs_difference": float(ad[idx]),
    }


def head_stats(pc, thor):
    """
    Expected shape: [B, H, S, D], with B=1.
    Returns one dict per head.
    """
    if pc.ndim != 4 or thor.ndim != 4:
        raise RuntimeError(
            f"Expected [B,H,S,D], got PC={pc.shape}, Thor={thor.shape}"
        )
    if pc.shape != thor.shape:
        raise RuntimeError(
            f"Shape mismatch: PC={pc.shape}, Thor={thor.shape}"
        )

    if pc.shape[0] != 1:
        raise RuntimeError(f"Expected B=1, got {pc.shape[0]}")

    heads = pc.shape[1]
    rows = []

    for h in range(heads):
        p = pc[:, h, :, :]
        t = thor[:, h, :, :]
        s = basic_stats(p, t)
        u = ulp_stats(p, t)
        w = worst_element(p, t)

        rows.append({
            "head": h,
            "stats": s,
            "fp16_spacing": u,
            "worst": w,
        })

    return rows


class Engine:
    def __init__(self, path):
        self.path = Path(path)
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)

        with open(self.path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine: {self.path}"
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                "Failed to create TensorRT execution context"
            )

        self.ptrs = {}

        print("=" * 90)
        print(f"Loaded engine: {self.path}")
        print("=" * 90)

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

            if not self.context.set_tensor_address(
                name,
                int(ptr),
            ):
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
    print("=" * 90)
    print("12k-5 — Prefill K/V Head-level + FP16-pattern Diagnostic")
    print("=" * 90)
    print(f"Reference dir : {ref}")
    print(f"Visual shape  : {visual.shape}")
    print()
    print("Runs TensorRT Prefill ONCE.")
    print("No Action Expert. No Euler. No per-step diffusion loop.")

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
    per_head_rows = []

    global_worst_k = None
    global_worst_v = None

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

        if pc_k.shape != thor_k.shape:
            raise RuntimeError(
                f"K shape mismatch layer {layer}: "
                f"PC={pc_k.shape}, Thor={thor_k.shape}"
            )
        if pc_v.shape != thor_v.shape:
            raise RuntimeError(
                f"V shape mismatch layer {layer}: "
                f"PC={pc_v.shape}, Thor={thor_v.shape}"
            )

        if pc_k.shape[1] != EXPECTED_HEADS:
            raise RuntimeError(
                f"Expected {EXPECTED_HEADS} heads, got "
                f"{pc_k.shape[1]}"
            )

        ks = basic_stats(pc_k, thor_k)
        vs = basic_stats(pc_v, thor_v)
        ku = ulp_stats(pc_k, thor_k)
        vu = ulp_stats(pc_v, thor_v)
        kw = worst_element(pc_k, thor_k)
        vw = worst_element(pc_v, thor_v)

        if (
            global_worst_k is None
            or kw["abs_difference"] > global_worst_k["abs_difference"]
        ):
            global_worst_k = {
                "layer": layer,
                **kw,
            }

        if (
            global_worst_v is None
            or vw["abs_difference"] > global_worst_v["abs_difference"]
        ):
            global_worst_v = {
                "layer": layer,
                **vw,
            }

        per_layer.append({
            "layer": layer,
            "k": {
                "stats": ks,
                "fp16_spacing": ku,
                "worst": kw,
            },
            "v": {
                "stats": vs,
                "fp16_spacing": vu,
                "worst": vw,
            },
        })

        k_heads = head_stats(pc_k, thor_k)
        v_heads = head_stats(pc_v, thor_v)

        for h in range(EXPECTED_HEADS):
            per_head_rows.append({
                "layer": layer,
                "kind": "K",
                "head": h,
                **k_heads[h]["stats"],
                **{
                    f"fp16_{key}": value
                    for key, value in k_heads[h]["fp16_spacing"].items()
                },
                "worst_index": k_heads[h]["worst"]["index"],
                "worst_pc_value": k_heads[h]["worst"]["pc_value"],
                "worst_thor_value": k_heads[h]["worst"]["thor_value"],
                "worst_abs_difference": k_heads[h]["worst"]["abs_difference"],
            })
            per_head_rows.append({
                "layer": layer,
                "kind": "V",
                "head": h,
                **v_heads[h]["stats"],
                **{
                    f"fp16_{key}": value
                    for key, value in v_heads[h]["fp16_spacing"].items()
                },
                "worst_index": v_heads[h]["worst"]["index"],
                "worst_pc_value": v_heads[h]["worst"]["pc_value"],
                "worst_thor_value": v_heads[h]["worst"]["thor_value"],
                "worst_abs_difference": v_heads[h]["worst"]["abs_difference"],
            })

    # ---------------------------------------------------------------
    # Print layer summary.
    # ---------------------------------------------------------------
    print()
    print("=" * 170)
    print("24-layer summary")
    print("=" * 170)
    print(
        "layer | "
        "K max      K mean     K RMSE    "
        "V max      V mean     V RMSE"
    )
    print("-" * 170)

    for x in per_layer:
        print(
            f"{x['layer']:5d} | "
            f"{x['k']['stats']['max_abs']:9.3e} "
            f"{x['k']['stats']['mean_abs']:9.3e} "
            f"{x['k']['stats']['rmse']:9.3e} "
            f"{x['v']['stats']['max_abs']:9.3e} "
            f"{x['v']['stats']['mean_abs']:9.3e} "
            f"{x['v']['stats']['rmse']:9.3e}"
        )

    # ---------------------------------------------------------------
    # Find worst head.
    # ---------------------------------------------------------------
    k_head_candidates = [
        r for r in per_head_rows if r["kind"] == "K"
    ]
    v_head_candidates = [
        r for r in per_head_rows if r["kind"] == "V"
    ]

    worst_k_head_max = max(
        k_head_candidates,
        key=lambda r: r["max_abs"],
    )
    worst_v_head_max = max(
        v_head_candidates,
        key=lambda r: r["max_abs"],
    )

    worst_k_head_rmse = max(
        k_head_candidates,
        key=lambda r: r["rmse"],
    )
    worst_v_head_rmse = max(
        v_head_candidates,
        key=lambda r: r["rmse"],
    )

    # Aggregate by head across all 24 layers.
    aggregate_head = []
    for kind in ("K", "V"):
        for h in range(EXPECTED_HEADS):
            rows = [
                r for r in per_head_rows
                if r["kind"] == kind and r["head"] == h
            ]
            all_mean = np.array(
                [r["mean_abs"] for r in rows],
                dtype=np.float64,
            )
            all_rmse = np.array(
                [r["rmse"] for r in rows],
                dtype=np.float64,
            )
            all_max = np.array(
                [r["max_abs"] for r in rows],
                dtype=np.float64,
            )

            aggregate_head.append({
                "kind": kind,
                "head": h,
                "mean_of_layer_mean_abs": float(all_mean.mean()),
                "mean_of_layer_rmse": float(all_rmse.mean()),
                "max_abs_across_layers": float(all_max.max()),
                "layer_of_max_abs": int(
                    rows[int(np.argmax(all_max))]["layer"]
                ),
            })

    worst_aggregate_k = max(
        [x for x in aggregate_head if x["kind"] == "K"],
        key=lambda x: x["mean_of_layer_rmse"],
    )
    worst_aggregate_v = max(
        [x for x in aggregate_head if x["kind"] == "V"],
        key=lambda x: x["mean_of_layer_rmse"],
    )

    # ---------------------------------------------------------------
    # Print head-level summary.
    # ---------------------------------------------------------------
    print()
    print("=" * 150)
    print("Worst layer/head combinations")
    print("=" * 150)

    print(
        f"K worst max_abs : layer={worst_k_head_max['layer']} "
        f"head={worst_k_head_max['head']} "
        f"max_abs={worst_k_head_max['max_abs']:.8e} "
        f"RMSE={worst_k_head_max['rmse']:.8e}"
    )
    print(
        f"K worst RMSE   : layer={worst_k_head_rmse['layer']} "
        f"head={worst_k_head_rmse['head']} "
        f"max_abs={worst_k_head_rmse['max_abs']:.8e} "
        f"RMSE={worst_k_head_rmse['rmse']:.8e}"
    )
    print(
        f"V worst max_abs : layer={worst_v_head_max['layer']} "
        f"head={worst_v_head_max['head']} "
        f"max_abs={worst_v_head_max['max_abs']:.8e} "
        f"RMSE={worst_v_head_max['rmse']:.8e}"
    )
    print(
        f"V worst RMSE   : layer={worst_v_head_rmse['layer']} "
        f"head={worst_v_head_rmse['head']} "
        f"max_abs={worst_v_head_rmse['max_abs']:.8e} "
        f"RMSE={worst_v_head_rmse['rmse']:.8e}"
    )

    print()
    print("Aggregate across all 24 layers:")
    print(
        f"  K head {worst_aggregate_k['head']} "
        f"has highest mean layer RMSE: "
        f"{worst_aggregate_k['mean_of_layer_rmse']:.8e}"
    )
    print(
        f"  V head {worst_aggregate_v['head']} "
        f"has highest mean layer RMSE: "
        f"{worst_aggregate_v['mean_of_layer_rmse']:.8e}"
    )

    # ---------------------------------------------------------------
    # Print per-head matrix of RMSE and max_abs.
    # ---------------------------------------------------------------
    print()
    print("=" * 150)
    print("Per-layer K RMSE by head")
    print("=" * 150)
    print("layer " + " ".join([f"H{h:>7d}" for h in range(EXPECTED_HEADS)]))
    for layer in range(NUM_LAYERS):
        vals = [
            next(
                r["rmse"]
                for r in k_head_candidates
                if r["layer"] == layer and r["head"] == h
            )
            for h in range(EXPECTED_HEADS)
        ]
        print(
            f"{layer:5d} "
            + " ".join(f"{x:8.2e}" for x in vals)
        )

    print()
    print("=" * 150)
    print("Per-layer V RMSE by head")
    print("=" * 150)
    print("layer " + " ".join([f"H{h:>7d}" for h in range(EXPECTED_HEADS)]))
    for layer in range(NUM_LAYERS):
        vals = [
            next(
                r["rmse"]
                for r in v_head_candidates
                if r["layer"] == layer and r["head"] == h
            )
            for h in range(EXPECTED_HEADS)
        ]
        print(
            f"{layer:5d} "
            + " ".join(f"{x:8.2e}" for x in vals)
        )

    # ---------------------------------------------------------------
    # FP16 pattern summary.
    # ---------------------------------------------------------------
    layer_k_fp16 = np.array([
        x["k"]["stats"]["thor_fp16_exact_fraction"]
        for x in per_layer
    ])
    layer_v_fp16 = np.array([
        x["v"]["stats"]["thor_fp16_exact_fraction"]
        for x in per_layer
    ])
    pc_k_fp16 = np.array([
        x["k"]["stats"]["pc_fp16_exact_fraction"]
        for x in per_layer
    ])
    pc_v_fp16 = np.array([
        x["v"]["stats"]["pc_fp16_exact_fraction"]
        for x in per_layer
    ])

    print()
    print("=" * 110)
    print("FP16 representability summary")
    print("=" * 110)
    print(
        f"PC K exact FP16 fraction   : "
        f"mean={pc_k_fp16.mean():.9f}, "
        f"min={pc_k_fp16.min():.9f}"
    )
    print(
        f"Thor K exact FP16 fraction : "
        f"mean={layer_k_fp16.mean():.9f}, "
        f"min={layer_k_fp16.min():.9f}"
    )
    print(
        f"PC V exact FP16 fraction   : "
        f"mean={pc_v_fp16.mean():.9f}, "
        f"min={pc_v_fp16.min():.9f}"
    )
    print(
        f"Thor V exact FP16 fraction : "
        f"mean={layer_v_fp16.mean():.9f}, "
        f"min={layer_v_fp16.min():.9f}"
    )

    print()
    print("Interpretation:")
    print(
        "  These fractions describe the saved output values only; "
        "they do NOT prove the internal TensorRT precision."
    )

    # ---------------------------------------------------------------
    # Global worst element.
    # ---------------------------------------------------------------
    print()
    print("=" * 120)
    print("GLOBAL WORST ELEMENTS")
    print("=" * 120)

    print("\nK:")
    print(
        f"  layer      : {global_worst_k['layer']}\n"
        f"  index      : {global_worst_k['index']}\n"
        f"  PC value   : {global_worst_k['pc_value']:.9e}\n"
        f"  Thor value : {global_worst_k['thor_value']:.9e}\n"
        f"  Thor-PC    : {global_worst_k['difference_thor_minus_pc']:.9e}\n"
        f"  abs diff   : {global_worst_k['abs_difference']:.9e}"
    )

    print("\nV:")
    print(
        f"  layer      : {global_worst_v['layer']}\n"
        f"  index      : {global_worst_v['index']}\n"
        f"  PC value   : {global_worst_v['pc_value']:.9e}\n"
        f"  Thor value : {global_worst_v['thor_value']:.9e}\n"
        f"  Thor-PC    : {global_worst_v['difference_thor_minus_pc']:.9e}\n"
        f"  abs diff   : {global_worst_v['abs_difference']:.9e}"
    )

    # ---------------------------------------------------------------
    # Save CSV.
    # ---------------------------------------------------------------
    csv_path = ref / "metrics_12k_prefill_kv_head_diagnostic.csv"

    fieldnames = sorted({
        key
        for row in per_head_rows
        for key in row.keys()
    })

    # Serialize list-valued worst_index as a string.
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in per_head_rows:
            out_row = dict(row)
            out_row["worst_index"] = str(
                out_row["worst_index"]
            )
            writer.writerow(out_row)

    # ---------------------------------------------------------------
    # Save JSON.
    # ---------------------------------------------------------------
    result = {
        "type": "12k-5_prefill_kv_head_level_fp16_diagnostic",
        "engine": str(Path(args.engine).resolve()),
        "reference_dir": str(ref),
        "num_layers": NUM_LAYERS,
        "num_heads": EXPECTED_HEADS,
        "visual_shape": list(visual.shape),
        "per_layer": per_layer,
        "per_head": per_head_rows,
        "aggregate_head": aggregate_head,
        "global_worst_k": global_worst_k,
        "global_worst_v": global_worst_v,
        "worst_k_head_max_abs": worst_k_head_max,
        "worst_v_head_max_abs": worst_v_head_max,
        "worst_k_head_rmse": worst_k_head_rmse,
        "worst_v_head_rmse": worst_v_head_rmse,
        "worst_aggregate_k": worst_aggregate_k,
        "worst_aggregate_v": worst_aggregate_v,
        "fp16_representability": {
            "pc_k_mean": float(pc_k_fp16.mean()),
            "pc_k_min": float(pc_k_fp16.min()),
            "thor_k_mean": float(layer_k_fp16.mean()),
            "thor_k_min": float(layer_k_fp16.min()),
            "pc_v_mean": float(pc_v_fp16.mean()),
            "pc_v_min": float(pc_v_fp16.min()),
            "thor_v_mean": float(layer_v_fp16.mean()),
            "thor_v_min": float(layer_v_fp16.min()),
        },
    }

    json_path = ref / "metrics_12k_prefill_kv_head_diagnostic.json"
    with open(json_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
            default=json_default,
        )

    eng.cleanup()

    print()
    print("=" * 100)
    print("Saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print()
    print("[OK] 12k-5 Prefill K/V Head-level diagnostic completed.")


if __name__ == "__main__":
    main()
