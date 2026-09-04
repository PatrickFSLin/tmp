#!/usr/bin/env python3
"""
12k-6 — Prefill divergence-onset diagnostic

Goal
----
Use the existing PC Official 24-layer K/V capture and the existing Thor
TensorRT Prefill engine to determine WHERE the PC↔Thor K/V discrepancy
becomes material.

Important:
  - This is NOT an internal TensorRT graph dump.
  - The current engine exposes only final K/V outputs for each layer.
  - Therefore this experiment can localize the first OUTPUT LAYER where
    divergence becomes material, but cannot prove the exact internal
    operation (Q/K matmul, softmax, LayerNorm, etc.) responsible.
  - No engine rebuild is performed.
  - No Action Expert or Euler is executed.
  - Prefill is executed once.

The script reports:
  * absolute and relative-L2 K/V error for all 24 layers
  * K/V RMS magnitude of the PC reference
  * error/RMS ratio
  * layer-to-layer error growth
  * first layer crossing several relative-error thresholds
  * a compact "onset window" around the first material divergence
  * correlation between K and V error across layers

It saves:
  metrics_12k_prefill_divergence_onset.json
  metrics_12k_prefill_divergence_onset.csv
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


def compare(pc, thor):
    pc = np.asarray(pc, dtype=np.float32)
    thor = np.asarray(thor, dtype=np.float32)

    if pc.shape != thor.shape:
        raise RuntimeError(
            f"Shape mismatch: PC={pc.shape}, Thor={thor.shape}"
        )

    diff = thor - pc
    ad = np.abs(diff)

    pc_rms = float(np.sqrt(np.mean(pc * pc)))
    diff_rms = float(np.sqrt(np.mean(diff * diff)))

    pc_l2 = float(np.linalg.norm(pc.reshape(-1)))
    diff_l2 = float(np.linalg.norm(diff.reshape(-1)))

    return {
        "num_elements": int(pc.size),
        "max_abs": float(ad.max()),
        "mean_abs": float(ad.mean()),
        "rmse": diff_rms,
        "relative_l2": float(
            diff_l2 / pc_l2 if pc_l2 > 0 else 0.0
        ),
        "pc_rms": pc_rms,
        "pc_mean_abs": float(np.mean(np.abs(pc))),
        "pc_abs_p95": float(np.percentile(np.abs(pc), 95)),
        "pc_abs_p99": float(np.percentile(np.abs(pc), 99)),
        "pc_min": float(pc.min()),
        "pc_max": float(pc.max()),
        "thor_min": float(thor.min()),
        "thor_max": float(thor.max()),
        # RMSE / reference RMS is a useful scale-normalized measure.
        "error_to_pc_rms": float(
            diff_rms / pc_rms if pc_rms > 0 else 0.0
        ),
    }


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
                name, int(ptr)
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
            check(cuda.cuMemFree(int(ptr)), "cuMemFree")
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


def threshold_crossings(rows, metric_key, thresholds):
    result = {}
    for th in thresholds:
        found = None
        for r in rows:
            if r[metric_key] >= th:
                found = int(r["layer"])
                break
        result[str(th)] = found
    return result


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
    print("12k-6 — Prefill divergence-onset diagnostic")
    print("=" * 90)
    print(f"Reference dir : {ref}")
    print(f"Visual shape  : {visual.shape}")
    print()
    print("Runs TensorRT Prefill ONCE.")
    print("No Action Expert. No Euler.")
    print()
    print(
        "NOTE: this localizes the first OUTPUT LAYER where divergence "
        "becomes material; it does not identify the exact internal op."
    )

    check(cuda.cuInit(0), "cuInit")
    device = check(
        cuda.cuDeviceGet(0),
        "cuDeviceGet",
    )[1]
    ctx = check(
        cuda.cuDevicePrimaryCtxRetain(device),
        "cuDevicePrimaryCtxRetain",
    )[1]
    check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    stream = check(
        cuda.cuStreamCreate(0),
        "cuStreamCreate",
    )[1]

    eng = Engine(args.engine)

    inputs = eng.input_names()
    outputs = eng.output_names()

    if len(inputs) != 1:
        raise RuntimeError(
            f"Expected one input, got {inputs}"
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
            f"K={len(k_names)}, V={len(v_names)}"
        )

    rows = []

    for layer in range(NUM_LAYERS):
        pc_k = np.load(
            ref / f"pc_official_kv_k_{layer:02d}.npy"
        ).astype(np.float32)
        pc_v = np.load(
            ref / f"pc_official_kv_v_{layer:02d}.npy"
        ).astype(np.float32)

        thor_k = np.empty(
            eng.shape(k_names[layer]),
            dtype=eng.dtype(k_names[layer]),
        )
        thor_v = np.empty(
            eng.shape(v_names[layer]),
            dtype=eng.dtype(v_names[layer]),
        )

        d2h(eng.ptrs[k_names[layer]], thor_k)
        d2h(eng.ptrs[v_names[layer]], thor_v)

        k = compare(
            pc_k,
            thor_k.astype(np.float32),
        )
        v = compare(
            pc_v,
            thor_v.astype(np.float32),
        )

        rows.append({
            "layer": layer,
            "k_max_abs": k["max_abs"],
            "k_mean_abs": k["mean_abs"],
            "k_rmse": k["rmse"],
            "k_relative_l2": k["relative_l2"],
            "k_pc_rms": k["pc_rms"],
            "k_error_to_pc_rms": k["error_to_pc_rms"],
            "v_max_abs": v["max_abs"],
            "v_mean_abs": v["mean_abs"],
            "v_rmse": v["rmse"],
            "v_relative_l2": v["relative_l2"],
            "v_pc_rms": v["pc_rms"],
            "v_error_to_pc_rms": v["error_to_pc_rms"],
        })

    # ---------------------------------------------------------------
    # Layer-to-layer growth.
    # ---------------------------------------------------------------
    for i, r in enumerate(rows):
        if i == 0:
            r["k_rmse_growth_vs_prev"] = None
            r["v_rmse_growth_vs_prev"] = None
            r["k_relative_l2_growth_vs_prev"] = None
            r["v_relative_l2_growth_vs_prev"] = None
        else:
            p = rows[i - 1]

            r["k_rmse_growth_vs_prev"] = float(
                r["k_rmse"] / p["k_rmse"]
                if p["k_rmse"] > 0 else np.inf
            )
            r["v_rmse_growth_vs_prev"] = float(
                r["v_rmse"] / p["v_rmse"]
                if p["v_rmse"] > 0 else np.inf
            )
            r["k_relative_l2_growth_vs_prev"] = float(
                r["k_relative_l2"] / p["k_relative_l2"]
                if p["k_relative_l2"] > 0 else np.inf
            )
            r["v_relative_l2_growth_vs_prev"] = float(
                r["v_relative_l2"] / p["v_relative_l2"]
                if p["v_relative_l2"] > 0 else np.inf
            )

    # ---------------------------------------------------------------
    # Threshold crossings.
    # Error-to-reference-RMS is deliberately used here instead of an
    # arbitrary absolute threshold because K/V magnitudes change by layer.
    # ---------------------------------------------------------------
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

    k_cross = threshold_crossings(
        rows,
        "k_error_to_pc_rms",
        thresholds,
    )
    v_cross = threshold_crossings(
        rows,
        "v_error_to_pc_rms",
        thresholds,
    )

    # First layer where the error becomes larger than 2x the previous
    # layer's error, as an "abrupt growth" indicator.
    k_growth_events = []
    v_growth_events = []

    for r in rows[1:]:
        if (
            r["k_rmse_growth_vs_prev"] is not None
            and r["k_rmse_growth_vs_prev"] >= 2.0
        ):
            k_growth_events.append(int(r["layer"]))
        if (
            r["v_rmse_growth_vs_prev"] is not None
            and r["v_rmse_growth_vs_prev"] >= 2.0
        ):
            v_growth_events.append(int(r["layer"]))

    # ---------------------------------------------------------------
    # Correlation of layer-wise K/V errors.
    # ---------------------------------------------------------------
    k_rmse = np.array(
        [r["k_rmse"] for r in rows],
        dtype=np.float64,
    )
    v_rmse = np.array(
        [r["v_rmse"] for r in rows],
        dtype=np.float64,
    )

    k_rel = np.array(
        [r["k_relative_l2"] for r in rows],
        dtype=np.float64,
    )
    v_rel = np.array(
        [r["v_relative_l2"] for r in rows],
        dtype=np.float64,
    )

    kv_rmse_corr = float(
        np.corrcoef(k_rmse, v_rmse)[0, 1]
    )
    kv_rel_corr = float(
        np.corrcoef(k_rel, v_rel)[0, 1]
    )

    # ---------------------------------------------------------------
    # Identify first "material" output-layer onset using 5% error/RMS.
    # This is a diagnostic convention, NOT a correctness threshold.
    # ---------------------------------------------------------------
    first_k_material = k_cross["0.05"]
    first_v_material = v_cross["0.05"]

    onset_layers = [
        x for x in (first_k_material, first_v_material)
        if x is not None
    ]
    first_material = min(onset_layers) if onset_layers else None

    # Also find the first layer where either K or V exceeds 10%.
    first_10 = [
        x for x in (
            k_cross["0.1"],
            v_cross["0.1"],
        )
        if x is not None
    ]
    first_10 = min(first_10) if first_10 else None

    # Window ±2 layers around first material onset.
    if first_material is not None:
        window_lo = max(0, first_material - 2)
        window_hi = min(NUM_LAYERS - 1, first_material + 2)
        onset_window = rows[window_lo:window_hi + 1]
    else:
        onset_window = []

    # ---------------------------------------------------------------
    # Print full compact table.
    # ---------------------------------------------------------------
    print()
    print("=" * 150)
    print("Layer-wise divergence")
    print("=" * 150)
    print(
        "layer | "
        "K RMSE   K err/RMS  K relL2 | "
        "V RMSE   V err/RMS  V relL2 | "
        "K growth V growth"
    )
    print("-" * 150)

    for r in rows:
        kg = (
            f"{r['k_rmse_growth_vs_prev']:.2f}x"
            if r["k_rmse_growth_vs_prev"] is not None
            else "   - "
        )
        vg = (
            f"{r['v_rmse_growth_vs_prev']:.2f}x"
            if r["v_rmse_growth_vs_prev"] is not None
            else "   - "
        )

        print(
            f"{r['layer']:5d} | "
            f"{r['k_rmse']:.3e} "
            f"{r['k_error_to_pc_rms']:.3e} "
            f"{r['k_relative_l2']:.3e} | "
            f"{r['v_rmse']:.3e} "
            f"{r['v_error_to_pc_rms']:.3e} "
            f"{r['v_relative_l2']:.3e} | "
            f"{kg:>7s} {vg:>7s}"
        )

    # ---------------------------------------------------------------
    # Print threshold crossings.
    # ---------------------------------------------------------------
    print()
    print("=" * 110)
    print("First layer crossing error/reference-RMS thresholds")
    print("=" * 110)
    print("threshold | K first layer | V first layer")
    print("---------------------------------------------")
    for th in thresholds:
        print(
            f"{th:9.2%} | "
            f"{str(k_cross[str(th)]):>13s} | "
            f"{str(v_cross[str(th)]):>13s}"
        )

    print()
    print(
        "5% is used only as a convenient diagnostic marker; "
        "it is NOT a model accuracy threshold."
    )

    print()
    print("=" * 110)
    print("Potential abrupt-growth events (>=2x previous layer RMSE)")
    print("=" * 110)
    print(f"K layers: {k_growth_events if k_growth_events else 'none'}")
    print(f"V layers: {v_growth_events if v_growth_events else 'none'}")

    print()
    print("=" * 110)
    print("K/V layer-wise error correlation")
    print("=" * 110)
    print(f"corr(K RMSE, V RMSE)       : {kv_rmse_corr:.6f}")
    print(f"corr(K relative L2, V rel): {kv_rel_corr:.6f}")

    print()
    print("=" * 110)
    print("Divergence onset interpretation")
    print("=" * 110)
    print(f"First K >= 5% error/RMS : layer {first_k_material}")
    print(f"First V >= 5% error/RMS : layer {first_v_material}")
    print(f"First either >= 10%     : layer {first_10}")

    if first_material is not None:
        print()
        print(
            f"Onset window: layers "
            f"{window_lo}..{window_hi}"
        )
        print(
            "This identifies the first OUTPUT layer where the "
            "captured K/V difference becomes material."
        )
        print(
            "It does NOT prove which internal operation caused it."
        )
    else:
        print()
        print("No layer crossed the 5% error/RMS diagnostic marker.")

    # ---------------------------------------------------------------
    # Save CSV.
    # ---------------------------------------------------------------
    csv_path = ref / "metrics_12k_prefill_divergence_onset.csv"

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------------
    # Save JSON.
    # ---------------------------------------------------------------
    result = {
        "type": "12k-6_prefill_divergence_onset",
        "engine": str(Path(args.engine).resolve()),
        "reference_dir": str(ref),
        "visual_shape": list(visual.shape),
        "num_layers": NUM_LAYERS,
        "diagnostic_thresholds": thresholds,
        "threshold_semantics": (
            "error_to_pc_rms = K/V RMSE divided by the PC reference RMS "
            "for that layer; thresholds are diagnostic markers, not "
            "accuracy requirements."
        ),
        "per_layer": rows,
        "threshold_crossings": {
            "k": k_cross,
            "v": v_cross,
        },
        "abrupt_growth_layers": {
            "k": k_growth_events,
            "v": v_growth_events,
        },
        "first_k_material_layer_5pct": first_k_material,
        "first_v_material_layer_5pct": first_v_material,
        "first_either_layer_10pct": first_10,
        "onset_window": onset_window,
        "kv_error_correlation": {
            "rmse": kv_rmse_corr,
            "relative_l2": kv_rel_corr,
        },
        "global_summary": {
            "max_k_rmse": float(k_rmse.max()),
            "layer_max_k_rmse": int(np.argmax(k_rmse)),
            "max_v_rmse": float(v_rmse.max()),
            "layer_max_v_rmse": int(np.argmax(v_rmse)),
            "max_k_relative_l2": float(k_rel.max()),
            "layer_max_k_relative_l2": int(np.argmax(k_rel)),
            "max_v_relative_l2": float(v_rel.max()),
            "layer_max_v_relative_l2": int(np.argmax(v_rel)),
        },
    }

    json_path = ref / "metrics_12k_prefill_divergence_onset.json"

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
    print("[OK] 12k-6 Prefill divergence-onset diagnostic completed.")


if __name__ == "__main__":
    main()
