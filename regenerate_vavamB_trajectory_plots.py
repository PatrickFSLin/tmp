#!/usr/bin/env python3
"""
Regenerate VaVAM-B trajectory plots from an existing accuracy NPZ result.

Expected NPZ content (the script tries several common key names):
  - predictions:      (N, M, 6, 2) or (N, 1, M, 6, 2)
  - gt / gt_all:      (N, 6, 2)
  - scene_names:      (N,)
  - window_idxs:      (N,)

For each sample, the script plots:
  - GT trajectory
  - Best M=1 trajectory
  - Best M=4 trajectory
  - Best M=10 trajectory
  - Ego origin

"Best" means the trajectory with minimum ADE against GT within
the corresponding candidate set. No inference is performed.

Example:
  python3 regenerate_vavamB_trajectory_plots.py

Or:
  python3 regenerate_vavamB_trajectory_plots.py \
      --npz ./thor_eval/vavamB_nuscenes_accuracy_results.npz \
      --out-dir ./thor_eval/trajectory_plots_v2
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_NPZ = os.path.expanduser(
    "~/vblkdev2/VaVAM_Thor/thor_eval/vavamB_nuscenes_accuracy_results.npz"
)
DEFAULT_OUT = os.path.expanduser(
    "~/vblkdev2/VaVAM_Thor/thor_eval/trajectory_plots_v2"
)


def get_first(npz, names, required=True):
    """Return the first existing NPZ key from names."""
    for name in names:
        if name in npz.files:
            return npz[name]
    if required:
        raise KeyError(
            f"None of these keys were found: {names}\n"
            f"Available keys:\n  " + "\n  ".join(npz.files)
        )
    return None


def normalize_predictions(pred):
    """
    Normalize predictions to (N, M, T, 2).

    Supported examples:
      (N, M, T, 2)
      (N, 1, M, T, 2)
      (M, T, 2) for a single sample
    """
    pred = np.asarray(pred)

    if pred.ndim == 5:
        # Common accidental retained batch dimension:
        # (N, 1, M, T, 2) -> (N, M, T, 2)
        if pred.shape[1] == 1:
            pred = pred[:, 0]
        else:
            raise ValueError(
                f"Unsupported 5-D predictions shape {pred.shape}. "
                "Expected (N,1,M,T,2)."
            )

    if pred.ndim == 4:
        return pred

    if pred.ndim == 3 and pred.shape[-1] == 2:
        # Single sample: (M,T,2)
        return pred[None, ...]

    raise ValueError(
        f"Unsupported predictions shape {pred.shape}. "
        "Expected (N,M,T,2), (N,1,M,T,2), or (M,T,2)."
    )


def normalize_gt(gt, n_samples):
    """
    Normalize GT to (N, T, 2).
    """
    gt = np.asarray(gt)

    if gt.ndim == 2 and gt.shape[-1] == 2:
        if n_samples == 1:
            return gt[None, ...]
        # Could be flattened only if exactly N*T rows; do not guess.
        raise ValueError(
            f"GT has shape {gt.shape}, but predictions contain "
            f"{n_samples} samples."
        )

    if gt.ndim == 3 and gt.shape[-1] == 2:
        return gt

    if gt.ndim == 4 and gt.shape[1] == 1 and gt.shape[-1] == 2:
        return gt[:, 0]

    raise ValueError(
        f"Unsupported GT shape {gt.shape}. Expected (N,T,2) or (T,2)."
    )


def compute_ade(pred, gt):
    """
    pred: (M,T,2)
    gt:   (T,2)
    returns:
      ades: (M,)
    """
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"pred must be (M,T,2), got {pred.shape}")
    if gt.ndim != 2 or gt.shape[-1] != 2:
        raise ValueError(f"gt must be (T,2), got {gt.shape}")

    if pred.shape[1] != gt.shape[0]:
        raise ValueError(
            f"Trajectory length mismatch: prediction T={pred.shape[1]}, "
            f"GT T={gt.shape[0]}"
        )

    return np.linalg.norm(pred - gt[None, :, :], axis=-1).mean(axis=-1)


def as_text_array(x, n, default_prefix):
    """Convert scene/window metadata to length-N arrays."""
    if x is None:
        return np.array([f"{default_prefix}{i}" for i in range(n)], dtype=object)

    x = np.asarray(x)

    if x.ndim == 0:
        return np.array([str(x.item())] * n, dtype=object)

    x = x.reshape(-1)

    if len(x) != n:
        print(
            f"[WARN] Metadata length {len(x)} != N={n}; "
            f"using generated {default_prefix} values."
        )
        return np.array([f"{default_prefix}{i}" for i in range(n)], dtype=object)

    return np.array(
        [
            v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in x
        ],
        dtype=object,
    )


def plot_one(
    sample_index,
    gt,
    predictions,
    scene,
    window_idx,
    out_path,
):
    """
    predictions: (M,T,2), with M >= 10.
    """
    candidate_ms = [1, 4, 10]

    if predictions.shape[0] < 10:
        raise ValueError(
            f"Sample {sample_index} has only M={predictions.shape[0]} "
            "predictions; M=10 plot requires at least 10."
        )

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    # GT
    ax.plot(
        gt[:, 0],
        gt[:, 1],
        marker="o",
        markersize=5,
        linewidth=2.0,
        label="GT",
    )

    legend_items = []

    # Best candidate for M=1 / M=4 / M=10.
    for m in candidate_ms:
        candidates = predictions[:m]
        ades = compute_ade(candidates, gt)
        best_idx = int(np.argmin(ades))
        best_traj = candidates[best_idx]
        best_ade = float(ades[best_idx])

        # Keep the visual style close to the reference image:
        # x markers for predictions, one line per M.
        line = ax.plot(
            best_traj[:, 0],
            best_traj[:, 1],
            marker="x",
            markersize=6,
            markeredgewidth=1.8,
            linewidth=2.0,
            label=f"Best M={m} (traj {best_idx}, minADE={best_ade:.3f} m)",
        )[0]
        legend_items.append(line)

    # Ego origin
    ax.plot(
        0.0,
        0.0,
        marker="s",
        markersize=8,
        linestyle="None",
        label="Ego origin",
    )

    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    ax.set_title("VaVAM-B — GT / M=1 / M=4 / M=10", fontsize=13, pad=10)

    ax.text(
        0.5,
        1.015,
        f"dataset_index={sample_index} | scene={scene} | window={window_idx}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Let matplotlib determine sensible limits, but include the ego origin.
    all_xy = np.concatenate(
        [gt.reshape(-1, 2), predictions[:10].reshape(-1, 2), np.zeros((1, 2))],
        axis=0,
    )
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)

    # Small margin; avoid zero-width axes.
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    mx = 0.08 * dx
    my = 0.08 * dy
    ax.set_xlim(xmin - mx, xmax + mx)
    ax.set_ylim(ymin - my, ymax + my)

    ax.legend(
        loc="best",
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate VaVAM-B GT/M1/M4/M10 trajectory plots from NPZ."
    )
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First dataset index to plot.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to plot.",
    )
    args = parser.parse_args()

    npz_path = Path(os.path.expanduser(args.npz))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    print("=" * 72)
    print("VaVAM-B trajectory plot regeneration")
    print("=" * 72)
    print(f"NPZ     : {npz_path}")
    print(f"Output  : {out_dir}")
    print()

    with np.load(npz_path, allow_pickle=True) as data:
        print("NPZ keys:")
        for key in data.files:
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        print()

        pred_raw = get_first(
            data,
            [
                "predictions_all",
                "predictions",
                "trajectories_all",
                "trajectories",
            ],
        )

        gt_raw = get_first(
            data,
            [
                "gt_all",
                "gt",
                "ground_truth",
                "gt_trajectory",
                "gt_trajectories",
            ],
        )

        scene_raw = get_first(
            data,
            [
                "scene_names",
                "scenes",
                "scene",
            ],
            required=False,
        )

        window_raw = get_first(
            data,
            [
                "window_idxs",
                "window_idx",
                "windows",
            ],
            required=False,
        )

        predictions = normalize_predictions(pred_raw)
        n_samples = predictions.shape[0]
        gt = normalize_gt(gt_raw, n_samples)

        if gt.shape[0] != n_samples:
            raise ValueError(
                f"Sample count mismatch: predictions N={n_samples}, "
                f"GT N={gt.shape[0]}"
            )

        scenes = as_text_array(scene_raw, n_samples, "scene-")
        windows = as_text_array(window_raw, n_samples, "window-")

        print(f"Predictions shape : {predictions.shape}")
        print(f"GT shape          : {gt.shape}")
        print(f"Samples           : {n_samples}")
        print()

        start = max(0, args.start)
        end = n_samples if args.limit is None else min(
            n_samples, start + max(0, args.limit)
        )

        if start >= n_samples:
            raise ValueError(
                f"--start {start} is outside dataset size {n_samples}"
            )

        generated = 0

        for i in range(start, end):
            out_path = out_dir / f"trajectory_{i:04d}.png"

            plot_one(
                sample_index=i,
                gt=gt[i],
                predictions=predictions[i],
                scene=scenes[i],
                window_idx=windows[i],
                out_path=out_path,
            )

            generated += 1

            if generated == 1 or generated % 25 == 0 or i == end - 1:
                print(
                    f"[{i + 1:>4}/{n_samples}] "
                    f"{out_path.name}  scene={scenes[i]} window={windows[i]}"
                )

        print()
        print("=" * 72)
        print("DONE")
        print("=" * 72)
        print(f"Plots generated : {generated}")
        print(f"Output directory: {out_dir}")
        print()
        print("No TensorRT inference was executed.")


if __name__ == "__main__":
    main()
