"""
NumPy-only minADE implementation.

Matches the semantics of VaVAM's torch min_ade():

    pred_trajectory: [B, M, T, 2]
    gt_trajectory:   [B, T, 2]

For one sampled trajectory:
    minADE == ADE
"""

import numpy as np


def min_ade(
    pred_trajectory,
    gt_trajectory,
    return_idx=False,
    reduction="mean",
):
    pred = np.asarray(pred_trajectory, dtype=np.float32)
    gt = np.asarray(gt_trajectory, dtype=np.float32)

    if pred.ndim != 4:
        raise ValueError(f"pred must be [B,M,T,2], got {pred.shape}")
    if gt.ndim != 3:
        raise ValueError(f"gt must be [B,T,2], got {gt.shape}")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"batch mismatch: {pred.shape} vs {gt.shape}")
    if pred.shape[-1] != 2 or gt.shape[-1] != 2:
        raise ValueError(f"last dimension must be 2: {pred.shape}, {gt.shape}")
    if pred.shape[2] != gt.shape[1]:
        raise ValueError(f"time mismatch: {pred.shape} vs {gt.shape}")

    # [B,T,2] -> [B,1,T,2]
    diff = pred - gt[:, None, :, :]

    # Euclidean distance -> [B,M,T]
    distances = np.linalg.norm(diff, axis=-1)

    # Mean over T -> [B,M]
    ade = distances.mean(axis=-1)

    # Best mode -> [B]
    best_idx = np.argmin(ade, axis=-1)
    best_ade = ade[np.arange(ade.shape[0]), best_idx]

    if reduction == "mean":
        loss = np.mean(best_ade)
    elif reduction == "sum":
        loss = np.sum(best_ade)
    elif reduction == "none":
        loss = best_ade
    else:
        raise ValueError(
            f"Unsupported reduction={reduction}; "
            "expected mean, sum, or none."
        )

    if return_idx:
        return loss, best_idx

    return loss
