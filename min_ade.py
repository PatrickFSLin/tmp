#!/usr/bin/env python3
"""NumPy implementation of VaVAM minADE."""

import numpy as np


def min_ade(
    pred_trajectory,
    gt_trajectory,
    return_idx=False,
    reduction="mean",
):
    pred = np.asarray(pred_trajectory, dtype=np.float64)
    gt = np.asarray(gt_trajectory, dtype=np.float64)

    if pred.ndim != 4:
        raise ValueError(f"pred must be [B,M,T,2], got {pred.shape}")
    if gt.ndim != 3:
        raise ValueError(f"gt must be [B,T,2], got {gt.shape}")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError("batch mismatch")
    if pred.shape[2] != gt.shape[1]:
        raise ValueError("trajectory length mismatch")
    if pred.shape[-1] != 2 or gt.shape[-1] != 2:
        raise ValueError("trajectory dimension must be 2")

    distances = np.linalg.norm(
        pred - gt[:, None, :, :],
        axis=-1,
    )
    ade = distances.mean(axis=-1)

    best_idx = np.argmin(ade, axis=-1)
    best_ade = ade[
        np.arange(ade.shape[0]),
        best_idx,
    ]

    if reduction == "mean":
        loss = np.mean(best_ade)
    elif reduction == "sum":
        loss = np.sum(best_ade)
    elif reduction == "none":
        loss = best_ade
    else:
        raise ValueError(f"Unsupported reduction={reduction}")

    if return_idx:
        return loss, best_idx
    return loss
