"""
Minimal minADE implementation matching VaVAM's evaluation semantics.

Input:
    pred_trajectory: [B, M, T, 2]
    gt_trajectory:   [B, T, 2]

Returns:
    reduction="mean": scalar mean over batch
    reduction="sum":  scalar sum over batch
    return_idx=True: also returns best mode index for each batch item
"""

import torch


def min_ade(
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
    return_idx: bool = False,
    reduction: str = "mean",
):
    assert pred_trajectory.ndim == 4
    assert gt_trajectory.ndim == 3
    assert len(pred_trajectory) == len(gt_trajectory)
    assert pred_trajectory.shape[-1] == 2
    assert gt_trajectory.shape[-1] == 2
    assert pred_trajectory.shape[2] == gt_trajectory.shape[1]

    # [B, T, 2] -> [B, 1, T, 2]
    gt_trajectory = gt_trajectory.unsqueeze(1)

    # Euclidean distance for every timestep.
    ade_diff = torch.norm(
        pred_trajectory - gt_trajectory,
        p=2,
        dim=-1,
    )

    # Mean over trajectory timesteps -> [B, M]
    ade_losses = ade_diff.mean(-1)

    # Best sampled trajectory -> [B]
    ade_losses, ade_indices = ade_losses.min(-1)

    if reduction == "mean":
        ade_losses = ade_losses.mean()
    elif reduction == "sum":
        ade_losses = ade_losses.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported reduction={reduction}; "
            "expected mean, sum, or none."
        )

    if return_idx:
        return ade_losses, ade_indices

    return ade_losses
