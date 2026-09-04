#!/usr/bin/env python3
"""
12k-7 — Action Expert propagation diagnostic

Purpose
-------
The 12k-5/12k-6 diagnostics established that PC PyTorch FP16 and Thor
TensorRT FP16 produce different Prefill K/V caches.  This is the FINAL
numerical diagnostic in the agreed stop-loss plan.

This script asks one question:

    How much of the Prefill difference propagates into the Action Expert
    velocity/action trajectory over the 10 Euler steps?

It compares the previously captured PC Official Forward traces against
Thor Official Forward traces.

Expected files in --reference-dir:

PC:
  pc_official_action_before.npy
  pc_official_velocity.npy
  pc_official_action_after.npy
  pc_official_diffusion_t.npy

Thor:
  thor_official_action_before.npy
  thor_official_velocity.npy
  thor_official_action_after.npy
  thor_official_diffusion_t.npy

The script DOES NOT execute inference and does not modify any engine.
It only compares the already captured traces.

Expected shapes:
  action_before : (10, 1, 6, 2)
  velocity      : (10, 1, 6, 2)
  action_after  : (10, 1, 6, 2)
  diffusion_t   : (10, 1)

It also verifies the Euler identity:

  action_after = action_before + 0.1 * velocity

with the actual captured arrays, separately for PC and Thor.

Outputs:
  metrics_12k_action_expert_propagation.json
  metrics_12k_action_expert_propagation.csv

Interpretation
--------------
This diagnostic can establish whether Prefill K/V divergence is propagated
into Action Expert outputs and how it grows over the 10 steps.

It cannot prove the internal Action Expert operation responsible for the
difference.

After this experiment, stop the numerical root-cause investigation according
to the agreed plan and move to GPU profiling or M=4/M=10.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NUM_STEPS = 10
DT = 0.1


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def stats(pc, thor):
    pc = np.asarray(pc, dtype=np.float32)
    thor = np.asarray(thor, dtype=np.float32)

    if pc.shape != thor.shape:
        raise RuntimeError(
            f"Shape mismatch: PC={pc.shape}, Thor={thor.shape}"
        )

    diff = thor - pc
    abs_diff = np.abs(diff)

    pc_l2 = float(np.linalg.norm(pc.reshape(-1)))
    diff_l2 = float(np.linalg.norm(diff.reshape(-1)))

    return {
        "shape": list(pc.shape),
        "num_elements": int(pc.size),
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_l2": float(
            diff_l2 / pc_l2 if pc_l2 > 0 else 0.0
        ),
        "pc_rms": float(np.sqrt(np.mean(pc * pc))),
        "thor_rms": float(np.sqrt(np.mean(thor * thor))),
        "pc_mean_abs": float(np.mean(np.abs(pc))),
        "thor_mean_abs": float(np.mean(np.abs(thor))),
    }


def euler_check(action_before, velocity, action_after):
    expected = action_before.astype(np.float32) + DT * velocity.astype(np.float32)
    diff = action_after.astype(np.float32) - expected
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def load_required(ref, name):
    path = ref / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def normalize_steps(name, arr):
    """
    Accept the expected 4-D action shape. For diffusion t, accept (10,1)
    and also squeeze harmless singleton dimensions if present.
    """
    arr = np.asarray(arr)

    if name.endswith("diffusion_t.npy"):
        if arr.shape == (NUM_STEPS, 1):
            return arr
        if arr.shape == (NUM_STEPS,):
            return arr[:, None]
        if arr.shape == (NUM_STEPS, 1, 1):
            return arr[:, 0, :]
        raise RuntimeError(
            f"Unexpected diffusion_t shape: {arr.shape}"
        )

    if arr.shape != (NUM_STEPS, 1, 6, 2):
        raise RuntimeError(
            f"Unexpected {name} shape: {arr.shape}; "
            f"expected {(NUM_STEPS, 1, 6, 2)}"
        )
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir",
        default=str(
            Path.home()
            / "vblkdev2"
            / "VaVAM_Thor"
            / "pc_reference"
        ),
    )
    args = parser.parse_args()

    ref = Path(args.reference_dir).expanduser().resolve()

    print("=" * 100)
    print("12k-7 — Action Expert propagation diagnostic")
    print("=" * 100)
    print(f"Reference dir : {ref}")
    print(f"Steps         : {NUM_STEPS}")
    print(f"Euler dt      : {DT}")
    print()
    print("This is a trace comparison only.")
    print("No TensorRT inference is executed.")
    print()

    pc_before = normalize_steps(
        "pc_official_action_before.npy",
        load_required(ref, "pc_official_action_before.npy"),
    )
    pc_vel = normalize_steps(
        "pc_official_velocity.npy",
        load_required(ref, "pc_official_velocity.npy"),
    )
    pc_after = normalize_steps(
        "pc_official_action_after.npy",
        load_required(ref, "pc_official_action_after.npy"),
    )
    pc_t = normalize_steps(
        "pc_official_diffusion_t.npy",
        load_required(ref, "pc_official_diffusion_t.npy"),
    )

    thor_before = normalize_steps(
        "thor_official_action_before.npy",
        load_required(ref, "thor_official_action_before.npy"),
    )
    thor_vel = normalize_steps(
        "thor_official_velocity.npy",
        load_required(ref, "thor_official_velocity.npy"),
    )
    thor_after = normalize_steps(
        "thor_official_action_after.npy",
        load_required(ref, "thor_official_action_after.npy"),
    )
    thor_t = normalize_steps(
        "thor_official_diffusion_t.npy",
        load_required(ref, "thor_official_diffusion_t.npy"),
    )

    print("Loaded traces:")
    print(f"  PC before : {pc_before.shape} {pc_before.dtype}")
    print(f"  PC vel    : {pc_vel.shape} {pc_vel.dtype}")
    print(f"  PC after  : {pc_after.shape} {pc_after.dtype}")
    print(f"  PC t      : {pc_t.shape} {pc_t.dtype}")
    print(f"  Thor before: {thor_before.shape} {thor_before.dtype}")
    print(f"  Thor vel   : {thor_vel.shape} {thor_vel.dtype}")
    print(f"  Thor after : {thor_after.shape} {thor_after.dtype}")
    print(f"  Thor t     : {thor_t.shape} {thor_t.dtype}")

    # ---------------------------------------------------------------
    # Diffusion schedule comparison
    # ---------------------------------------------------------------
    t_stats = stats(
        pc_t.astype(np.float32),
        thor_t.astype(np.float32),
    )

    print()
    print("-" * 100)
    print("Diffusion t comparison")
    print("-" * 100)
    print(f"max_abs : {t_stats['max_abs']:.8e}")
    print(f"mean_abs: {t_stats['mean_abs']:.8e}")
    print(f"RMSE    : {t_stats['rmse']:.8e}")

    # ---------------------------------------------------------------
    # Euler identity checks
    # ---------------------------------------------------------------
    pc_euler = euler_check(pc_before, pc_vel, pc_after)
    thor_euler = euler_check(thor_before, thor_vel, thor_after)

    print()
    print("-" * 100)
    print("Euler consistency")
    print("-" * 100)
    print("PC:")
    print(f"  max_abs : {pc_euler['max_abs']:.8e}")
    print(f"  mean_abs: {pc_euler['mean_abs']:.8e}")
    print(f"  RMSE    : {pc_euler['rmse']:.8e}")
    print("Thor:")
    print(f"  max_abs : {thor_euler['max_abs']:.8e}")
    print(f"  mean_abs: {thor_euler['mean_abs']:.8e}")
    print(f"  RMSE    : {thor_euler['rmse']:.8e}")

    # ---------------------------------------------------------------
    # Step-wise comparison.
    # ---------------------------------------------------------------
    rows = []

    print()
    print("=" * 150)
    print("Step-wise PC ↔ Thor propagation")
    print("=" * 150)
    print(
        "step | t | "
        "before max  before RMSE | "
        "velocity max velocity RMSE | "
        "after max   after RMSE"
    )
    print("-" * 150)

    for step in range(NUM_STEPS):
        b = stats(
            pc_before[step],
            thor_before[step],
        )
        v = stats(
            pc_vel[step],
            thor_vel[step],
        )
        a = stats(
            pc_after[step],
            thor_after[step],
        )

        t_pc = float(np.asarray(pc_t[step]).reshape(-1)[0])
        t_thor = float(np.asarray(thor_t[step]).reshape(-1)[0])
        t_abs = abs(t_thor - t_pc)

        row = {
            "step": step,
            "pc_t": t_pc,
            "thor_t": t_thor,
            "t_abs_diff": t_abs,

            "before_max_abs": b["max_abs"],
            "before_mean_abs": b["mean_abs"],
            "before_rmse": b["rmse"],
            "before_relative_l2": b["relative_l2"],

            "velocity_max_abs": v["max_abs"],
            "velocity_mean_abs": v["mean_abs"],
            "velocity_rmse": v["rmse"],
            "velocity_relative_l2": v["relative_l2"],

            "after_max_abs": a["max_abs"],
            "after_mean_abs": a["mean_abs"],
            "after_rmse": a["rmse"],
            "after_relative_l2": a["relative_l2"],
        }
        rows.append(row)

        print(
            f"{step:4d} | {t_pc:.1f} | "
            f"{b['max_abs']:.3e} {b['rmse']:.3e} | "
            f"{v['max_abs']:.3e} {v['rmse']:.3e} | "
            f"{a['max_abs']:.3e} {a['rmse']:.3e}"
        )

    # ---------------------------------------------------------------
    # Aggregate final and all-step statistics.
    # ---------------------------------------------------------------
    def array_stats(key):
        vals = np.array(
            [r[key] for r in rows],
            dtype=np.float64,
        )
        return {
            "mean": float(vals.mean()),
            "max": float(vals.max()),
            "min": float(vals.min()),
            "argmax_step": int(np.argmax(vals)),
        }

    before_rmse = array_stats("before_rmse")
    vel_rmse = array_stats("velocity_rmse")
    after_rmse = array_stats("after_rmse")

    before_max = array_stats("before_max_abs")
    vel_max = array_stats("velocity_max_abs")
    after_max = array_stats("after_max_abs")

    # ---------------------------------------------------------------
    # Final state comparison
    # ---------------------------------------------------------------
    final_before = stats(
        pc_before[-1],
        thor_before[-1],
    )
    final_vel = stats(
        pc_vel[-1],
        thor_vel[-1],
    )
    final_after = stats(
        pc_after[-1],
        thor_after[-1],
    )

    # ---------------------------------------------------------------
    # Compare the error growth from step 0 to step 9.
    # ---------------------------------------------------------------
    def growth(first, last):
        if first <= 0:
            return None
        return float(last / first)

    propagation = {
        "before_rmse_growth_step0_to_9": growth(
            rows[0]["before_rmse"],
            rows[-1]["before_rmse"],
        ),
        "velocity_rmse_growth_step0_to_9": growth(
            rows[0]["velocity_rmse"],
            rows[-1]["velocity_rmse"],
        ),
        "after_rmse_growth_step0_to_9": growth(
            rows[0]["after_rmse"],
            rows[-1]["after_rmse"],
        ),
        "before_max_growth_step0_to_9": growth(
            rows[0]["before_max_abs"],
            rows[-1]["before_max_abs"],
        ),
        "velocity_max_growth_step0_to_9": growth(
            rows[0]["velocity_max_abs"],
            rows[-1]["velocity_max_abs"],
        ),
        "after_max_growth_step0_to_9": growth(
            rows[0]["after_max_abs"],
            rows[-1]["after_max_abs"],
        ),
    }

    # ---------------------------------------------------------------
    # Simple propagation interpretation.
    # ---------------------------------------------------------------
    initial_vel_rmse = rows[0]["velocity_rmse"]
    final_vel_rmse = rows[-1]["velocity_rmse"]
    initial_after_rmse = rows[0]["after_rmse"]
    final_after_rmse = rows[-1]["after_rmse"]

    if final_vel_rmse < 1e-3:
        velocity_conclusion = (
            "Thor Action Expert velocity remains numerically very close "
            "to PC at the end of the 10-step rollout."
        )
    elif final_vel_rmse < 1e-2:
        velocity_conclusion = (
            "Thor Action Expert velocity shows a small but measurable "
            "PC↔Thor divergence."
        )
    else:
        velocity_conclusion = (
            "Thor Action Expert velocity shows a clear measurable "
            "PC↔Thor divergence."
        )

    if final_after_rmse > initial_after_rmse * 2.0:
        state_conclusion = (
            "Action-state divergence grows substantially over the rollout."
        )
    else:
        state_conclusion = (
            "Action-state divergence does not show a >2x RMSE growth "
            "from step 0 to step 9."
        )

    # ---------------------------------------------------------------
    # Print final summary.
    # ---------------------------------------------------------------
    print()
    print("=" * 110)
    print("Final-step propagation")
    print("=" * 110)

    print("Action before, step 9:")
    print(f"  max_abs : {final_before['max_abs']:.8e}")
    print(f"  mean_abs: {final_before['mean_abs']:.8e}")
    print(f"  RMSE    : {final_before['rmse']:.8e}")
    print(f"  rel L2  : {final_before['relative_l2']:.8e}")

    print()
    print("Velocity, step 9:")
    print(f"  max_abs : {final_vel['max_abs']:.8e}")
    print(f"  mean_abs: {final_vel['mean_abs']:.8e}")
    print(f"  RMSE    : {final_vel['rmse']:.8e}")
    print(f"  rel L2  : {final_vel['relative_l2']:.8e}")

    print()
    print("Action after, step 9:")
    print(f"  max_abs : {final_after['max_abs']:.8e}")
    print(f"  mean_abs: {final_after['mean_abs']:.8e}")
    print(f"  RMSE    : {final_after['rmse']:.8e}")
    print(f"  rel L2  : {final_after['relative_l2']:.8e}")

    print()
    print("=" * 110)
    print("Propagation summary")
    print("=" * 110)

    print(
        f"Before RMSE : "
        f"step0={rows[0]['before_rmse']:.6e} → "
        f"step9={rows[-1]['before_rmse']:.6e} "
        f"({propagation['before_rmse_growth_step0_to_9']})"
    )

    print(
        f"Velocity RMSE: "
        f"step0={rows[0]['velocity_rmse']:.6e} → "
        f"step9={rows[-1]['velocity_rmse']:.6e} "
        f"({propagation['velocity_rmse_growth_step0_to_9']})"
    )

    print(
        f"After RMSE  : "
        f"step0={rows[0]['after_rmse']:.6e} → "
        f"step9={rows[-1]['after_rmse']:.6e} "
        f"({propagation['after_rmse_growth_step0_to_9']})"
    )

    print()
    print(velocity_conclusion)
    print(state_conclusion)

    # ---------------------------------------------------------------
    # Overall conclusion — deliberately conservative.
    # ---------------------------------------------------------------
    print()
    print("=" * 110)
    print("12k numerical investigation stop-loss conclusion")
    print("=" * 110)
    print(
        "This experiment determines whether Prefill numerical divergence "
        "propagates into Action Expert outputs."
    )
    print(
        "It does NOT identify an internal TensorRT operation as the root cause."
    )
    print(
        "After this report, proceed to GPU profiling or M=4/M=10; "
        "do not continue the numerical root-cause investigation."
    )

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------
    csv_path = ref / "metrics_12k_action_expert_propagation.csv"
    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------
    result = {
        "type": "12k-7_action_expert_propagation",
        "reference_dir": str(ref),
        "num_steps": NUM_STEPS,
        "euler_dt": DT,

        "diffusion_t": t_stats,

        "euler_consistency": {
            "pc": pc_euler,
            "thor": thor_euler,
        },

        "per_step": rows,

        "aggregate": {
            "before_rmse": before_rmse,
            "velocity_rmse": vel_rmse,
            "after_rmse": after_rmse,
            "before_max_abs": before_max,
            "velocity_max_abs": vel_max,
            "after_max_abs": after_max,
        },

        "final_step": {
            "before": final_before,
            "velocity": final_vel,
            "after": final_after,
        },

        "propagation": propagation,

        "conclusion": {
            "velocity": velocity_conclusion,
            "action_state": state_conclusion,
            "stop_loss": (
                "Stop numerical root-cause investigation after 12k-7 "
                "and proceed to GPU profiling or M=4/M=10."
            ),
        },
    }

    json_path = ref / "metrics_12k_action_expert_propagation.json"
    with open(json_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
            default=json_default,
        )

    print()
    print("Saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print()
    print("[OK] 12k-7 Action Expert propagation diagnostic completed.")


if __name__ == "__main__":
    main()
