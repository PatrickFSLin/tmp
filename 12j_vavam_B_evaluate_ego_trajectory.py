"""
12j - VaVAM-B ego trajectory evaluation on DRIVE AGX Thor.

Purpose
-------
Reproduce the first, minimal PC-side evaluation flow on Thor:

    nuScenes pickle + visual tokens
        |
        v
    ThorEgoTrajectoryDataset
        |
        v
    visual_tokens + command
        |
        v
    TRT Prefill
        |
        v
    TRT Action x 10 + CUDA Euler
        |
        v
    predicted trajectory [1, 1, 6, 2]
        |
        v
    minADE against GT

IMPORTANT
---------
This script assumes the existing Thor TRT runner from the previous
12i / validation work is available.

Because engine I/O names and the exact CUDA Euler wrapper are tied to the
existing 12i implementation, the adapter section below is deliberately
isolated in three functions:

    run_prefill(...)
    run_action(...)
    euler_step(...)

Replace only those adapters with the corresponding functions/classes from
your existing 12i GPU-resident benchmark. The evaluation/dataset/minADE
logic below does not depend on the full VaVAM repository.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from thor_ego_trajectory_dataset import ThorEgoTrajectoryDataset
from min_ade import min_ade


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PREFILL_ENGINE = (
    "../Engines/vavam_joint_kv_prefill_B_v10_fp16.engine"
)
DEFAULT_ACTION_ENGINE = (
    "../Engines/vavam_joint_action_B_fp16.engine"
)

DEFAULT_PICKLE = "../data/nuScenes-mini/nuscenes_mini_data_cleaned.pkl"
DEFAULT_TOKENS = "../data/nuScenes-mini/tokens"


# ---------------------------------------------------------------------------
# TRT / CUDA adapter
# ---------------------------------------------------------------------------

class ThorVaVAMRunner:
    """
    Thin adapter around the existing 12i GPU-resident VaVAM runner.

    DO NOT create a second inference implementation here if your 12i script
    already contains:
        - TensorRT engine loading
        - GPU buffers
        - CUDA Euler kernel
        - the exact input/output tensor bindings

    Instead, copy/import those working pieces into this class.

    The three methods below are the only places that should need adjustment.
    """

    def __init__(self, prefill_engine: str, action_engine: str):
        self.prefill_engine = os.path.expanduser(prefill_engine)
        self.action_engine = os.path.expanduser(action_engine)

        if not os.path.isfile(self.prefill_engine):
            raise FileNotFoundError(self.prefill_engine)

        if not os.path.isfile(self.action_engine):
            raise FileNotFoundError(self.action_engine)

        # ------------------------------------------------------------------
        # TODO: connect this to the existing 12i implementation.
        #
        # Example:
        #   from thor_vavam_runner_12i import ThorVaVAMRunner12i
        #   self.runner = ThorVaVAMRunner12i(
        #       self.prefill_engine,
        #       self.action_engine,
        #   )
        #
        # Then the methods below simply delegate to self.runner.
        # ------------------------------------------------------------------

        self.runner = None

    def run_prefill(self, visual_tokens: torch.Tensor):
        """
        Input:
            visual_tokens: CUDA tensor, typically [1, 8, N]

        Output:
            KV cache / prefill output in the exact format expected by the
            existing Action engine.

        The returned tensors MUST remain on CUDA.
        """
        raise NotImplementedError(
            "Connect run_prefill() to the existing 12i GPU-resident "
            "TRT prefill implementation."
        )

    def run_action(
        self,
        kv_cache,
        action_input: torch.Tensor,
        command: torch.Tensor,
    ):
        """
        Run the Action TensorRT engine.

        All inputs/outputs should remain GPU-resident.
        """
        raise NotImplementedError(
            "Connect run_action() to the existing 12i GPU-resident "
            "TRT action implementation."
        )

    def euler_step(
        self,
        action_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        One CUDA Euler diffusion step.

        This should call the exact CUDA Euler kernel already validated in 12i.
        """
        raise NotImplementedError(
            "Connect euler_step() to the existing 12i CUDA Euler kernel."
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_one(
    runner: ThorVaVAMRunner,
    sample: dict,
    num_diffusion_steps: int = 10,
):
    visual_tokens = sample["visual_tokens"].cuda(non_blocking=True)
    command = sample["high_level_command"].cuda(
        non_blocking=True
    ).reshape(1, 1)

    ground_truth = sample["positions"].cuda(
        non_blocking=True
    ).reshape(1, -1, 2)

    # Ensure copies are complete before starting the measured inference.
    torch.cuda.synchronize()

    t0 = time.perf_counter()

    # Prefill.
    kv_cache = runner.run_prefill(visual_tokens)

    torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    # Initial action state.
    #
    # IMPORTANT:
    # For a simple smoke test, random initialization is sufficient.
    # For PC-vs-Thor numerical A/B comparison, replace this with a fixed
    # saved initial action tensor so both platforms receive exactly the same
    # diffusion initialization.
    action_state = torch.randn(
        (1, 6, 2),
        device="cuda",
        dtype=torch.float32,
    )

    # Diffusion / action loop.
    #
    # The exact ordering and tensor shapes must match the validated 12i
    # implementation. This loop is intentionally an adapter boundary.
    for step in range(num_diffusion_steps):
        # Replace with the exact timestep construction from 12i.
        t = torch.tensor(
            [1.0 - step / num_diffusion_steps],
            device="cuda",
            dtype=torch.float32,
        )

        action_output = runner.run_action(
            kv_cache=kv_cache,
            action_input=action_state,
            command=command,
        )

        action_state = runner.euler_step(
            action_state=action_output,
            t=t,
        )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    # Expected final shape: [1, 6, 2]
    prediction = action_state.reshape(1, 1, 6, 2)

    # Exact VaVAM minADE semantics.
    loss, best_idx = min_ade(
        prediction,
        ground_truth,
        return_idx=True,
        reduction="sum",
    )

    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "minade": float(loss.item()),
        "best_idx": best_idx.cpu().numpy(),
        "prefill_ms": (t_prefill - t0) * 1000.0,
        "e2e_ms": (t_end - t0) * 1000.0,
    }


def print_sample_info(sample: dict):
    visual_tokens = sample["visual_tokens"]
    command = sample["high_level_command"]
    positions = sample["positions"]
    window_idx = sample["window_idx"]

    print("\n=== Dataset sample ===")
    print(f"visual_tokens : shape={tuple(visual_tokens.shape)}, "
          f"dtype={visual_tokens.dtype}")
    print(f"command       : {command.tolist()}")
    print(f"GT positions  : shape={tuple(positions.shape)}, "
          f"dtype={positions.dtype}")
    print(f"window_idx    : {window_idx.tolist()}")

    print("\nGT trajectory:")
    print(positions.numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefill-engine",
        default=DEFAULT_PREFILL_ENGINE,
    )
    parser.add_argument(
        "--action-engine",
        default=DEFAULT_ACTION_ENGINE,
    )
    parser.add_argument(
        "--pickle",
        default=DEFAULT_PICKLE,
    )
    parser.add_argument(
        "--tokens-root",
        default=DEFAULT_TOKENS,
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only verify pickle/token loading and print sample 0.",
    )
    args = parser.parse_args()

    print("==============================================")
    print(" VaVAM-B Thor Ego Trajectory Evaluation (12j)")
    print("==============================================")

    print("\nPaths:")
    print(f"  prefill engine : {os.path.abspath(args.prefill_engine)}")
    print(f"  action engine  : {os.path.abspath(args.action_engine)}")
    print(f"  pickle         : {os.path.abspath(args.pickle)}")
    print(f"  tokens         : {os.path.abspath(args.tokens_root)}")

    dataset = ThorEgoTrajectoryDataset(
        pickle_path=args.pickle,
        tokens_rootdir=args.tokens_root,
    )

    print("\nDataset:")
    print(f"  number of valid sequences = {len(dataset)}")

    sample = dataset[args.index]
    print_sample_info(sample)

    if args.dataset_only:
        print("\nDATASET-ONLY PASS")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("\nCUDA:")
    print(f"  device = {torch.cuda.get_device_name(0)}")
    print(f"  torch  = {torch.__version__}")

    runner = ThorVaVAMRunner(
        prefill_engine=args.prefill_engine,
        action_engine=args.action_engine,
    )

    result = evaluate_one(
        runner=runner,
        sample=sample,
        num_diffusion_steps=args.diffusion_steps,
    )

    print("\n=== Evaluation result ===")
    print(f"minADE       : {result['minade']:.6f} m")
    print(f"best mode    : {result['best_idx']}")
    print(f"prefill      : {result['prefill_ms']:.3f} ms")
    print(f"E2E inference: {result['e2e_ms']:.3f} ms")

    print("\nPrediction:")
    print(result["prediction"].squeeze(0).squeeze(0).cpu().numpy())

    print("\nGround truth:")
    print(result["ground_truth"].squeeze(0).cpu().numpy())


if __name__ == "__main__":
    main()
