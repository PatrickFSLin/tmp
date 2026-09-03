"""
12j - VaVAM-B ego trajectory evaluation shell for DRIVE AGX Thor.

NO PyTorch.

This file first validates the dataset path/schema. The TensorRT/Euler adapter
is intentionally isolated and will be connected to the already validated 12i
GPU-resident implementation.
"""

import argparse
import os

import numpy as np

from thor_ego_trajectory_dataset import ThorEgoTrajectoryDataset
from min_ade import min_ade


DEFAULT_PREFILL_ENGINE = (
    "../Engines/vavam_joint_kv_prefill_B_v10_fp16.engine"
)
DEFAULT_ACTION_ENGINE = (
    "../Engines/vavam_joint_action_B_fp16.engine"
)
DEFAULT_PICKLE = (
    "../data/nuScenes-mini/nuscenes_mini_data_cleaned.pkl"
)
DEFAULT_TOKENS = (
    "../data/nuScenes-mini/tokens"
)


class ThorVaVAMRunner:
    """
    Adapter to the existing 12i TensorRT + CUDA Euler implementation.

    We intentionally do NOT guess binding names, tensor shapes, timestep
    schedule, or random action initialization here.
    """

    def __init__(self, prefill_engine, action_engine):
        self.prefill_engine = os.path.expanduser(prefill_engine)
        self.action_engine = os.path.expanduser(action_engine)

        if not os.path.isfile(self.prefill_engine):
            raise FileNotFoundError(self.prefill_engine)
        if not os.path.isfile(self.action_engine):
            raise FileNotFoundError(self.action_engine)

    def run_prefill(self, visual_tokens, command):
        raise NotImplementedError(
            "Connect to the validated 12i Prefill runner."
        )

    def run_action(
        self,
        kv_cache,
        action_state,
        command,
        timestep,
    ):
        raise NotImplementedError(
            "Connect to the validated 12i Action runner."
        )

    def euler_step(
        self,
        action_state,
        action_output,
        timestep,
    ):
        raise NotImplementedError(
            "Connect to the validated 12i CUDA Euler kernel."
        )


def print_sample(sample):
    print("\n=== Dataset sample ===")
    print(
        "visual_tokens : "
        f"shape={sample['visual_tokens'].shape}, "
        f"dtype={sample['visual_tokens'].dtype}"
    )
    print(
        "high_level_command : "
        f"shape={sample['high_level_command'].shape}, "
        f"dtype={sample['high_level_command'].dtype}, "
        f"value={sample['high_level_command'].tolist()}"
    )
    print(
        "positions : "
        f"shape={sample['positions'].shape}, "
        f"dtype={sample['positions'].dtype}"
    )
    print(
        "positions[-1] : "
        f"shape={sample['positions'][-1].shape}"
    )
    print(
        "window_idx : "
        f"{sample['window_idx']}"
    )

    print("\nGT trajectory used by PC evaluation:")
    np.set_printoptions(precision=6, suppress=True)
    print(sample["positions"][-1])

    print("\nCommand used by PC evaluation:")
    print(sample["high_level_command"][-1])


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
        "--dataset-only",
        action="store_true",
    )

    args = parser.parse_args()

    print("==============================================")
    print(" VaVAM-B Thor Ego Trajectory Evaluation (12j)")
    print(" NO PyTorch")
    print(" Official pickle schema")
    print("==============================================")

    dataset = ThorEgoTrajectoryDataset(
        pickle_path=args.pickle,
        tokens_rootdir=args.tokens_root,
    )

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[args.index]
    print_sample(sample)

    # PC evaluation uses:
    #
    # commands = batch["high_level_command"][:, -1:]
    # ground_truth = batch["positions"][:, -1]
    #
    # For one sample:
    commands = sample["high_level_command"][-1:].reshape(1, 1)
    ground_truth = sample["positions"][-1:][:, :, :]

    print("\n=== Exact PC-equivalent evaluation inputs ===")
    print(
        f"visual_tokens : {sample['visual_tokens'].shape}"
    )
    print(
        f"commands      : {commands.shape}, "
        f"{commands.tolist()}"
    )
    print(
        f"ground_truth  : {ground_truth.shape}"
    )

    if args.dataset_only:
        print("\nDATASET-ONLY PASS")
        return

    print(
        "\nDataset stage passed. Next step is to connect the existing "
        "12i TensorRT + CUDA Euler runner."
    )


if __name__ == "__main__":
    main()
