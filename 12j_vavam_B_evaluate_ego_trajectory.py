"""
12j - VaVAM-B ego trajectory evaluation on DRIVE AGX Thor.

NO PyTorch.

This script is the evaluation shell around the already validated 12i
TensorRT + CUDA Euler implementation.

The data/evaluation side is:
    pkl + visual tokens
        -> NumPy arrays
        -> TRT/CUDA inference
        -> prediction [1,1,6,2]
        -> NumPy minADE

IMPORTANT:
The existing 12i runner should be reused for TRT/Euler. Do not create a
second independent TRT implementation here.

Only the adapter class below needs to be connected to 12i.
"""

import argparse
import os
import time

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
    Adapter boundary for the already-working 12i implementation.

    Expected interface:

        kv_cache = run_prefill(visual_tokens, command)

        action_output = run_action(
            kv_cache,
            action_state,
            command,
            timestep,
        )

        action_state = euler_step(
            action_state,
            action_output,
            timestep,
        )

    All inference tensors should remain GPU-resident inside 12i.

    The methods intentionally raise NotImplementedError until we connect
    the exact 12i code, because engine binding names/shapes must not be
    guessed.
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
            "Connect this to the validated 12i GPU-resident "
            "TensorRT Prefill implementation."
        )

    def run_action(
        self,
        kv_cache,
        action_state,
        command,
        timestep,
    ):
        raise NotImplementedError(
            "Connect this to the validated 12i GPU-resident "
            "TensorRT Action implementation."
        )

    def euler_step(
        self,
        action_state,
        action_output,
        timestep,
    ):
        raise NotImplementedError(
            "Connect this to the validated 12i CUDA Euler kernel."
        )


def evaluate_one(
    runner,
    sample,
    num_diffusion_steps=10,
):
    visual_tokens = np.ascontiguousarray(
        sample["visual_tokens"]
    )
    command = np.asarray(
        [[sample["high_level_command"]]],
        dtype=np.int64,
    )
    ground_truth = np.asarray(
        sample["positions"],
        dtype=np.float32,
    )[None, :, :]

    print("\n=== Input to inference ===")
    print(
        f"visual_tokens : shape={visual_tokens.shape}, "
        f"dtype={visual_tokens.dtype}"
    )
    print(
        f"command       : shape={command.shape}, "
        f"dtype={command.dtype}, value={command.tolist()}"
    )
    print(
        f"ground_truth   : shape={ground_truth.shape}, "
        f"dtype={ground_truth.dtype}"
    )

    # Do not generate a random action state here yet.
    # The exact initial action distribution / shape must be copied from
    # the validated 12i implementation.
    #
    # For deterministic PC-vs-Thor comparison, we will eventually feed
    # exactly the same saved initial action tensor to both platforms.

    t0 = time.perf_counter()

    kv_cache = runner.run_prefill(
        visual_tokens=visual_tokens,
        command=command,
    )

    t_prefill = time.perf_counter()

    action_state = None

    for step in range(num_diffusion_steps):
        # The exact timestep tensor and action_state initialization must
        # come from 12i. Do not guess them here.
        timestep = step

        action_output = runner.run_action(
            kv_cache=kv_cache,
            action_state=action_state,
            command=command,
            timestep=timestep,
        )

        action_state = runner.euler_step(
            action_state=action_state,
            action_output=action_output,
            timestep=timestep,
        )

    t_end = time.perf_counter()

    prediction = np.asarray(
        action_state,
        dtype=np.float32,
    )

    if prediction.shape == (1, 6, 2):
        prediction = prediction[:, None, :, :]
    elif prediction.shape != (1, 1, 6, 2):
        raise ValueError(
            "Expected prediction shape [1,6,2] or [1,1,6,2], "
            f"got {prediction.shape}"
        )

    loss, best_idx = min_ade(
        prediction,
        ground_truth,
        return_idx=True,
        reduction="sum",
    )

    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "minade": float(loss),
        "best_idx": best_idx,
        "prefill_ms": (t_prefill - t0) * 1000.0,
        "e2e_ms": (t_end - t0) * 1000.0,
    }


def print_sample(sample):
    print("\n=== Dataset sample ===")
    print(
        "visual_tokens : "
        f"shape={sample['visual_tokens'].shape}, "
        f"dtype={sample['visual_tokens'].dtype}"
    )
    print(
        "command       : "
        f"{sample['high_level_command']}"
    )
    print(
        "GT positions  : "
        f"shape={sample['positions'].shape}, "
        f"dtype={sample['positions'].dtype}"
    )
    print(
        "window_idx    : "
        f"{sample['window_idx'].tolist()}"
    )

    print("\nGT trajectory:")
    np.set_printoptions(precision=6, suppress=True)
    print(sample["positions"])


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
    )

    args = parser.parse_args()

    print("==============================================")
    print(" VaVAM-B Thor Ego Trajectory Evaluation (12j)")
    print(" NO PyTorch")
    print("==============================================")

    dataset = ThorEgoTrajectoryDataset(
        pickle_path=args.pickle,
        tokens_rootdir=args.tokens_root,
    )

    print("\nDataset:")
    print(
        f"  valid evaluation sequences = {len(dataset)}"
    )

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(
            f"index={args.index}, dataset length={len(dataset)}"
        )

    sample = dataset[args.index]
    print_sample(sample)

    if args.dataset_only:
        print("\nDATASET-ONLY PASS")
        return

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
    print(f"minADE        : {result['minade']:.6f} m")
    print(f"best mode     : {result['best_idx']}")
    print(f"prefill       : {result['prefill_ms']:.3f} ms")
    print(f"E2E inference : {result['e2e_ms']:.3f} ms")

    print("\nPrediction:")
    print(result["prediction"][0, 0])

    print("\nGround truth:")
    print(result["ground_truth"][0])


if __name__ == "__main__":
    main()
