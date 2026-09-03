"""
Minimal nuScenes EgoTrajectoryDataset for VaVAM evaluation on DRIVE AGX Thor.

This is intentionally standalone and does NOT import the full VideoActionModel
repository. It loads:
  - nuscenes_mini_data_cleaned.pkl
  - precomputed visual-token .npy files

The data-processing semantics are kept aligned with the official
vam/datalib/ego_trajectory_dataset.py for the fields needed by trajectory
evaluation:
  visual_tokens
  high_level_command
  positions
  window_idx
"""

import os
import pickle
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset


class HighLevelCommand(IntEnum):
    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2
    FOLLOW_REFERENCE = 3


class ThorEgoTrajectoryDataset(Dataset):
    """
    Minimal standalone implementation of the official EgoTrajectoryDataset.

    Expected pickle structure:
        pickle_data = list of records
        record["scene"]
        record["camera"]
        record["ego_pose"]
        record["path"]
        ...

    Defaults match the evaluation dataset configuration used by the PC code:
        camera="CAM_FRONT"
        sequence_length=8
        action_length=6
        subsampling_factor=1
    """

    def __init__(
        self,
        pickle_path: str,
        tokens_rootdir: str,
        camera: str = "CAM_FRONT",
        sequence_length: int = 8,
        action_length: int = 6,
        subsampling_factor: int = 1,
        command_distance_threshold: float = 2.0,
    ):
        self.pickle_path = os.path.expanduser(pickle_path)
        self.tokens_rootdir = os.path.expanduser(tokens_rootdir)
        self.camera_name = camera
        self.sequence_length = sequence_length
        self.action_length = action_length
        self.subsampling_factor = subsampling_factor
        self.command_distance_threshold = command_distance_threshold

        if not os.path.isfile(self.pickle_path):
            raise FileNotFoundError(
                f"Pickle file not found: {self.pickle_path}"
            )

        if not os.path.isdir(self.tokens_rootdir):
            raise FileNotFoundError(
                f"Tokens directory not found: {self.tokens_rootdir}"
            )

        with open(self.pickle_path, "rb") as f:
            self.pickle_data = pickle.load(f)

        self._prepare_data()

    @staticmethod
    def _scene_name(record: Dict[str, Any]) -> str:
        scene = record["scene"]
        if isinstance(scene, dict):
            return str(scene.get("name", scene.get("token", "")))
        return str(getattr(scene, "name", scene))

    @staticmethod
    def _camera_name(record: Dict[str, Any]) -> str:
        camera = record["camera"]
        if isinstance(camera, dict):
            return str(camera.get("channel", camera.get("name", "")))
        return str(
            getattr(
                camera,
                "channel",
                getattr(camera, "name", camera),
            )
        )

    @staticmethod
    def _timestamp(record: Dict[str, Any]) -> int:
        camera = record["camera"]
        if isinstance(camera, dict):
            return int(camera["timestamp"])
        return int(camera.timestamp)

    @staticmethod
    def _pose_translation(ego_pose: Any) -> np.ndarray:
        if isinstance(ego_pose, dict):
            value = ego_pose["translation"]
        else:
            value = ego_pose.translation
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _pose_rotation(ego_pose: Any) -> Quaternion:
        if isinstance(ego_pose, dict):
            value = ego_pose["rotation"]
        else:
            value = ego_pose.rotation

        if isinstance(value, Quaternion):
            return value

        return Quaternion(value)

    @staticmethod
    def _get_field(record: Dict[str, Any], key: str) -> Any:
        value = record[key]
        return value

    def _prepare_data(self):
        # Keep the same basic ordering used by the official dataset:
        # scene name, then camera timestamp.
        records = [
            r for r in self.pickle_data
            if self._camera_name(r) == self.camera_name
        ]

        records.sort(
            key=lambda r: (
                self._scene_name(r),
                self._timestamp(r),
            )
        )

        self.data = records

        # Group consecutive records by scene.
        self.scene_indices: Dict[str, List[int]] = {}
        for idx, record in enumerate(self.data):
            scene = self._scene_name(record)
            self.scene_indices.setdefault(scene, []).append(idx)

        # A valid sample contains 8 observation frames and 6 future action
        # frames. With subsampling_factor=1 this is the same sequence layout
        # used in the PC evaluation.
        total = self.sequence_length + self.action_length
        self.window_indices: List[List[int]] = []

        for indices in self.scene_indices.values():
            if len(indices) < total:
                continue

            for start in range(0, len(indices) - total + 1):
                window = indices[start:start + total]
                self.window_indices.append(window)

        if not self.window_indices:
            raise RuntimeError(
                "No valid evaluation sequences were found. "
                f"records={len(self.data)}, "
                f"sequence_length={self.sequence_length}, "
                f"action_length={self.action_length}, "
                f"camera={self.camera_name}"
            )

    def __len__(self) -> int:
        return len(self.window_indices)

    def _token_path(self, record: Dict[str, Any]) -> str:
        """
        Resolve the token path from the same image/path naming convention used
        by the official dataset: image .jpg -> token .npy.

        The pickle may store the path as:
          - record["path"]
          - record["camera"]["path"]
        """
        path = record.get("path", None)

        if path is None and isinstance(record.get("camera"), dict):
            path = record["camera"].get("path", None)

        if path is None:
            raise KeyError(
                "Cannot find image path in pickle record; expected "
                "record['path'] or record['camera']['path']."
            )

        path = os.path.expanduser(str(path))
        token_rel = os.path.splitext(path)[0] + ".npy"

        # If pickle contains an absolute PC path, strip everything before the
        # dataset token directory by using only the basename hierarchy when
        # possible. First try the path exactly as stored under tokens_rootdir.
        if os.path.isabs(token_rel):
            candidates = [
                token_rel,
                os.path.join(
                    self.tokens_rootdir,
                    os.path.basename(token_rel),
                ),
            ]
        else:
            candidates = [
                os.path.join(self.tokens_rootdir, token_rel),
                os.path.join(self.tokens_rootdir, path),
            ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        # Common nuScenes layout: tokens_rootdir mirrors the relative path
        # below the dataset root. Try locating the basename recursively.
        basename = os.path.basename(token_rel)
        matches = []
        for root, _, files in os.walk(self.tokens_rootdir):
            if basename in files:
                matches.append(os.path.join(root, basename))

        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            "Visual token file not found.\n"
            f"record path: {path}\n"
            f"expected token basename: {basename}\n"
            f"tokens_rootdir: {self.tokens_rootdir}\n"
            f"tried: {candidates}"
        )

    def _load_visual_tokens(self, record: Dict[str, Any]) -> torch.Tensor:
        token_path = self._token_path(record)
        tokens = np.load(token_path)

        # Official evaluation feeds integer visual tokens.
        if tokens.dtype != np.int64:
            tokens = tokens.astype(np.int64)

        return torch.from_numpy(tokens)

    def _compute_command(
        self,
        current_translation: np.ndarray,
        current_rotation: Quaternion,
        future_translation: np.ndarray,
    ) -> int:
        # Transform future position into the current ego frame.
        relative = current_rotation.inverse.rotate(
            future_translation - current_translation
        )

        if relative[1] > self.command_distance_threshold:
            return int(HighLevelCommand.LEFT)
        if relative[1] < -self.command_distance_threshold:
            return int(HighLevelCommand.RIGHT)
        return int(HighLevelCommand.STRAIGHT)

    def _relative_position(
        self,
        current_translation: np.ndarray,
        current_rotation: Quaternion,
        target_translation: np.ndarray,
    ) -> np.ndarray:
        return current_rotation.inverse.rotate(
            target_translation - current_translation
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window = self.window_indices[idx]

        # First 8 frames are observations. The last observation frame is the
        # reference ego frame, matching the PC evaluation's batch["positions"]
        # and high-level command usage.
        obs_indices = window[: self.sequence_length]
        future_indices = window[
            self.sequence_length:
            self.sequence_length + self.action_length
        ]

        ref_record = self.data[obs_indices[-1]]
        ref_pose = ref_record["ego_pose"]

        ref_translation = self._pose_translation(ref_pose)
        ref_rotation = self._pose_rotation(ref_pose)

        # Visual tokens for all 8 observation frames.
        visual_tokens = []
        for record_idx in obs_indices:
            visual_tokens.append(
                self._load_visual_tokens(self.data[record_idx])
            )

        visual_tokens = torch.stack(visual_tokens, dim=0).long()

        # High-level command is determined from the future trajectory.
        # Use the final future pose, consistent with the official dataset's
        # command construction.
        future_final = self.data[future_indices[-1]]
        future_final_translation = self._pose_translation(
            future_final["ego_pose"]
        )

        command = self._compute_command(
            ref_translation,
            ref_rotation,
            future_final_translation,
        )

        # Ground-truth future ego positions in the reference ego frame.
        positions = []
        for record_idx in future_indices:
            pose = self.data[record_idx]["ego_pose"]
            translation = self._pose_translation(pose)
            positions.append(
                self._relative_position(
                    ref_translation,
                    ref_rotation,
                    translation,
                )
            )

        positions = torch.from_numpy(
            np.stack(positions, axis=0).astype(np.float32)
        )

        return {
            "positions": positions,
            "high_level_command": torch.tensor(
                command, dtype=torch.long
            ),
            "visual_tokens": visual_tokens,
            "window_idx": torch.tensor(window, dtype=torch.long),
        }


def load_dataset(
    pickle_path: str,
    tokens_rootdir: str,
) -> ThorEgoTrajectoryDataset:
    return ThorEgoTrajectoryDataset(
        pickle_path=pickle_path,
        tokens_rootdir=tokens_rootdir,
    )
