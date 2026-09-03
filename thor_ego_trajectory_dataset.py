"""
Minimal nuScenes ego-trajectory dataset loader for DRIVE AGX Thor.

IMPORTANT:
- No PyTorch.
- No torchvision / PIL.
- No full VideoActionModel repository.
- Returns NumPy arrays so they can be passed directly to the existing
  TensorRT / cuda-python inference pipeline.

Required data:
    nuscenes_mini_data_cleaned.pkl
    tokens/**/*.npy
"""

import os
import pickle
from typing import Any, Dict, List

import numpy as np


class HighLevelCommand:
    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2
    FOLLOW_REFERENCE = 3


class ThorEgoTrajectoryDataset:
    """
    Minimal NumPy-only equivalent for the fields needed by trajectory eval.

    Output of dataset[idx]:
        visual_tokens       : np.ndarray, integer
        high_level_command  : np.int64 scalar
        positions           : np.float32 [6, 2]
        window_idx          : np.int64 [14]

    The default sequence configuration follows the PC evaluation:
        8 observation frames
        6 future frames
        subsampling factor = 1
        CAM_FRONT
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
            raise FileNotFoundError(self.pickle_path)
        if not os.path.isdir(self.tokens_rootdir):
            raise FileNotFoundError(self.tokens_rootdir)

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
        return str(getattr(camera, "channel",
                           getattr(camera, "name", camera)))

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
    def _pose_rotation(ego_pose: Any) -> np.ndarray:
        """
        Return quaternion as [w, x, y, z].

        nuScenes / pyquaternion convention is [w, x, y, z].
        """
        if isinstance(ego_pose, dict):
            value = ego_pose["rotation"]
        else:
            value = ego_pose.rotation

        # pyquaternion-like object
        if hasattr(value, "elements"):
            value = value.elements

        q = np.asarray(value, dtype=np.float64).reshape(-1)
        if q.size != 4:
            raise ValueError(
                f"Expected quaternion with 4 elements, got shape {q.shape}"
            )

        # Official nuScenes/pyquaternion convention.
        return q

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate 3D vector v by unit quaternion q=[w,x,y,z].

        Implemented directly with NumPy; no pyquaternion required.
        """
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        w, x, y, z = q
        qvec = np.array([x, y, z], dtype=np.float64)

        # Equivalent to q * [0,v] * q^-1, optimized for vectors.
        return (
            2.0 * np.dot(qvec, v) * qvec
            + (w * w - np.dot(qvec, qvec)) * v
            + 2.0 * w * np.cross(qvec, v)
        )

    @classmethod
    def _relative_position(
        cls,
        current_translation: np.ndarray,
        current_rotation: np.ndarray,
        target_translation: np.ndarray,
    ) -> np.ndarray:
        delta = np.asarray(target_translation) - np.asarray(current_translation)
        return cls._quat_rotate(cls._quat_conjugate(current_rotation), delta)

    def _prepare_data(self):
        records = [
            r for r in self.pickle_data
            if self._camera_name(r) == self.camera_name
        ]

        records.sort(
            key=lambda r: (self._scene_name(r), self._timestamp(r))
        )

        self.data = records

        scene_indices: Dict[str, List[int]] = {}
        for idx, record in enumerate(self.data):
            scene = self._scene_name(record)
            scene_indices.setdefault(scene, []).append(idx)

        self.scene_indices = scene_indices

        step = self.subsampling_factor
        total_frames = self.sequence_length + self.action_length

        # A window consists of:
        #   8 observation frames + 6 future frames
        # with the configured subsampling factor.
        span = (total_frames - 1) * step + 1

        self.window_indices: List[List[int]] = []

        for indices in self.scene_indices.values():
            if len(indices) < span:
                continue

            for start in range(0, len(indices) - span + 1):
                window = indices[start:start + span:step]
                if len(window) == total_frames:
                    self.window_indices.append(window)

    def __len__(self):
        return len(self.window_indices)

    def _record_path(self, record: Dict[str, Any]) -> str:
        path = record.get("path")

        if path is None and isinstance(record.get("camera"), dict):
            camera = record["camera"]
            path = camera.get("path")

        if path is None:
            raise KeyError(
                "Cannot find image path. Expected record['path'] or "
                "record['camera']['path']."
            )

        return os.path.expanduser(str(path))

    def _token_path(self, record: Dict[str, Any]) -> str:
        image_path = self._record_path(record)
        token_rel = os.path.splitext(image_path)[0] + ".npy"

        candidates = []

        if os.path.isabs(token_rel):
            candidates.append(token_rel)

            # Handle PC absolute path such as:
            # /home/patrick/VideoActionModel/data/nuScenes-mini/...
            marker = "/nuScenes-mini/"
            if marker in token_rel:
                relative = token_rel.split(marker, 1)[1]
                candidates.append(
                    os.path.join(self.tokens_rootdir, relative)
                )
        else:
            candidates.append(
                os.path.join(self.tokens_rootdir, token_rel)
            )

        candidates.append(
            os.path.join(
                self.tokens_rootdir,
                os.path.basename(token_rel),
            )
        )

        for path in candidates:
            if os.path.isfile(path):
                return path

        # Last resort: basename recursive search.
        basename = os.path.basename(token_rel)
        matches = []

        for root, _, files in os.walk(self.tokens_rootdir):
            if basename in files:
                matches.append(os.path.join(root, basename))

        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            "\nVisual token file not found.\n"
            f"image path     : {image_path}\n"
            f"token basename : {basename}\n"
            f"tokens root    : {self.tokens_rootdir}\n"
            f"candidates     : {candidates}\n"
            f"recursive hits : {len(matches)}"
        )

    def _load_visual_tokens(self, record: Dict[str, Any]) -> np.ndarray:
        path = self._token_path(record)
        tokens = np.load(path)

        # PC VaVAM visual tokens are integer token IDs.
        if not np.issubdtype(tokens.dtype, np.integer):
            tokens = tokens.astype(np.int64)

        return np.ascontiguousarray(tokens.astype(np.int64, copy=False))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window = self.window_indices[idx]

        obs_indices = window[:self.sequence_length]
        future_indices = window[
            self.sequence_length:
            self.sequence_length + self.action_length
        ]

        ref_record = self.data[obs_indices[-1]]
        ref_pose = ref_record["ego_pose"]

        ref_translation = self._pose_translation(ref_pose)
        ref_rotation = self._pose_rotation(ref_pose)

        visual_tokens = [
            self._load_visual_tokens(self.data[i])
            for i in obs_indices
        ]
        visual_tokens = np.stack(visual_tokens, axis=0)

        # Official command logic:
        # final future position in current ego frame
        final_pose = self.data[future_indices[-1]]["ego_pose"]
        final_translation = self._pose_translation(final_pose)

        relative_final = self._relative_position(
            ref_translation,
            ref_rotation,
            final_translation,
        )

        if relative_final[1] > self.command_distance_threshold:
            command = HighLevelCommand.LEFT
        elif relative_final[1] < -self.command_distance_threshold:
            command = HighLevelCommand.RIGHT
        else:
            command = HighLevelCommand.STRAIGHT

        positions = []
        for i in future_indices:
            pose = self.data[i]["ego_pose"]
            translation = self._pose_translation(pose)

            positions.append(
                self._relative_position(
                    ref_translation,
                    ref_rotation,
                    translation,
                )
            )

        positions = np.asarray(positions, dtype=np.float32)

        return {
            "positions": positions[:, :2],
            "high_level_command": np.int64(command),
            "visual_tokens": visual_tokens,
            "window_idx": np.asarray(window, dtype=np.int64),
        }
