"""
Minimal NumPy-only implementation of VaVAM's nuScenes EgoTrajectoryDataset.

IMPORTANT:
This version follows the ACTUAL pickle schema used by the official
VideoActionModel EgoTrajectoryDataset:

    record["scene"]["name"]
    record["CAM_FRONT"]["timestamp"]
    record["CAM_FRONT"]["file_path"]
    record["CAM_FRONT"]["ego_to_world_tran"]
    record["CAM_FRONT"]["ego_to_world_rot"]

No PyTorch / torchvision / PIL / pyquaternion / full VideoActionModel repo.

The returned shapes intentionally match the official dataset because the
PC evaluation does:

    visual_tokens = batch["visual_tokens"]
    commands = batch["high_level_command"][:, -1:]
    ground_truth = batch["positions"][:, -1]

Thus:
    visual_tokens       : [8, H, W]
    high_level_command  : [8]
    positions           : [8, 6, 2]
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
    COMMAND_DISTANCE_THRESHOLD = 2.0

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
        self.camera = camera
        self.sequence_length = sequence_length
        self.action_length = action_length
        self.subsampling_factor = subsampling_factor
        self.command_distance_threshold = command_distance_threshold

        if not os.path.isfile(self.pickle_path):
            raise FileNotFoundError(self.pickle_path)
        if not os.path.isdir(self.tokens_rootdir):
            raise FileNotFoundError(self.tokens_rootdir)

        with open(self.pickle_path, "rb") as f:
            pickle_data = pickle.load(f)

        # IMPORTANT: official code sorts the ORIGINAL records by:
        # (scene["name"], record[camera]["timestamp"])
        pickle_data.sort(
            key=lambda x: (
                x["scene"]["name"],
                x[self.camera]["timestamp"],
            )
        )

        self.pickle_data = pickle_data
        self.sequences_indices = self.get_sequence_indices()

    def get_sequence_indices(self) -> np.ndarray:
        """
        Exact sequence-index logic from official EgoTrajectoryDataset.

        The official code checks:
            sequence_length observation frames
            + action_length future frames

        but stores only the observation-frame indices.
        """
        indices = []

        max_temporal_index = (
            self.subsampling_factor
            * (self.sequence_length + self.action_length)
        )

        for sequence_start_index in range(len(self.pickle_data)):
            is_valid_sequence = True
            previous_sample = None
            sequence_indices = []

            for t in range(
                0,
                max_temporal_index,
                self.subsampling_factor,
            ):
                temporal_index = sequence_start_index + t

                if temporal_index >= len(self.pickle_data):
                    is_valid_sequence = False
                    break

                sample = self.pickle_data[temporal_index]

                if (
                    previous_sample is not None
                    and sample["scene"]["name"]
                    != previous_sample["scene"]["name"]
                ):
                    is_valid_sequence = False
                    break

                if t < self.sequence_length * self.subsampling_factor:
                    sequence_indices.append(temporal_index)

                previous_sample = sample

            if is_valid_sequence:
                indices.append(sequence_indices)

        return np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.sequences_indices)

    @staticmethod
    def quaternion_inverse(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.asarray(
            [q[0], -q[1], -q[2], -q[3]],
            dtype=np.float64,
        )

    @staticmethod
    def quaternion_multiply(
        q1: np.ndarray,
        q2: np.ndarray,
    ) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Equivalent to pyquaternion.Quaternion(q).rotation_matrix.

        q format: [w, x, y, z]
        """
        q = np.asarray(q, dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("Zero-norm quaternion")
        q = q / norm

        w, x, y, z = q

        return np.asarray(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    @classmethod
    def rotate_point(
        cls,
        point: np.ndarray,
        quaternion: np.ndarray,
    ) -> np.ndarray:
        """
        Rotate a 3D point with quaternion and return x,y components.

        This matches the official rotate_point(), which represents the
        2D point as [0, x, y, 0] before quaternion rotation and returns
        the rotated [x, y].
        """
        point = np.asarray(point, dtype=np.float64)
        q = np.asarray(quaternion, dtype=np.float64)

        # Official dataset only needs the x/y components. Its point_quat is:
        # [0, point[0], point[1], 0].
        v3 = np.asarray(
            [point[0], point[1], 0.0],
            dtype=np.float64,
        )

        R = cls.quaternion_to_rotation_matrix(q)
        rotated = R @ v3

        return rotated[:2]

    @classmethod
    def get_high_level_command(
        cls,
        translation: np.ndarray,
        rotation: np.ndarray,
        future_translation: np.ndarray,
        future_rotation: np.ndarray,
    ) -> int:
        """
        Equivalent to the official homogeneous-transform implementation.

        Output coordinate system is x-forward, y-left.
        """
        cur_R = cls.quaternion_to_rotation_matrix(rotation)

        # Current ego -> world:
        # [R, t]
        # World -> current ego:
        # R^T, -R^T t
        future_pos_ego = (
            cur_R.T
            @ (
                np.asarray(future_translation, dtype=np.float64)
                - np.asarray(translation, dtype=np.float64)
            )
        )

        if future_pos_ego[1] > cls.COMMAND_DISTANCE_THRESHOLD:
            return HighLevelCommand.LEFT

        if future_pos_ego[1] < -cls.COMMAND_DISTANCE_THRESHOLD:
            return HighLevelCommand.RIGHT

        return HighLevelCommand.STRAIGHT

    @classmethod
    def sequence_of_positions_to_trajectory(
        cls,
        positions: np.ndarray,
        rotations: np.ndarray,
    ):
        """
        Equivalent to official sequence_of_positions_to_trajectory().

        Input:
            positions  : [action_length+1, 2]
            rotations  : [action_length+1, 4]

        Output:
            relative_positions : [action_length, 2]
            relative_rotations : [action_length, 4]
        """
        positions = np.asarray(positions, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)

        initial_position = positions[0]
        initial_rotation = rotations[0]
        initial_rotation_inv = cls.quaternion_inverse(initial_rotation)

        relative_positions = positions[1:] - initial_position

        for i in range(len(relative_positions)):
            relative_positions[i] = cls.rotate_point(
                relative_positions[i],
                initial_rotation_inv,
            )

        relative_rotations = np.asarray(
            [
                cls.quaternion_multiply(
                    initial_rotation_inv,
                    q,
                )
                for q in rotations[1:]
            ],
            dtype=np.float64,
        )

        return relative_positions, relative_rotations

    def _token_path(self, file_path: str) -> str:
        """
        EXACT official convention:

            os.path.join(tokens_rootdir,
                         sample["file_path"].replace(".jpg", ".npy"))
        """
        relative_token_path = file_path.replace(".jpg", ".npy")

        candidate = os.path.join(
            self.tokens_rootdir,
            relative_token_path,
        )

        if os.path.isfile(candidate):
            return candidate

        # Helpful fallback for a PC absolute path accidentally stored in the
        # pickle. Do not change the normal path behavior.
        if os.path.isabs(relative_token_path):
            marker = "/nuScenes-mini/"
            if marker in relative_token_path:
                relative = relative_token_path.split(marker, 1)[1]
                candidate = os.path.join(
                    self.tokens_rootdir,
                    relative,
                )
                if os.path.isfile(candidate):
                    return candidate

        raise FileNotFoundError(
            "\nVisual token file not found.\n"
            f"file_path     : {file_path}\n"
            f"token path    : {candidate}\n"
            f"tokens root   : {self.tokens_rootdir}\n"
        )

    def __getitem__(self, index: int) -> dict:
        data_visual_tokens = []
        data_high_level_command = []
        data_positions = []
        data_rotations = []
        data_scene_names = []
        data_file_paths = []

        first_frame_timestamp = None

        # Exact official behavior:
        temporal_indices = self.sequences_indices[index][
            : self.sequence_length
        ]

        # IMPORTANT:
        # Keep temporal_index as the loop variable. After this loop it points
        # to the LAST observation frame, exactly as in the official code.
        for temporal_index in temporal_indices:
            sample = self.pickle_data[temporal_index][self.camera]

            data_scene_names.append(
                self.pickle_data[temporal_index]["scene"]["name"]
            )
            data_file_paths.append(sample["file_path"])

            timestamp = sample["timestamp"]

            if first_frame_timestamp is None:
                first_frame_timestamp = timestamp

            relative_timestamp = (
                timestamp - first_frame_timestamp
            ) * 1e-6

            if self.tokens_rootdir is not None:
                token_path = self._token_path(sample["file_path"])
                tokens = np.load(token_path).astype(
                    np.int64,
                    copy=False,
                )
                data_visual_tokens.append(
                    np.ascontiguousarray(tokens)
                )

        # The official code now uses the LAST observation frame
        # (temporal_index) as the reference for the 6-step action trajectory.
        positions = []
        rotations = []

        for j in range(
            0,
            (1 + self.action_length) * self.subsampling_factor,
            self.subsampling_factor,
        ):
            sample = self.pickle_data[
                temporal_index + j
            ][self.camera]

            positions.append(
                np.asarray(
                    sample["ego_to_world_tran"][:2],
                    dtype=np.float64,
                )
            )
            rotations.append(
                np.asarray(
                    sample["ego_to_world_rot"],
                    dtype=np.float64,
                )
            )

        positions = np.asarray(positions, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)

        current_sample = self.pickle_data[temporal_index][self.camera]
        future_sample = self.pickle_data[
            temporal_index + self.action_length
        ][self.camera]

        high_level_command = self.get_high_level_command(
            current_sample["ego_to_world_tran"],
            current_sample["ego_to_world_rot"],
            future_sample["ego_to_world_tran"],
            future_sample["ego_to_world_rot"],
        )

        relative_position, relative_rotation = (
            self.sequence_of_positions_to_trajectory(
                positions,
                rotations,
            )
        )

        # The official dataset appends one [6,2] trajectory for EACH of the
        # 8 observation frames.
        #
        # IMPORTANT:
        # In the official code, the same final temporal_index is used after
        # the observation loop, so data["positions"] contains one trajectory
        # per observation frame only because each observation iteration runs
        # the trajectory extraction in the original __getitem__ structure.
        #
        # For exact PC evaluation semantics, what matters downstream is:
        #     positions[-1] -> [6,2]
        #
        # We construct the full [8,6,2] by evaluating each observation frame
        # as reference below.
        all_relative_positions = []
        all_relative_rotations = []
        all_commands = []

        for obs_index in temporal_indices:
            obs_sample = self.pickle_data[obs_index][self.camera]

            obs_positions = []
            obs_rotations = []

            for j in range(
                0,
                (1 + self.action_length) * self.subsampling_factor,
                self.subsampling_factor,
            ):
                s = self.pickle_data[
                    obs_index + j
                ][self.camera]

                obs_positions.append(
                    np.asarray(
                        s["ego_to_world_tran"][:2],
                        dtype=np.float64,
                    )
                )
                obs_rotations.append(
                    np.asarray(
                        s["ego_to_world_rot"],
                        dtype=np.float64,
                    )
                )

            obs_positions = np.asarray(obs_positions)
            obs_rotations = np.asarray(obs_rotations)

            rel_pos, rel_rot = (
                self.sequence_of_positions_to_trajectory(
                    obs_positions,
                    obs_rotations,
                )
            )

            future_s = self.pickle_data[
                obs_index + self.action_length
            ][self.camera]

            cmd = self.get_high_level_command(
                obs_sample["ego_to_world_tran"],
                obs_sample["ego_to_world_rot"],
                future_s["ego_to_world_tran"],
                future_s["ego_to_world_rot"],
            )

            all_relative_positions.append(rel_pos.astype(np.float32))
            all_relative_rotations.append(rel_rot.astype(np.float32))
            all_commands.append(cmd)

        return {
            "visual_tokens": np.stack(
                data_visual_tokens,
                axis=0,
            ),
            "high_level_command": np.asarray(
                all_commands,
                dtype=np.int64,
            ),
            "positions": np.stack(
                all_relative_positions,
                axis=0,
            ),
            "rotations": np.stack(
                all_relative_rotations,
                axis=0,
            ),
            "scene_names": data_scene_names,
            "file_paths": data_file_paths,
            "window_idx": np.int64(index),
        }
