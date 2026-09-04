#!/usr/bin/env python3
"""NumPy-only VaVAM EgoTrajectoryDataset-compatible loader."""

import os
import pickle
import numpy as np


class Command:
    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2
    FOLLOW_REFERENCE = 3


class ThorEgoTrajectoryDataset:
    """Minimal implementation matching the official dataset behavior."""

    COMMAND_DISTANCE_THRESHOLD = 2.0

    def __init__(
        self,
        pickle_path,
        tokens_rootdir,
        camera="CAM_FRONT",
        sequence_length=8,
        action_length=6,
        subsampling_factor=1,
    ):
        self.pickle_path = os.path.expanduser(str(pickle_path))
        self.tokens_rootdir = os.path.expanduser(str(tokens_rootdir))
        self.camera = camera
        self.sequence_length = sequence_length
        self.action_length = action_length
        self.subsampling_factor = subsampling_factor

        with open(self.pickle_path, "rb") as f:
            pickle_data = pickle.load(f)

        # Same ordering as official EgoTrajectoryDataset.
        pickle_data.sort(
            key=lambda x: (
                x["scene"]["name"],
                x[self.camera]["timestamp"],
            )
        )
        self.pickle_data = pickle_data
        self.sequences_indices = self.get_sequence_indices()

    def __len__(self):
        return len(self.sequences_indices)

    def get_sequence_indices(self):
        indices = []

        for start in range(len(self.pickle_data)):
            valid = True
            previous = None
            sequence_indices = []

            max_temporal_index = self.subsampling_factor * (
                self.sequence_length + self.action_length
            )

            for t in range(
                0,
                max_temporal_index,
                self.subsampling_factor,
            ):
                idx = start + t

                if idx >= len(self.pickle_data):
                    valid = False
                    break

                sample = self.pickle_data[idx]

                if (
                    previous is not None
                    and sample["scene"]["name"]
                    != previous["scene"]["name"]
                ):
                    valid = False
                    break

                if t < self.sequence_length * self.subsampling_factor:
                    sequence_indices.append(idx)

                previous = sample

            if valid:
                indices.append(sequence_indices)

        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.asarray([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)

    @staticmethod
    def quaternion_inverse(q):
        q = np.asarray(q, dtype=np.float64)
        return np.asarray(
            [q[0], -q[1], -q[2], -q[3]],
            dtype=np.float64,
        )

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n == 0:
            raise ValueError("Zero-norm quaternion")
        w, x, y, z = q / n
        return np.asarray([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ], dtype=np.float64)

    @classmethod
    def rotate_point(cls, point, quaternion):
        R = cls.quaternion_to_rotation_matrix(quaternion)
        v = np.asarray([point[0], point[1], 0.0], dtype=np.float64)
        return (R @ v)[:2]

    @classmethod
    def pose_to_matrix(cls, translation, rotation):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = cls.quaternion_to_rotation_matrix(rotation)
        matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
        return matrix

    @classmethod
    def get_high_level_command(
        cls,
        translation,
        rotation,
        future_translation,
        future_rotation,
    ):
        cur = cls.pose_to_matrix(translation, rotation)
        nxt = cls.pose_to_matrix(future_translation, future_rotation)
        future_pos_ego = np.linalg.inv(cur) @ nxt[:, -1]

        if future_pos_ego[1] > cls.COMMAND_DISTANCE_THRESHOLD:
            return Command.LEFT
        if future_pos_ego[1] < -cls.COMMAND_DISTANCE_THRESHOLD:
            return Command.RIGHT
        return Command.STRAIGHT

    def sequence_of_positions_to_trajectory(self, positions, rotations):
        positions = np.asarray(positions, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)

        initial_position = positions[0]
        initial_rotation = rotations[0]
        inv_rotation = self.quaternion_inverse(initial_rotation)

        relative_positions = positions[1:] - initial_position
        relative_rotations = rotations[1:].copy()

        for i in range(len(relative_positions)):
            relative_positions[i] = self.rotate_point(
                relative_positions[i],
                inv_rotation,
            )
            relative_rotations[i] = self.quaternion_multiply(
                inv_rotation,
                relative_rotations[i],
            )

        return relative_positions, relative_rotations

    def _token_path(self, file_path):
        relative = file_path.replace(".jpg", ".npy")
        candidate = os.path.join(self.tokens_rootdir, relative)

        if os.path.isfile(candidate):
            return candidate

        # Fallback for a PC absolute path in the pickle.
        if os.path.isabs(relative):
            marker = "/nuScenes-mini/"
            if marker in relative:
                candidate = os.path.join(
                    self.tokens_rootdir,
                    relative.split(marker, 1)[1],
                )
                if os.path.isfile(candidate):
                    return candidate

        raise FileNotFoundError(
            f"Visual token not found:\n"
            f"  file_path={file_path}\n"
            f"  expected={candidate}\n"
            f"  tokens_root={self.tokens_rootdir}"
        )

    def __getitem__(self, index):
        visual_tokens = []
        commands = []
        positions = []
        rotations = []
        scene_names = []
        file_paths = []
        timestamps = []

        first_timestamp = None

        temporal_indices = self.sequences_indices[index][
            :self.sequence_length
        ]

        # Mirrors the official __getitem__: each of the 8 observation
        # frames gets its own 6-step future trajectory.
        for temporal_index in temporal_indices:
            sample = self.pickle_data[temporal_index][self.camera]

            scene_names.append(
                self.pickle_data[temporal_index]["scene"]["name"]
            )
            file_paths.append(sample["file_path"])

            timestamp = sample["timestamp"]
            if first_timestamp is None:
                first_timestamp = timestamp
            timestamps.append(
                (timestamp - first_timestamp) * 1e-6
            )

            token_path = self._token_path(sample["file_path"])
            tokens = np.load(token_path)

            if tokens.shape != (18, 32):
                raise RuntimeError(
                    f"Unexpected token shape {tokens.shape} for "
                    f"{token_path}; expected (18, 32)"
                )

            visual_tokens.append(
                np.ascontiguousarray(tokens.astype(np.int64))
            )

            future_positions = []
            future_rotations = []

            for j in range(
                0,
                (1 + self.action_length) * self.subsampling_factor,
                self.subsampling_factor,
            ):
                future = self.pickle_data[
                    temporal_index + j
                ][self.camera]

                future_positions.append(
                    future["ego_to_world_tran"][:2]
                )
                future_rotations.append(
                    future["ego_to_world_rot"]
                )

            cmd = self.get_high_level_command(
                sample["ego_to_world_tran"],
                sample["ego_to_world_rot"],
                self.pickle_data[
                    temporal_index + self.action_length
                ][self.camera]["ego_to_world_tran"],
                self.pickle_data[
                    temporal_index + self.action_length
                ][self.camera]["ego_to_world_rot"],
            )

            rel_pos, rel_rot = self.sequence_of_positions_to_trajectory(
                future_positions,
                future_rotations,
            )

            commands.append(cmd)
            positions.append(rel_pos.astype(np.float32))
            rotations.append(rel_rot.astype(np.float32))

        return {
            "visual_tokens": np.stack(visual_tokens, axis=0),
            "high_level_command": np.asarray(commands, dtype=np.int64),
            "positions": np.stack(positions, axis=0),
            "rotations": np.stack(rotations, axis=0),
            "timestamps": np.asarray(timestamps, dtype=np.float64),
            "scene_names": scene_names,
            "file_paths": file_paths,
            "window_idx": int(index),
        }
