"""Motion representation converters for common motion formats.

This module implements converters for transforming between motion representations
like GVHMR, HumanML3D, SMPL, MHR, and our axis-angle Motion class.
"""

import logging
from typing import Any, Literal

import numpy as np
import torch

from ..core.motion import Motion
from ..core.skeleton import Skeleton
from ..utils.humanml_utils import (
    HMMLSkeleton,
    cont6d_to_matrix_np,
    matrix_to_cont6d_np,
    qfix_np,
    qinv_np,
    qmul_np,
    qrot_np,
    quaternion_to_cont6d_np,
    rot_yaw_np,
)
from ..utils.quaternion import axis_angle_to_quaternion
from .base_converter import BaseMotionConverter

logger = logging.getLogger(__name__)


def _create_simple_smpl_skeleton() -> Skeleton:
    """Create a simple SMPL-like skeleton for testing purposes."""
    joint_names = (
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    )

    parent_indices = (None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

    # Simple T-pose offsets
    offsets = np.array(
        [
            [0, 0, 0],  # pelvis
            [-0.1, -0.1, 0],
            [0.1, -0.1, 0],  # hips
            [0, 0.1, 0],  # spine1
            [0, -0.4, 0],
            [0, -0.4, 0],  # knees
            [0, 0.2, 0],  # spine2
            [0, -0.4, 0],
            [0, -0.4, 0],  # ankles
            [0, 0.2, 0],  # spine3
            [0, -0.1, 0.1],
            [0, -0.1, 0.1],  # feet
            [0, 0.3, 0],  # neck
            [-0.2, 0.1, 0],
            [0.2, 0.1, 0],  # collars
            [0, 0.2, 0],  # head
            [-0.3, 0, 0],
            [0.3, 0, 0],  # shoulders
            [0, -0.3, 0],
            [0, -0.3, 0],  # elbows
            [0, -0.3, 0],
            [0, -0.3, 0],  # wrists
        ]
    )

    return Skeleton(joint_names, parent_indices, offsets)


class MotionToHumanML3DConverter(BaseMotionConverter):
    """Convert our Motion class to HumanML3D 263-element format.

    **TO BE IMPLEMENTED**
    """

    input_format = "motion"
    output_format = "humanml3d"

    def convert(self, data: Motion) -> np.ndarray:
        raise NotImplementedError("MotionToHumanML3DConverter is not yet implemented. Contributions welcome!")


class SMPLPositionsToHumanML3DConverter(BaseMotionConverter):
    """Convert SMPL 22-joint positions to HumanML3D 263D per-frame features."""

    input_format = "smpl_positions"
    output_format = "humanml3d"

    def __init__(self, feet_threshold: float = 0.002):
        super().__init__()
        self.feet_threshold = float(feet_threshold)
        # HumanML3D canonical kinematic chain and raw offsets (unit directions)
        self._t2m_kinematic_chain = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
        ]
        self._t2m_raw_offsets = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
        self._fid_r = [8, 11]
        self._fid_l = [7, 10]
        self._face_joint_indices = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l

    def convert(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise ValueError("Expected numpy ndarray input with shape (frames, 22, 3)")
        if data.ndim != 3 or data.shape[1:] != (22, 3):
            raise ValueError(f"Expected input shape (frames, 22, 3), got {data.shape}")

        positions = data.copy()

        def foot_detect(pos: np.ndarray, thres: float):
            velfactor = np.array([thres, thres])
            flx = (pos[1:, self._fid_l, 0] - pos[:-1, self._fid_l, 0]) ** 2
            fly = (pos[1:, self._fid_l, 1] - pos[:-1, self._fid_l, 1]) ** 2
            flz = (pos[1:, self._fid_l, 2] - pos[:-1, self._fid_l, 2]) ** 2
            feet_l = ((flx + fly + flz) < velfactor).astype(np.float32)

            frx = (pos[1:, self._fid_r, 0] - pos[:-1, self._fid_r, 0]) ** 2
            fry = (pos[1:, self._fid_r, 1] - pos[:-1, self._fid_r, 1]) ** 2
            frz = (pos[1:, self._fid_r, 2] - pos[:-1, self._fid_r, 2]) ** 2
            feet_r = ((frx + fry + frz) < velfactor).astype(np.float32)
            return feet_l, feet_r

        skel = HMMLSkeleton(self._t2m_raw_offsets, self._t2m_kinematic_chain)
        quat_params = skel.inverse_kinematics_np(positions, self._face_joint_indices)
        quat_params = qfix_np(quat_params)

        cont6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()

        velocity = positions[1:, 0] - positions[:-1, 0]
        velocity = qrot_np(r_rot[1:], velocity)

        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

        positions_local = positions.copy()
        positions_local[..., 0] -= positions_local[:, 0:1, 0]
        positions_local[..., 2] -= positions_local[:, 0:1, 2]
        positions_local = qrot_np(np.repeat(r_rot[:, None, :], positions_local.shape[1], axis=1), positions_local)

        feet_l, feet_r = foot_detect(positions, self.feet_threshold)

        root_y = positions_local[:, 0, 1:2]
        r_velocity_y = np.arcsin(r_velocity[:, 2:3])
        l_velocity_xz = velocity[:, [0, 2]]
        root_data = np.concatenate([r_velocity_y, l_velocity_xz, root_y[:-1]], axis=-1)

        rot_data = cont6d_params[:, 1:].reshape(len(cont6d_params), -1)
        ric_data = positions_local[:, 1:].reshape(len(positions_local), -1)

        local_vel = qrot_np(np.repeat(r_rot[:-1, None, :], positions.shape[1], axis=1), positions[1:] - positions[:-1])
        local_vel = local_vel.reshape(len(local_vel), -1)

        data_out = root_data
        data_out = np.concatenate([data_out, ric_data[:-1]], axis=-1)
        data_out = np.concatenate([data_out, rot_data[:-1]], axis=-1)
        data_out = np.concatenate([data_out, local_vel], axis=-1)
        data_out = np.concatenate([data_out, feet_l, feet_r], axis=-1)

        return data_out.astype(np.float32)


class HumanML3DToMotionConverter(BaseMotionConverter):
    """Convert HumanML3D format to our axis-angle Motion class.

    **TO BE IMPLEMENTED**
    """

    input_format = "humanml3d"
    output_format = "motion"
    _saves_objects = True

    def __init__(self, skeleton: Skeleton | None = None, body_model_path: str | None = None):
        raise NotImplementedError("HumanML3DToMotionConverter is not yet implemented. Contributions welcome!")

    def convert(self, data: np.ndarray) -> Motion:
        raise NotImplementedError("HumanML3DToMotionConverter is not yet implemented. Contributions welcome!")


class HumanML3DTo272Converter(BaseMotionConverter):
    """Convert HumanML3D 263D format to 272D motion representation."""

    input_format = "humanml3d"
    output_format = "format_272"

    def __init__(self, num_joints: int = 22):
        super().__init__()
        self.num_joints = num_joints
        self._t2m_kinematic_chain = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
        ]
        self._t2m_raw_offsets = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

    def convert(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise ValueError("Expected numpy ndarray input")
        if data.ndim != 2:
            raise ValueError(f"Expected 2D input (frames-1, 263), got shape {data.shape}")
        if data.shape[1] != 263:
            raise ValueError(f"Expected 263 features per frame, got {data.shape[1]}")

        T_minus_1 = data.shape[0]
        T = T_minus_1 + 1

        r_velocity_y = data[:, 0:1]
        l_velocity_xz = data[:, 1:3]
        root_y = data[:, 3:4]
        ric_data = data[:, 4 : 4 + 63].reshape(T_minus_1, 21, 3)
        rot_data = data[:, 4 + 63 : 4 + 63 + 126].reshape(T_minus_1, 21, 6)

        yaw_angles = np.zeros(T, dtype=np.float32)
        for t in range(T_minus_1):
            yaw_angles[t + 1] = yaw_angles[t] + r_velocity_y[t, 0]

        root_pos_xz = np.zeros((T, 2), dtype=np.float32)
        for t in range(T_minus_1):
            cos_yaw = np.cos(-yaw_angles[t])
            sin_yaw = np.sin(-yaw_angles[t])
            vel_global = np.array(
                [
                    cos_yaw * l_velocity_xz[t, 0] - sin_yaw * l_velocity_xz[t, 1],
                    sin_yaw * l_velocity_xz[t, 0] + cos_yaw * l_velocity_xz[t, 1],
                ],
                dtype=np.float32,
            )
            root_pos_xz[t + 1] = root_pos_xz[t] + vel_global

        root_pos = np.zeros((T, 3), dtype=np.float32)
        # Initialize frame 0 with first root_y value (important for floor alignment)
        root_pos[0, 1] = root_y[0, 0] if T_minus_1 > 0 else 0.0
        root_pos[1:, [0, 2]] = root_pos_xz[1:]
        root_pos[1:, 1] = root_y[:, 0]

        # Compute global positions correctly:
        # RIC data is root-relative in a facing-forward frame, so we need to:
        # 1. Rotate RIC by the yaw angle (to account for character facing direction)
        # 2. Add the root position
        positions_global = np.zeros((T, 22, 3), dtype=np.float32)

        # Root joint is just the root position
        positions_global[:, 0, :] = root_pos

        # For other joints, transform RIC data to global coordinates
        # NOTE: RIC Y values are ALREADY absolute heights (not relative to root)
        # Only X and Z need root position added (and yaw rotation applied)
        for t in range(1, T):  # Start from frame 1 since frame 0 has no RIC data
            yaw = yaw_angles[t]
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            for joint_idx in range(21):  # 21 joints in RIC
                # Get RIC position (X,Z are root-relative; Y is absolute)
                x, y, z = ric_data[t - 1, joint_idx]  # t-1 because ric_data is T_minus_1 frames
                # Rotate X,Z by yaw
                x_rot = cos_yaw * x - sin_yaw * z
                z_rot = sin_yaw * x + cos_yaw * z
                # Add root XZ position (NOT Y - RIC Y is already absolute)
                positions_global[t, joint_idx + 1, 0] = x_rot + root_pos[t, 0]
                positions_global[t, joint_idx + 1, 1] = y  # Y is already absolute!
                positions_global[t, joint_idx + 1, 2] = z_rot + root_pos[t, 2]

        # For frame 0: use RIC data from frame 0 (index 0 in ric_data) with frame 0's yaw
        if T > 1 and T_minus_1 > 0:
            yaw = yaw_angles[0]  # Frame 0 yaw (usually 0)
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            for joint_idx in range(21):
                # Use first RIC frame for frame 0
                x, y, z = ric_data[0, joint_idx]
                x_rot = cos_yaw * x - sin_yaw * z
                z_rot = sin_yaw * x + cos_yaw * z
                positions_global[0, joint_idx + 1, 0] = x_rot + root_pos[0, 0]
                positions_global[0, joint_idx + 1, 1] = y  # Y is already absolute!
                positions_global[0, joint_idx + 1, 2] = z_rot + root_pos[0, 2]

        rot_matrices = np.zeros((T, 22, 3, 3), dtype=np.float32)
        rot_matrices[0] = np.eye(3, dtype=np.float32)
        for t in range(1, T):
            yaw = yaw_angles[t]
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rot_matrices[t, 0, 0, 0] = cos_yaw
            rot_matrices[t, 0, 0, 2] = sin_yaw
            rot_matrices[t, 0, 1, 1] = 1.0
            rot_matrices[t, 0, 2, 0] = -sin_yaw
            rot_matrices[t, 0, 2, 2] = cos_yaw

        if T_minus_1 > 0:
            rot_matrices[1:, 1:, :, :] = cont6d_to_matrix_np(rot_data)

        velocities_root = np.zeros((T, 3), dtype=np.float32)
        velocities_root[1:] = positions_global[1:, 0, :] - positions_global[:-1, 0, :]

        output = np.zeros((T, 272), dtype=np.float32)
        heading_rot_matrices = rot_yaw_np(yaw_angles)

        velocities_root_xz_no_heading = np.zeros((T, 2), dtype=np.float32)
        for t in range(T):
            vel_rotated = heading_rot_matrices[t] @ velocities_root[t]
            velocities_root_xz_no_heading[t] = vel_rotated[[0, 2]]
        output[:, 0:2] = velocities_root_xz_no_heading

        heading_diff = np.zeros((T, 3, 3), dtype=np.float32)
        heading_diff[0] = np.eye(3, dtype=np.float32)
        for t in range(1, T):
            heading_diff[t] = heading_rot_matrices[t] @ heading_rot_matrices[t - 1].T
        heading_6d = matrix_to_cont6d_np(heading_diff)
        output[:, 2:8] = heading_6d

        positions_no_heading = np.zeros((T, 22, 3), dtype=np.float32)
        for t in range(T):
            # Center only XZ at root; Y stays as absolute world height
            # (matching the original 272D representation convention)
            pos_xz_centered = positions_global[t].copy()
            pos_xz_centered[:, 0] -= positions_global[t, 0, 0]
            pos_xz_centered[:, 2] -= positions_global[t, 0, 2]
            for joint in range(22):
                positions_no_heading[t, joint] = heading_rot_matrices[t] @ pos_xz_centered[joint]
            positions_no_heading[t, :, 0] -= positions_no_heading[t, 0:1, 0]
            positions_no_heading[t, :, 2] -= positions_no_heading[t, 0:1, 2]

        output[:, 8 : 8 + 66] = positions_no_heading.reshape(T, 66)

        velocities_no_heading = np.zeros((T, 22, 3), dtype=np.float32)
        for t in range(1, T):
            vel = positions_no_heading[t] - positions_no_heading[t - 1]
            velocities_no_heading[t] = vel
        output[:, 8 + 66 : 8 + 132] = velocities_no_heading.reshape(T, 66)

        rotations_no_heading = np.zeros((T, 22, 3, 3), dtype=np.float32)
        for t in range(T):
            rotations_no_heading[t, 0] = heading_rot_matrices[t] @ rot_matrices[t, 0]
            rotations_no_heading[t, 1:] = rot_matrices[t, 1:]
        rotations_6d = matrix_to_cont6d_np(rotations_no_heading)
        output[:, 8 + 132 : 8 + 264] = rotations_6d.reshape(T, 132)

        return output.astype(np.float32)


class Format272ToMotionConverter(BaseMotionConverter):
    """Convert 272D motion representation to our axis-angle Motion class.

    Based on the 272-dimensional motion representation from:
    https://github.com/Li-xingXiao/272-dim-Motion-Representation

    Format: [Root_Lin_Vel (2), Root_Ang_Vel (6), Local_Pos (66), Joint_Vel (66), Rotations_6D (132)]
    """

    input_format = "format_272"
    output_format = "motion"
    _saves_objects = True

    def __init__(self, skeleton: Skeleton | None = None, framerate: float = 30.0, body_model_path: str | None = None):
        super().__init__()
        if skeleton is None:
            # Use HumanML3D skeleton by default since 272D format is HumanML3D-based
            skeleton = self._create_humanml3d_skeleton()
        self.skeleton = skeleton
        self.framerate = framerate
        self.body_model_path = body_model_path

    def _create_humanml3d_skeleton(self) -> Skeleton:
        from ..core.skeleton import Skeleton

        return Skeleton.from_humanml3d()

    def _create_skeleton_from_positions(
        self, global_positions: np.ndarray, joint_rotations_aa: np.ndarray | None = None
    ) -> Skeleton:
        """Create a skeleton with bone offsets in parent-local (rest-pose) frame.

        When joint_rotations_aa is provided, bone vectors are inverse-rotated
        by the parent's accumulated global rotation so that the offsets live in
        the parent's local frame — exactly what hierarchical FK expects.  When
        rotations are unavailable, falls back to averaging global-frame bone
        lengths applied along the template rest-pose directions.

        Args:
            global_positions: Global joint positions (num_frames, 22, 3)
            joint_rotations_aa: Joint rotations as axis-angle (num_frames, >=22, 3).
                                If provided, used to compute parent-local offsets.

        Returns:
            Skeleton with proper bone offsets in parent-local frame
        """
        from ..core.skeleton import Skeleton

        template_skeleton = Skeleton.from_humanml3d()
        num_joints = 22
        offsets = np.zeros((num_joints, 3), dtype=np.float32)

        if joint_rotations_aa is not None:
            t2m_raw = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                ],
                dtype=np.float32,
            )
            start_f = min(10, global_positions.shape[0] - 1)
            end_f = min(50, global_positions.shape[0])

            for ji in range(num_joints):
                pi = template_skeleton.parent_indices[ji]
                if pi is None:
                    offsets[ji] = np.array([0.0, 0.0, 0.0])
                else:
                    lengths = [
                        float(np.linalg.norm(global_positions[t, ji] - global_positions[t, pi]))
                        for t in range(start_f, end_f)
                    ]
                    avg_len = float(np.mean(lengths))
                    offsets[ji] = t2m_raw[ji] * avg_len
        else:
            start_frame = min(10, global_positions.shape[0] - 1)
            end_frame = min(50, global_positions.shape[0])
            tmpl_offsets = template_skeleton.offsets
            for ji in range(num_joints):
                pi = template_skeleton.parent_indices[ji]
                if pi is None:
                    offsets[ji] = np.array([0.0, 0.0, 0.0])
                else:
                    lengths = [
                        float(np.linalg.norm(global_positions[t, ji] - global_positions[t, pi]))
                        for t in range(start_frame, end_frame)
                    ]
                    avg_len = float(np.mean(lengths))
                    tmpl_dir = tmpl_offsets[ji]
                    tmpl_norm = float(np.linalg.norm(tmpl_dir))
                    if tmpl_norm > 1e-8:
                        offsets[ji] = (tmpl_dir / tmpl_norm) * avg_len
                    else:
                        offsets[ji] = np.array([0.0, -avg_len, 0.0])

        return Skeleton(
            joint_names=template_skeleton.joint_names, parent_indices=template_skeleton.parent_indices, offsets=offsets
        )

    def _create_smpl_compatible_skeleton(self, humanml_skeleton: Skeleton) -> Skeleton:
        """Create a 24-joint SMPL-compatible skeleton from 22-joint HumanML3D skeleton.

        Adds two hand joints (left_hand, right_hand) as children of the wrists.

        Args:
            humanml_skeleton: 22-joint HumanML3D skeleton

        Returns:
            24-joint SMPL-compatible skeleton
        """
        from ..core.skeleton import Skeleton

        # HumanML3D has 22 joints: 0-21
        # SMPL has 24 joints: 0-21 (same as HumanML3D), then 22-23 (hands)
        # Joint 22 (left_hand) is child of joint 20 (left_wrist)
        # Joint 23 (right_hand) is child of joint 21 (right_wrist)

        joint_names = humanml_skeleton.joint_names + ("left_hand", "right_hand")
        parent_indices = humanml_skeleton.parent_indices + (20, 21)  # Parents are wrists

        # Copy existing offsets and add hand offsets
        offsets = np.zeros((24, 3), dtype=np.float32)
        offsets[:22] = humanml_skeleton.offsets

        # Hand offsets: extend from wrist with small offset
        # Use same direction as wrist bones but shorter
        left_wrist_offset = humanml_skeleton.offsets[20]
        right_wrist_offset = humanml_skeleton.offsets[21]

        # Make hands 60% of wrist length
        offsets[22] = left_wrist_offset * 0.6  # left_hand
        offsets[23] = right_wrist_offset * 0.6  # right_hand

        return Skeleton(joint_names=joint_names, parent_indices=parent_indices, offsets=offsets)

    def convert(self, data: str | np.ndarray) -> Motion:
        if isinstance(data, str):
            data_path = data
            if data_path.endswith(".npy"):
                try:
                    motion_272d = np.load(data_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f"272D data file not found: {data_path}")
            elif data_path.endswith(".npz"):
                try:
                    npz_data = np.load(data_path)
                    if "motion_data" in npz_data:
                        motion_272d = npz_data["motion_data"]
                    elif "data" in npz_data:
                        motion_272d = npz_data["data"]
                    elif len(npz_data.files) == 1:
                        motion_272d = npz_data[npz_data.files[0]]
                    else:
                        raise ValueError(
                            f"Could not find motion data in NPZ file. Available keys: {list(npz_data.files)}"
                        )
                except FileNotFoundError:
                    raise FileNotFoundError(f"272D data file not found: {data_path}")
            else:
                raise ValueError("272D data file must be in .npy or .npz format")
        elif isinstance(data, np.ndarray):
            motion_272d = data
        else:
            raise ValueError(f"Expected str (file path) or numpy array, got {type(data)}")

        if not isinstance(motion_272d, np.ndarray):
            raise ValueError("Expected numpy ndarray input")
        if motion_272d.ndim != 2:
            raise ValueError(f"Expected 2D input (frames, 272), got shape {motion_272d.shape}")
        if motion_272d.shape[1] != 272:
            raise ValueError(f"Expected 272 features per frame, got {motion_272d.shape[1]}")

        num_frames = motion_272d.shape[0]
        root_velocities_xz = motion_272d[:, 0:2]
        heading_6d = motion_272d[:, 2:8]
        local_positions = motion_272d[:, 8:74].reshape(num_frames, 22, 3)
        local_rotations_6d = motion_272d[:, 140:272].reshape(num_frames, 22, 6)

        from ..utils.humanml_utils import cont6d_to_matrix_np, rot_yaw_np

        # Robust heading decode: heading is always a pure Y-axis rotation, so only
        # channels 0 (cosθ) and 2 (sinθ) carry information. The other 4 channels are
        # structural constants (0 or 1) that become noisy after model inference.
        # Feeding the full 6D into Gram-Schmidt lets that noise tilt the rotation
        # out of the XZ plane, causing character tilt that accumulates over frames.
        # Fix: recover yaw from the two live channels and build a clean R_y matrix.
        yaw_deltas = np.arctan2(heading_6d[:, 2], heading_6d[:, 0])
        heading_delta_matrices = rot_yaw_np(yaw_deltas)

        local_rot_matrices = cont6d_to_matrix_np(local_rotations_6d)

        # The encoder stores per-frame heading *deltas* (angular velocity).
        # Accumulate to recover absolute heading at each frame.
        #
        # The encoder also doubles the heading into the root rotation:
        #   stored_root_rot = R_y(heading) @ R_original
        # where R_original itself has yaw = heading, so the stored rotation
        # has yaw = 2 * heading. We recover heading[0] from the frame-0
        # root rotation to restore the original facing direction.
        root_rot_frame0 = local_rot_matrices[0, 0]
        initial_yaw = np.arctan2(root_rot_frame0[0, 2], root_rot_frame0[2, 2]) / 2.0

        def _rot_yaw(yaw):
            cs, sn = np.cos(yaw), np.sin(yaw)
            return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]], dtype=np.float32)

        accumulated_heading = np.zeros((num_frames, 3, 3), dtype=np.float32)
        accumulated_heading[0] = _rot_yaw(initial_yaw)
        for t in range(1, num_frames):
            accumulated_heading[t] = accumulated_heading[t - 1] @ heading_delta_matrices[t]

        # Integrate root velocity to get root positions (XZ only).
        # Y is already embedded in local_positions, so root_positions Y stays 0
        # to avoid double-counting when computing global_pos = heading^T @ local + root.
        root_positions = np.zeros((num_frames, 3), dtype=np.float32)
        for t in range(1, num_frames):
            vel_3d = np.array([root_velocities_xz[t, 0], 0, root_velocities_xz[t, 1]], dtype=np.float32)
            global_vel = accumulated_heading[t - 1].T @ vel_3d
            root_positions[t] = root_positions[t - 1] + global_vel

        # Transform local (heading-normalized) positions back to global frame.
        # The encoder stored pos_local = R_y(heading) @ pos_centered, so
        # pos_centered = R_y(heading)^T @ pos_local = accumulated^T @ pos_local.
        global_positions = np.zeros((num_frames, 22, 3), dtype=np.float32)
        for t in range(num_frames):
            heading_inv = accumulated_heading[t].T
            for joint_idx in range(22):
                global_positions[t, joint_idx] = heading_inv @ local_positions[t, joint_idx] + root_positions[t]

        # Convert rotation matrices to axis-angles FIRST so we can pass them
        # to _create_skeleton_from_positions for parent-local offset computation.
        # Root rotation was encoded as R_y(heading) @ R_original, so
        # R_original = R_y(heading)^T @ stored = accumulated^T @ stored.
        from ..utils.quaternion import rotation_matrix_to_axis_angle

        use_24_joints = self.body_model_path is not None
        num_output_joints = 24 if use_24_joints else 22
        joint_rotations = np.zeros((num_frames, num_output_joints, 3), dtype=np.float32)

        for t in range(num_frames):
            heading_inv = accumulated_heading[t].T
            for joint_idx in range(22):
                rot_matrix = local_rot_matrices[t, joint_idx]
                if joint_idx == 0:
                    global_rot_matrix = heading_inv @ rot_matrix
                else:
                    global_rot_matrix = rot_matrix
                joint_rotations[t, joint_idx] = rotation_matrix_to_axis_angle(global_rot_matrix)

        skeleton_with_bone_lengths = self._create_skeleton_from_positions(
            global_positions, joint_rotations_aa=joint_rotations
        )

        # If padding to 24 joints for GLB export, extend skeleton
        if use_24_joints:
            final_skeleton = self._create_smpl_compatible_skeleton(skeleton_with_bone_lengths)
            precomputed_pos = np.zeros((num_frames, 24, 3), dtype=np.float32)
            precomputed_pos[:, :22] = global_positions
            for t in range(num_frames):
                precomputed_pos[t, 22] = global_positions[t, 20]
                precomputed_pos[t, 23] = global_positions[t, 21]
        else:
            final_skeleton = skeleton_with_bone_lengths
            precomputed_pos = global_positions

        from ..core.forward_kinematics import ForwardKinematics

        class _PrecomputedFK(ForwardKinematics):
            """Return pre-decoded positions instead of running FK from rotations."""

            def __init__(self, positions):
                self._pos = positions

            @property
            def name(self):
                return "Precomputed (272D decoded)"

            def compute_positions(self, joint_rotations, root_translation, skeleton, up_axis="y", identity=None):
                return self._pos

        motion = Motion(
            joint_rotations=joint_rotations,
            root_translation=root_positions,
            skeleton=final_skeleton,
            framerate=self.framerate,
            up_axis="y",
            body_model_path=self.body_model_path,
            forward_kinematics=_PrecomputedFK(precomputed_pos),
            floor_alignment="joint",
            body_model_type="format_272",
        )

        return motion


class SMPLToMotionConverter(BaseMotionConverter):
    """Convert SMPL format to our axis-angle Motion object.

    Accepts an .npz file path or a dict with keys ``pose_body``, ``root_orient``,
    ``trans``, and optionally ``betas`` and ``gender``.

    Parameters
    ----------
    skeleton:
        Custom skeleton.  Defaults to the standard 24-joint SMPL skeleton.
    body_model_path:
        Path to SMPL body model weights (required for mesh-based FK).
    framerate:
        Output framerate in Hz.  Defaults to 30.0 for backward compatibility.
    """

    input_format = "smpl"
    output_format = "motion"
    _saves_objects = True

    def __init__(
        self,
        skeleton: Skeleton | None = None,
        body_model_path: str | None = None,
        framerate: float = 30.0,
    ):
        super().__init__()
        if skeleton is None:
            skeleton = self._create_smpl_skeleton()
        self.skeleton = skeleton
        self.body_model_path = body_model_path
        self.framerate = framerate

    def _create_smpl_skeleton(self) -> Skeleton:
        from ..core.skeleton import Skeleton

        joint_names = [
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hand",
            "right_hand",
        ]
        parent_indices = [None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        offsets = np.zeros((24, 3), dtype=np.float32)
        return Skeleton(joint_names=tuple(joint_names), parent_indices=tuple(parent_indices), offsets=offsets)

    def convert(self, data: str | dict[str, Any]) -> Motion:
        """Convert SMPL data to a Motion object.

        Parameters
        ----------
        data:
            Either a path to an .npz file or a dict containing the SMPL arrays.
            Required keys: ``pose_body`` (T, 63), ``root_orient`` (T, 3), ``trans`` (T, 3).
            Optional keys: ``betas`` (10,), ``gender`` (str).
            If ``betas`` is absent, a neutral mean shape (zeros) is used.
        """
        if isinstance(data, str):
            data_path = data
            if not data_path.endswith(".npz"):
                raise ValueError("SMPL data file must be in .npz format")
            try:
                smpl_data = np.load(data_path, allow_pickle=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"SMPL data file not found: {data_path}")
        elif isinstance(data, dict):
            smpl_data = data
        else:
            raise ValueError(f"Expected str (file path) or dict (data), got {type(data)}")

        required_keys = ["pose_body", "root_orient", "trans"]
        missing_keys = [key for key in required_keys if key not in smpl_data]
        if missing_keys:
            raise ValueError(f"Missing required SMPL data keys: {missing_keys}")

        pose_body = smpl_data["pose_body"]
        root_orient = smpl_data["root_orient"]
        trans = smpl_data["trans"]

        if torch.is_tensor(pose_body):
            pose_body = pose_body.numpy()
        if torch.is_tensor(root_orient):
            root_orient = root_orient.numpy()
        if torch.is_tensor(trans):
            trans = trans.numpy()

        num_frames = pose_body.shape[0]
        n_body_params = pose_body.shape[1]

        # pose_body can be 63 (21 joints from HumanML3D) or 69 (23 joints from full SMPL).
        # Pad to 69 so joint_rotations always has 24 entries (root + 23).
        if n_body_params == 63:
            pose_body = np.concatenate([pose_body, np.zeros((num_frames, 6), dtype=np.float32)], axis=1)
        n_body_joints = pose_body.shape[1] // 3  # 23

        pose_body_reshaped = pose_body.reshape(num_frames, n_body_joints, 3)
        joint_rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
        joint_rotations[:, 0, :] = root_orient
        joint_rotations[:, 1 : n_body_joints + 1, :] = pose_body_reshaped

        root_positions = trans.copy()
        body_model_path = self.body_model_path if self.body_model_path and self.body_model_path.strip() else None

        # betas: use provided value or fall back to neutral mean shape
        if "betas" in smpl_data:
            betas = np.asarray(smpl_data["betas"], dtype=np.float32)
        else:
            betas = np.zeros(10, dtype=np.float32)

        model_data: dict[str, Any] = {
            "betas": betas,
            "body_pose": pose_body,
            "global_orient": root_orient,
            "transl": trans,
        }
        if "gender" in smpl_data:
            model_data["gender"] = smpl_data["gender"]

        motion = Motion(
            joint_rotations=joint_rotations,
            root_translation=root_positions,
            skeleton=self.skeleton,
            framerate=self.framerate,
            up_axis="y",
            body_model_path=body_model_path,
            forward_kinematics="smpl" if body_model_path else "custom",
            floor_alignment="none",
            model_data=model_data,
            body_model_type="smpl",
        )
        return motion


def load_smpl_npz(
    npz_path: str,
    body_model_path: str | None = None,
    framerate: float = 30.0,
) -> Motion:
    """Load an SMPL .npz file and return a Motion object.

    .. deprecated::
        Use :func:`motion_toolbox.io.load_motion` instead::

            from motion_toolbox.io import load_motion
            motion = load_motion("walk.npz", body_model_path=..., framerate=20)

    Parameters
    ----------
    npz_path:
        Path to the .npz file containing ``pose_body``, ``root_orient``,
        ``trans``, and optionally ``betas`` / ``gender``.
    body_model_path:
        Path to SMPL body model weights.  Required for mesh-based rendering via
        :class:`MeshVisualizer`; optional for skeleton-only visualization.
    framerate:
        Output framerate in Hz.  Defaults to 30.0.

    Returns
    -------
    Motion
        A Motion object ready for visualization or further processing.
    """
    import warnings

    warnings.warn(
        "load_smpl_npz is deprecated, use motion_toolbox.io.load_motion instead",
        DeprecationWarning,
        stacklevel=2,
    )
    converter = SMPLToMotionConverter(body_model_path=body_model_path, framerate=framerate)
    return converter.convert(npz_path)


class SMPLTo272Converter(BaseMotionConverter):
    """Convert SMPL format to 272-dimensional motion representation.

    The converter performs forward kinematics on SMPL parameters to obtain
    joint positions, then extracts canonicalized features for the 272-dim format.

    The 272-dim format consists of:
    - [0:2] Root linear velocity XZ (local frame, no heading)
    - [2:8] Root angular velocity (6D rotation representation)
    - [8:74] Local joint positions (22 joints × 3, centered at XZ origin)
    - [74:140] Local joint velocities (22 joints × 3)
    - [140:272] Local joint rotations (22 joints × 6D rotation)
    """

    input_format = "smpl"
    output_format = "format_272"

    def __init__(self, body_model_path: str | None = None, foot_threshold: float = 0.0015):
        """Initialize the SMPL to 272D converter.

        Args:
            body_model_path: Path to SMPL body models directory (required)
            foot_threshold: Threshold for foot contact detection in meters (default: 0.0015)
        """
        super().__init__()
        self.body_model_path = body_model_path
        self.foot_threshold = foot_threshold

        if not body_model_path:
            raise ValueError("body_model_path is required for SMPL to 272D conversion")

    def convert(self, data: str | dict[str, Any]) -> np.ndarray:
        """Convert SMPL parameters to 272-dimensional motion representation.

        Args:
            data: SMPL data as file path (.npz) or dict with keys:
                  'pose_body', 'root_orient', 'trans', optionally 'betas'

        Returns:
            272-dimensional motion features, shape (num_frames, 272)
        """
        # Load SMPL data
        if isinstance(data, str):
            if not data.endswith(".npz"):
                raise ValueError("SMPL data file must be in .npz format")
            try:
                smpl_data = np.load(data)
            except FileNotFoundError:
                raise FileNotFoundError(f"SMPL data file not found: {data}")
        elif isinstance(data, dict):
            smpl_data = data
        else:
            raise ValueError(f"Expected str (file path) or dict (data), got {type(data)}")

        # Extract SMPL parameters
        required_keys = ["pose_body", "root_orient", "trans"]
        missing_keys = [key for key in required_keys if key not in smpl_data]
        if missing_keys:
            raise ValueError(f"Missing required SMPL data keys: {missing_keys}")

        pose_body = smpl_data["pose_body"]
        root_orient = smpl_data["root_orient"]
        trans = smpl_data["trans"]
        betas = smpl_data.get("betas", np.zeros(10))

        # Convert tensors to numpy if needed
        if torch.is_tensor(pose_body):
            pose_body = pose_body.numpy()
        if torch.is_tensor(root_orient):
            root_orient = root_orient.numpy()
        if torch.is_tensor(trans):
            trans = trans.numpy()
        if torch.is_tensor(betas):
            betas = betas.numpy()

        num_frames = pose_body.shape[0]

        # Run SMPL forward kinematics to get joint positions
        import smplx

        # Determine gender
        gender = smpl_data.get("gender", "neutral")
        if isinstance(gender, np.ndarray):
            gender = gender.item()
        if isinstance(gender, bytes):
            gender = gender.decode()
        else:
            gender = str(gender).lower()

        # Create SMPL model and detect if it's actually SMPL-X
        smpl_model = smplx.create(
            model_path=self.body_model_path,
            model_type="smpl",
            gender=gender,
            ext="pkl",
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        )

        # Check if model expects SMPL-X dimensions (207 vs 189 pose features)
        # posedirs shape: (num_pose_features, num_verts * 3)
        num_pose_features = smpl_model.posedirs.shape[0]
        if num_pose_features == 207:  # SMPL-X disguised as SMPL
            model_type = "smplx"
            # Model is SMPL-X but stored in smpl/ folder - just use it as is
            # We'll need to provide extra pose parameters as zeros
        else:  # Standard SMPL (189 pose features)
            model_type = "smpl"

        # Prepare SMPL inputs
        global_orient_torch = torch.from_numpy(root_orient.copy()).float()
        transl_torch = torch.from_numpy(trans.copy()).float()

        # If using SMPL-X model (detected by posedirs), pad body_pose
        if model_type == "smplx":
            # Model expects 69 dims (23 joints × 3) but we have 63 (21 joints × 3)
            # Pad with 6 zeros for the extra 2 hand joints
            body_pose_padded = np.zeros((num_frames, 69), dtype=np.float32)
            body_pose_padded[:, :63] = pose_body  # Body joints (21 × 3)
            # Remaining 6 dims stay zero (2 extra joints × 3)
            body_pose_torch = torch.from_numpy(body_pose_padded).float()
        else:
            body_pose_torch = torch.from_numpy(pose_body.copy()).float()

        # Ensure betas has correct shape
        if betas.ndim == 1:
            betas_torch = torch.from_numpy(betas.copy()).float().unsqueeze(0).repeat(num_frames, 1)
        elif betas.shape[0] == 1:
            betas_torch = torch.from_numpy(betas.copy()).float().repeat(num_frames, 1)
        else:
            betas_torch = torch.from_numpy(betas[:num_frames].copy()).float()

        # Run forward kinematics
        with torch.no_grad():
            smpl_output = smpl_model(
                betas=betas_torch,
                global_orient=global_orient_torch,
                body_pose=body_pose_torch,
                transl=transl_torch,
            )

        # Get joint positions (first 22 joints only)
        position_data = smpl_output.joints.cpu().numpy()[:, :22, :]

        # Get rotation matrices from axis-angles
        # Combine root_orient and body_pose
        pose_body_reshaped = pose_body.reshape(num_frames, 21, 3)
        joint_rotations_aa = np.zeros((num_frames, 22, 3), dtype=np.float32)
        joint_rotations_aa[:, 0, :] = root_orient
        joint_rotations_aa[:, 1:22, :] = pose_body_reshaped

        # Convert axis-angles to rotation matrices using quaternions
        from ..utils.humanml_utils import quaternion_to_matrix_np

        # Convert to quaternions then to matrices
        rotations_quat = axis_angle_to_quaternion(joint_rotations_aa.reshape(-1, 3))
        rotations_quat = rotations_quat.reshape(num_frames, 22, 4)
        rotations_matrix = quaternion_to_matrix_np(rotations_quat)  # (nframes, 22, 3, 3)

        # Center at first frame root position (XZ only, preserve Y)
        # NOTE: Do NOT floor-align here - preserve source orientation
        root_idx = 0
        ori = position_data[0, root_idx].copy()  # First frame root position
        ori[1] = 0  # Don't modify Y - preserve vertical position/orientation
        position_data = position_data - ori

        # Calculate root velocities
        velocities_root = position_data[1:, root_idx, :] - position_data[:-1, root_idx, :]

        # Calculate local positions: all frames on XZ origin
        position_data[:, :, 0] -= position_data[:, 0:1, 0]
        position_data[:, :, 2] -= position_data[:, 0:1, 2]

        # Calculate heading from root rotation
        # Heading is the Y-axis rotation (yaw)
        # For Y-up: extract yaw from rotation matrix
        global_heading = np.arctan2(rotations_matrix[:, root_idx, 0, 2], rotations_matrix[:, root_idx, 2, 2])

        # Helper function for Y-axis rotation matrix
        def rot_yaw(yaw):
            cs = np.cos(yaw)
            sn = np.sin(yaw)
            return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])

        global_heading_rot = np.array([rot_yaw(x) for x in global_heading])
        global_heading_diff = global_heading[1:] - global_heading[:-1]
        global_heading_diff_rot = np.array([rot_yaw(x) for x in global_heading_diff])

        # Calculate positions without heading
        positions_no_heading = np.matmul(
            np.repeat(global_heading_rot[:, None, :, :], 22, axis=1), position_data[..., None]
        ).squeeze(-1)

        # Calculate velocities without heading
        velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

        # Calculate root velocity XZ without heading
        velocities_root_xz_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[
            ..., [0, 2]
        ]

        # Calculate rotations without heading
        # NOTE: Counter-intuitively, we apply the heading rotation (not inverse) to match the 272D format
        # The decoder will apply heading again, so this double-application cancels to identity for the relative rotation
        rotations_matrix[:, 0, ...] = np.matmul(global_heading_rot, rotations_matrix[:, 0, ...])

        # Prepare output: 272-dimensional features
        size_frame = 8 + 22 * 3 + 22 * 3 + 22 * 6
        final_x = np.zeros((num_frames, size_frame), dtype=np.float32)

        # Set first frame root rotation to identity
        final_x[0, 2] = 1.0
        final_x[0, 6] = 1.0

        # Root angular velocity (6D rotation)
        if num_frames > 1:
            from ..utils.humanml_utils import matrix_to_cont6d_np

            final_x[1:, 2:8] = matrix_to_cont6d_np(global_heading_diff_rot)

        # Root linear velocity XZ
        if num_frames > 1:
            final_x[1:, :2] = velocities_root_xz_no_heading

        # Local positions (no heading, centered at XZ origin)
        final_x[:, 8 : 8 + 66] = np.reshape(positions_no_heading, (num_frames, -1))

        # Local velocities
        if num_frames > 1:
            final_x[1:, 8 + 66 : 8 + 132] = np.reshape(velocities_no_heading, (num_frames - 1, -1))

        # Local rotations (6D representation)
        from ..utils.humanml_utils import matrix_to_cont6d_np

        rotations_6d = matrix_to_cont6d_np(rotations_matrix)
        final_x[:, 8 + 132 : 8 + 264] = np.reshape(rotations_6d, (num_frames, -1))

        return final_x


class Format272ToSMPLConverter(BaseMotionConverter):
    """Convert 272-dimensional motion representation to SMPL format.

    This converter chains Format272→Motion→SMPL conversions for efficiency,
    reusing existing well-tested converter implementations.
    """

    input_format = "format_272"
    output_format = "smpl"

    def __init__(self, skeleton: Skeleton | None = None, framerate: float = 30.0, body_model_path: str | None = None):
        """Initialize the 272D to SMPL converter.

        Args:
            skeleton: Skeleton to use for Motion (default: SMPL skeleton)
            framerate: Frames per second (default: 30.0)
            body_model_path: Path to body models directory
        """
        super().__init__()
        self.framerate = framerate
        self.skeleton = skeleton
        self.body_model_path = body_model_path

    def convert(self, data: str | np.ndarray) -> dict[str, Any]:
        """Convert 272D motion representation to SMPL parameters.

        This method chains two existing converters:
        1. Format272ToMotionConverter: 272D → Motion
        2. MotionToSMPLConverter: Motion → SMPL

        Args:
            data: 272D motion data as file path (.npy) or numpy array of shape (frames, 272)

        Returns:
            Dictionary with SMPL parameters:
            - body_pose: (frames, 63) - body joint axis-angles
            - global_orient: (frames, 3) - root orientation
            - transl: (frames, 3) - root translation
            - betas: (frames, 10) - shape parameters
            - gender: string
        """
        # Step 1: Convert 272D to Motion
        format_272_to_motion = Format272ToMotionConverter(
            skeleton=self.skeleton, framerate=self.framerate, body_model_path=self.body_model_path
        )
        motion = format_272_to_motion.convert(data)

        # Step 2: Convert Motion to SMPL
        motion_to_smpl = MotionToSMPLConverter()
        smpl_params = motion_to_smpl.convert(motion)

        return smpl_params


class GVHMRToMotionConverter(BaseMotionConverter):
    """Convert GVHMR format to our axis-angle Motion class."""

    input_format = "gvhmr"
    output_format = "motion"
    _saves_objects = True

    def __init__(
        self,
        max_frames: int | None = None,
        body_model_path: str | None = None,
        floor_alignment: str = "joint",
        hand_fix: Literal["zeros", "stablemotion"] = "stablemotion",
    ):
        super().__init__()
        self.max_frames = max_frames
        self.body_model_path = body_model_path
        self.floor_alignment = floor_alignment
        self.hand_fix = hand_fix

    def convert(self, data: str | dict[str, Any]) -> Motion:
        if isinstance(data, str):
            data_path = data
            if not data_path.endswith(".pt"):
                raise ValueError("GVHMR data file must be in .pt format")
            try:
                gvhmr_data = torch.load(data_path, map_location="cpu")
            except FileNotFoundError:
                raise FileNotFoundError(f"GVHMR data file not found: {data_path}")
        elif isinstance(data, dict):
            gvhmr_data = data
        else:
            raise ValueError(f"Expected str (file path) or dict (data), got {type(data)}")

        if not isinstance(gvhmr_data, dict) or "smpl_params_global" not in gvhmr_data:
            raise ValueError("Expected dict with 'smpl_params_global' key")

        smpl_params = gvhmr_data["smpl_params_global"]
        body_pose = smpl_params["body_pose"]
        global_orient = smpl_params["global_orient"]
        transl = smpl_params["transl"]

        if torch.is_tensor(body_pose):
            body_pose = body_pose.numpy()
        if torch.is_tensor(global_orient):
            global_orient = global_orient.numpy()
        if torch.is_tensor(transl):
            transl = transl.numpy()

        num_frames = body_pose.shape[0]
        if self.max_frames and num_frames > self.max_frames:
            body_pose = body_pose[: self.max_frames]
            global_orient = global_orient[: self.max_frames]
            transl = transl[: self.max_frames]
            num_frames = self.max_frames

        joint_rotations = np.zeros((num_frames, 24, 3))
        joint_rotations[:, 0, :] = global_orient
        body_pose_reshaped = body_pose.reshape(num_frames, 21, 3)
        joint_rotations[:, 1:22, :] = body_pose_reshaped

        if self.hand_fix == "zeros":
            joint_rotations[:, 22:24, :] = 0.0
        elif self.hand_fix == "stablemotion":
            joint_rotations[:, 22:24, :] = 0.0
            ratio2220, ratio2321 = 0.3438, 0.3345
            joint_rotations[..., 22, :] = joint_rotations[..., 20, :] + ratio2220 * (
                joint_rotations[..., 20, :] - joint_rotations[..., 18, :]
            )
            joint_rotations[..., 23, :] = joint_rotations[..., 21, :] + ratio2321 * (
                joint_rotations[..., 21, :] - joint_rotations[..., 19, :]
            )
        else:
            raise ValueError(f"Invalid hand fix method: {self.hand_fix}")

        from ..core.skeleton import Skeleton

        skeleton = Skeleton.from_smpl()

        # Extract SMPL parameters for model_data (needed for MHR conversion)
        model_data = {
            "body_pose": body_pose,  # Already numpy
            "global_orient": global_orient,  # Already numpy
            "transl": transl,  # Already numpy
        }
        if "betas" in smpl_params:
            betas = smpl_params["betas"]
            if torch.is_tensor(betas):
                betas = betas.numpy()
            model_data["betas"] = betas
        else:
            # Default betas
            model_data["betas"] = np.zeros((num_frames, 10))

        # Add gender if available
        model_data["gender"] = smpl_params.get("gender", "neutral")

        motion = Motion(
            joint_rotations=joint_rotations,
            root_translation=transl,
            skeleton=skeleton,
            framerate=30.0,
            body_model_path=self.body_model_path,
            floor_alignment=self.floor_alignment,
            model_data=model_data,
            body_model_type="smpl",
        )

        return motion


# ---------------------------------------------------------------------------
# Rotation-convention helpers for MotionToSMPLConverter
# ---------------------------------------------------------------------------

# body_model_type values whose joint_rotations are already in SMPL's T-pose
# rest-pose convention and can be passed directly to the SMPL model.
_SMPL_ROTATION_CONVENTIONS: frozenset[str] = frozenset({"smpl"})


def _uses_smpl_rotations(motion: "Motion") -> bool:
    """Return True if motion.joint_rotations are in SMPL-convention."""
    return motion.body_model_type in _SMPL_ROTATION_CONVENTIONS


def _find_default_smpl_path() -> str:
    """Return the default SMPL body-model path, or raise RuntimeError."""
    from pathlib import Path as _Path

    default = _Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl"
    if default.exists():
        return str(default)
    raise RuntimeError(
        "body_model_path is not set on the Motion object and the default SMPL "
        "model directory was not found. Pass body_model_path to your motion "
        "converter (e.g. Format272ToMotionConverter(body_model_path=...))."
    )


class MotionToSMPLConverter(BaseMotionConverter):
    """Convert axis-angle Motion class to SMPL format.

    Two strategies are used depending on the rotation convention stored in
    ``motion.body_model_type``:

    * **Fast path** (``body_model_type in {'smpl'}``): ``joint_rotations`` are
      already in SMPL T-pose IK convention and are passed directly to the SMPL
      model — identical to the previous behaviour for SMPL/GVHMR/AMASS sources.

    * **Slow path** (all other types, e.g. ``'format_272'``, ``None``):
      ``joint_rotations`` are in a different IK convention (e.g. HumanML3D
      unit-direction rest pose). The correct joint *positions* (always
      accurate thanks to ``PrecomputedFK``) are used as targets for a short
      SMPLify optimisation that produces valid SMPL parameters.
    """

    input_format = "motion"
    output_format = "smpl"

    def convert(self, data: Motion) -> dict[str, Any]:
        motion = data

        if _uses_smpl_rotations(motion):
            # ------------------------------------------------------------------
            # Fast path: rotations are SMPL-convention — use directly.
            # ------------------------------------------------------------------
            if motion.skeleton.num_joints < 24:
                raise ValueError(f"Motion skeleton has {motion.skeleton.num_joints} joints, need at least 24 for SMPL")

            num_frames = motion.num_frames
            global_orient = motion.joint_rotations[:, 0, :]
            body_pose = motion.joint_rotations[:, 1:24, :].reshape(num_frames, 69)
            transl = motion.root_translation

            betas = np.zeros((num_frames, 10))
            gender = "neutral"

            if "betas" in motion.model_data:
                b = motion.model_data["betas"]
                if len(b.shape) == 1:
                    betas = np.tile(b, (num_frames, 1))
                elif b.shape[0] == 1:
                    betas = np.repeat(b, num_frames, axis=0)
                else:
                    betas = b[:num_frames]
            elif motion.identity is not None:
                smpl_params_dict = motion.identity.get_smpl_params()
                betas = np.tile(smpl_params_dict["betas"], (num_frames, 1))
                gender = smpl_params_dict["gender"]

            if "gender" in motion.model_data:
                gender = motion.model_data["gender"]

        else:
            # ------------------------------------------------------------------
            # Slow path: rotations are not SMPL-convention.
            # Fit SMPL parameters to the precomputed joint positions via
            # a short SMPLify optimisation.  Positions are always correct
            # (produced by PrecomputedFK or equivalent), so this is safe.
            # ------------------------------------------------------------------
            from .humanml3d_to_smpl import HumanML3DToSMPLConverter

            smpl_body_path = motion.body_model_path or _find_default_smpl_path()
            fitter = HumanML3DToSMPLConverter(
                smpl_model_path=smpl_body_path,
                num_smplify_iters=20,
                use_batched=True,
                use_lbfgs=True,
            )
            positions = motion.positions[:, :22]  # (T, 22, 3) — always accurate
            result = fitter.convert_with_positions(positions, recompute_joints=True)

            num_frames = motion.num_frames
            global_orient = result["global_orient"]
            body_pose = result["body_pose"]
            transl = result["transl"]
            betas = np.tile(result["betas"], (num_frames, 1))
            gender = "neutral"

        return {
            "body_pose": torch.from_numpy(body_pose.copy()).float(),
            "global_orient": torch.from_numpy(global_orient.copy()).float(),
            "transl": torch.from_numpy(transl.copy()).float(),
            "betas": torch.from_numpy(betas.copy()).float(),
            "gender": gender,
        }


class MotionToGVHMRConverter(BaseMotionConverter):
    """Convert axis-angle Motion class to GVHMR format."""

    input_format = "motion"
    output_format = "gvhmr"

    def convert(self, data: Motion) -> dict[str, Any]:
        motion = data
        if motion.skeleton.num_joints < 22:
            raise ValueError(f"Motion skeleton has {motion.skeleton.num_joints} joints, need at least 22 for GVHMR")

        joint_rotations = motion.joint_rotations
        root_translation = motion.root_translation
        num_frames = motion.num_frames

        global_orient = joint_rotations[:, 0, :]
        body_pose_3d = joint_rotations[:, 1:22, :]
        body_pose = body_pose_3d.reshape(num_frames, 63)

        betas = np.zeros((num_frames, 10))

        if "betas" in motion.model_data:
            b = motion.model_data["betas"]
            if len(b.shape) == 1:
                betas = np.tile(b, (num_frames, 1))
            elif b.shape[0] == 1:
                betas = np.repeat(b, num_frames, axis=0)
            else:
                betas = b[:num_frames]
        elif motion.identity is not None:
            smpl_params_dict = motion.identity.get_smpl_params()
            betas = np.tile(smpl_params_dict["betas"], (num_frames, 1))

        gvhmr_params = {
            "smpl_params_global": {
                "body_pose": torch.from_numpy(body_pose.copy()).float(),
                "global_orient": torch.from_numpy(global_orient.copy()).float(),
                "transl": torch.from_numpy(root_translation.copy()).float(),
                "betas": torch.from_numpy(betas.copy()).float(),
            }
        }

        return gvhmr_params


class MHRToMotionConverter(BaseMotionConverter):
    """Convert MHR format to our Motion class.

    MHR parameters are stored in `model_data` of the Motion object.
    Since we don't have a direct mapping from MHR 204 params to a standard skeleton yet,
    we store the raw parameters and create a dummy skeleton/motion structure.
    """

    input_format = "mhr"
    output_format = "motion"

    def convert(self, data: dict[str, Any]) -> Motion:
        """Convert MHR dictionary to Motion.

        Args:
            data: Dict with keys 'identity_coeffs', 'model_parameters', 'face_expr_coeffs'.
        """
        # Extract parameters
        model_parameters = data.get("model_parameters")
        if model_parameters is None:
            raise ValueError("model_parameters required for MHR conversion")

        # Convert to numpy
        if torch.is_tensor(model_parameters):
            model_parameters = model_parameters.cpu().numpy()

        num_frames = model_parameters.shape[0]

        # MHR has 204 params. We don't have a mapping to standard skeleton rotation/translation yet.
        # We create a placeholder Motion with 0 rotations but store everything in model_data.

        # Use MHR body model class to get skeleton if possible, otherwise generic.
        from ..core.body_models.mhr import MHRBodyModel

        # Dummy path/lod, we just want skeleton
        skeleton = MHRBodyModel(model_path="", lod=1).get_skeleton()

        joint_rotations = np.zeros((num_frames, skeleton.num_joints, 3))
        root_translation = np.zeros((num_frames, 3))  # MHR root trans is usually part of model_parameters?

        # Store MHR params in model_data
        model_data = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in data.items()}

        return Motion(
            joint_rotations=joint_rotations,
            root_translation=root_translation,
            skeleton=skeleton,
            framerate=30.0,
            body_model_type="mhr",
            model_data=model_data,
            forward_kinematics="custom",  # Use generic FK to avoid SMPL requirement
            floor_alignment="none",  # Skip alignment as we don't have mesh/reliable joints
        )


class SMPLToR15Converter(BaseMotionConverter):
    """Convert SMPL motion to Roblox R15 format.

    This converter takes SMPL motion data (24 joints) and converts it to
    R15 format (16 joints) with proper joint mapping and rotation corrections.

    The R15 skeleton is a simplified version of SMPL where:
    - Multiple spine joints are combined into UpperTorso
    - Collar and shoulder joints are combined
    - The coordinate system and rest pose differences are handled

    Scale factor between SMPL and R15: 0.0357
    """

    input_format = "motion"
    output_format = "r15_motion"

    # Scale factor between SMPL (in meters) and R15 (in Roblox units)
    SCALE_FACTOR = 0.0357

    # SMPL joint indices
    SMPL_PELVIS = 0
    SMPL_L_HIP = 1
    SMPL_R_HIP = 2
    SMPL_SPINE1 = 3
    SMPL_L_KNEE = 4
    SMPL_R_KNEE = 5
    SMPL_SPINE2 = 6
    SMPL_L_ANKLE = 7
    SMPL_R_ANKLE = 8
    SMPL_SPINE3 = 9
    SMPL_L_FOOT = 10
    SMPL_R_FOOT = 11
    SMPL_NECK = 12
    SMPL_L_COLLAR = 13
    SMPL_R_COLLAR = 14
    SMPL_HEAD = 15
    SMPL_L_SHOULDER = 16
    SMPL_R_SHOULDER = 17
    SMPL_L_ELBOW = 18
    SMPL_R_ELBOW = 19
    SMPL_L_WRIST = 20
    SMPL_R_WRIST = 21
    SMPL_L_HAND = 22
    SMPL_R_HAND = 23

    # R15 joint indices
    R15_ROOT = 0
    R15_LOWER_TORSO = 1
    R15_UPPER_TORSO = 2
    R15_HEAD = 3
    R15_LEFT_UPPER_ARM = 4
    R15_LEFT_LOWER_ARM = 5
    R15_LEFT_HAND = 6
    R15_RIGHT_UPPER_ARM = 7
    R15_RIGHT_LOWER_ARM = 8
    R15_RIGHT_HAND = 9
    R15_LEFT_UPPER_LEG = 10
    R15_LEFT_LOWER_LEG = 11
    R15_LEFT_FOOT = 12
    R15_RIGHT_UPPER_LEG = 13
    R15_RIGHT_LOWER_LEG = 14
    R15_RIGHT_FOOT = 15

    def __init__(self, apply_scale: bool = True):
        """Initialize converter.

        Args:
            apply_scale: Whether to apply the SMPL to R15 scale factor to translation
        """
        super().__init__()
        self.apply_scale = apply_scale

    def _axis_angle_to_quat(self, axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle to quaternion (w, x, y, z format).

        Args:
            axis_angle: Axis-angle rotation, shape (..., 3)

        Returns:
            Quaternion in (w, x, y, z) format, shape (..., 4)
        """
        from scipy.spatial.transform import Rotation as R

        # Make a writable copy if needed
        aa = np.asarray(axis_angle).copy()
        rot = R.from_rotvec(aa.reshape(-1, 3))
        q_scipy = rot.as_quat()  # (x, y, z, w) format
        # Convert to (w, x, y, z) format
        q = np.column_stack([q_scipy[:, 3], q_scipy[:, 0], q_scipy[:, 1], q_scipy[:, 2]])
        return q.reshape(axis_angle.shape[:-1] + (4,))

    def _quat_to_axis_angle(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z format) to axis-angle.

        Args:
            quat: Quaternion in (w, x, y, z) format, shape (..., 4)

        Returns:
            Axis-angle rotation, shape (..., 3)
        """
        from scipy.spatial.transform import Rotation as R

        # Convert from (w, x, y, z) to (x, y, z, w) format
        q_scipy = np.column_stack(
            [quat.reshape(-1, 4)[:, 1], quat.reshape(-1, 4)[:, 2], quat.reshape(-1, 4)[:, 3], quat.reshape(-1, 4)[:, 0]]
        )
        rot = R.from_quat(q_scipy)
        axis_angle = rot.as_rotvec()
        return axis_angle.reshape(quat.shape[:-1] + (3,))

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions in (w, x, y, z) format.

        Args:
            q1: First quaternion, shape (..., 4)
            q2: Second quaternion, shape (..., 4)

        Returns:
            Product quaternion, shape (..., 4)
        """
        from scipy.spatial.transform import Rotation as R

        # Convert from (w, x, y, z) to (x, y, z, w) format
        q1_scipy = np.column_stack(
            [q1.reshape(-1, 4)[:, 1], q1.reshape(-1, 4)[:, 2], q1.reshape(-1, 4)[:, 3], q1.reshape(-1, 4)[:, 0]]
        )
        q2_scipy = np.column_stack(
            [q2.reshape(-1, 4)[:, 1], q2.reshape(-1, 4)[:, 2], q2.reshape(-1, 4)[:, 3], q2.reshape(-1, 4)[:, 0]]
        )
        rot1 = R.from_quat(q1_scipy)
        rot2 = R.from_quat(q2_scipy)
        result = (rot1 * rot2).as_quat()
        # Convert back to (w, x, y, z)
        q_out = np.column_stack([result[:, 3], result[:, 0], result[:, 1], result[:, 2]])
        return q_out.reshape(q1.shape)

    def _euler_to_quat(self, angles: tuple[float, float, float], order: str = "xyz") -> np.ndarray:
        """Convert Euler angles (in degrees) to quaternion.

        Args:
            angles: (x, y, z) angles in degrees
            order: Euler angle order

        Returns:
            Quaternion in (w, x, y, z) format, shape (4,)
        """
        from scipy.spatial.transform import Rotation as R

        rot = R.from_euler(order, angles, degrees=True)
        q_scipy = rot.as_quat()
        return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

    def convert(self, data: Motion) -> dict[str, Any]:
        """Convert SMPL Motion to R15 format.

        Args:
            data: Motion object with SMPL skeleton (24 joints)

        Returns:
            Dictionary containing R15 motion parameters:
            - 'joint_rotations': R15 joint rotations as axis-angles, shape (frames, 16, 3)
            - 'root_translation': Root translation, shape (frames, 3)
            - 'skeleton': R15 Skeleton instance
            - 'framerate': Frame rate
        """
        motion = data

        # Validate input
        if motion.skeleton.num_joints < 22:
            raise ValueError(
                f"Motion must have at least 22 joints for SMPL conversion, got {motion.skeleton.num_joints}"
            )

        num_frames = motion.num_frames
        smpl_rotations = motion.joint_rotations  # (frames, joints, 3) axis-angle
        smpl_translation = motion.root_translation  # (frames, 3)

        # Initialize R15 rotations (16 joints)
        r15_rotations = np.zeros((num_frames, 16, 3), dtype=np.float32)

        # Convert each frame
        for frame_idx in range(num_frames):
            frame_rotations = smpl_rotations[frame_idx]  # (joints, 3)

            # Convert all SMPL rotations to quaternions for this frame
            smpl_quats = self._axis_angle_to_quat(frame_rotations)  # (joints, 4)

            # === R15 Joint Mapping ===

            # R15_ROOT (0) - No rotation, just identity
            r15_rotations[frame_idx, self.R15_ROOT] = np.zeros(3)

            # R15_LOWER_TORSO (1) <- SMPL Pelvis (0)
            r15_rotations[frame_idx, self.R15_LOWER_TORSO] = frame_rotations[self.SMPL_PELVIS]

            # R15_UPPER_TORSO (2) <- SMPL Spine1 @ Spine2 @ Spine3
            spine1_q = smpl_quats[self.SMPL_SPINE1]
            spine2_q = smpl_quats[self.SMPL_SPINE2]
            spine3_q = smpl_quats[self.SMPL_SPINE3]
            upper_torso_q = self._quat_multiply(
                self._quat_multiply(spine1_q.reshape(1, 4), spine2_q.reshape(1, 4)), spine3_q.reshape(1, 4)
            )
            r15_rotations[frame_idx, self.R15_UPPER_TORSO] = self._quat_to_axis_angle(upper_torso_q)[0]

            # R15_HEAD (3) <- SMPL Neck @ Head
            neck_q = smpl_quats[self.SMPL_NECK]
            head_q = smpl_quats[self.SMPL_HEAD]
            head_combined_q = self._quat_multiply(neck_q.reshape(1, 4), head_q.reshape(1, 4))
            r15_rotations[frame_idx, self.R15_HEAD] = self._quat_to_axis_angle(head_combined_q)[0]

            # R15_LEFT_UPPER_ARM (4) <- SMPL L_Collar @ L_Shoulder + 90° Z correction
            l_collar_q = smpl_quats[self.SMPL_L_COLLAR]
            l_shoulder_q = smpl_quats[self.SMPL_L_SHOULDER]
            correction_q = self._euler_to_quat((0, 0, 90), "xyz")
            l_upper_arm_q = self._quat_multiply(
                self._quat_multiply(l_collar_q.reshape(1, 4), l_shoulder_q.reshape(1, 4)), correction_q.reshape(1, 4)
            )
            r15_rotations[frame_idx, self.R15_LEFT_UPPER_ARM] = self._quat_to_axis_angle(l_upper_arm_q)[0]

            # R15_LEFT_LOWER_ARM (5) <- SMPL L_Elbow + alignment corrections
            l_elbow_q = smpl_quats[self.SMPL_L_ELBOW]
            align_q = self._euler_to_quat((0, 0, -80), "xyz")
            align_inv_q = self._euler_to_quat((0, 0, 80), "xyz")
            correction_q = self._euler_to_quat((13, 0, 0), "xyz")
            l_lower_arm_q = self._quat_multiply(
                self._quat_multiply(
                    self._quat_multiply(align_q.reshape(1, 4), l_elbow_q.reshape(1, 4)), align_inv_q.reshape(1, 4)
                ),
                correction_q.reshape(1, 4),
            )
            r15_rotations[frame_idx, self.R15_LEFT_LOWER_ARM] = self._quat_to_axis_angle(l_lower_arm_q)[0]

            # R15_LEFT_HAND (6) <- SMPL L_Wrist + alignment corrections
            l_wrist_q = smpl_quats[self.SMPL_L_WRIST]
            l_hand_q = self._quat_multiply(
                self._quat_multiply(align_q.reshape(1, 4), l_wrist_q.reshape(1, 4)), align_inv_q.reshape(1, 4)
            )
            r15_rotations[frame_idx, self.R15_LEFT_HAND] = self._quat_to_axis_angle(l_hand_q)[0]

            # R15_RIGHT_UPPER_ARM (7) <- SMPL R_Collar @ R_Shoulder - 90° Z correction
            r_collar_q = smpl_quats[self.SMPL_R_COLLAR]
            r_shoulder_q = smpl_quats[self.SMPL_R_SHOULDER]
            correction_q = self._euler_to_quat((0, 0, -90), "xyz")
            r_upper_arm_q = self._quat_multiply(
                self._quat_multiply(r_collar_q.reshape(1, 4), r_shoulder_q.reshape(1, 4)), correction_q.reshape(1, 4)
            )
            r15_rotations[frame_idx, self.R15_RIGHT_UPPER_ARM] = self._quat_to_axis_angle(r_upper_arm_q)[0]

            # R15_RIGHT_LOWER_ARM (8) <- SMPL R_Elbow + alignment corrections
            r_elbow_q = smpl_quats[self.SMPL_R_ELBOW]
            align_q_r = self._euler_to_quat((0, 0, 80), "xyz")
            align_inv_q_r = self._euler_to_quat((0, 0, -80), "xyz")
            correction_q = self._euler_to_quat((13, 0, 0), "xyz")
            r_lower_arm_q = self._quat_multiply(
                self._quat_multiply(
                    self._quat_multiply(align_q_r.reshape(1, 4), r_elbow_q.reshape(1, 4)), align_inv_q_r.reshape(1, 4)
                ),
                correction_q.reshape(1, 4),
            )
            r15_rotations[frame_idx, self.R15_RIGHT_LOWER_ARM] = self._quat_to_axis_angle(r_lower_arm_q)[0]

            # R15_RIGHT_HAND (9) <- SMPL R_Wrist + alignment corrections
            r_wrist_q = smpl_quats[self.SMPL_R_WRIST]
            r_hand_q = self._quat_multiply(
                self._quat_multiply(align_q_r.reshape(1, 4), r_wrist_q.reshape(1, 4)), align_inv_q_r.reshape(1, 4)
            )
            r15_rotations[frame_idx, self.R15_RIGHT_HAND] = self._quat_to_axis_angle(r_hand_q)[0]

            # R15_LEFT_UPPER_LEG (10) <- SMPL L_Hip
            r15_rotations[frame_idx, self.R15_LEFT_UPPER_LEG] = frame_rotations[self.SMPL_L_HIP]

            # R15_LEFT_LOWER_LEG (11) <- SMPL L_Knee
            r15_rotations[frame_idx, self.R15_LEFT_LOWER_LEG] = frame_rotations[self.SMPL_L_KNEE]

            # R15_LEFT_FOOT (12) <- SMPL L_Ankle
            r15_rotations[frame_idx, self.R15_LEFT_FOOT] = frame_rotations[self.SMPL_L_ANKLE]

            # R15_RIGHT_UPPER_LEG (13) <- SMPL R_Hip
            r15_rotations[frame_idx, self.R15_RIGHT_UPPER_LEG] = frame_rotations[self.SMPL_R_HIP]

            # R15_RIGHT_LOWER_LEG (14) <- SMPL R_Knee
            r15_rotations[frame_idx, self.R15_RIGHT_LOWER_LEG] = frame_rotations[self.SMPL_R_KNEE]

            # R15_RIGHT_FOOT (15) <- SMPL R_Ankle
            r15_rotations[frame_idx, self.R15_RIGHT_FOOT] = frame_rotations[self.SMPL_R_ANKLE]

        # Scale translation if requested
        if self.apply_scale:
            r15_translation = smpl_translation * self.SCALE_FACTOR * 100  # Convert to cm
        else:
            r15_translation = smpl_translation.copy()

        # Create R15 skeleton
        r15_skeleton = Skeleton.from_r15()

        return {
            "joint_rotations": r15_rotations,
            "root_translation": r15_translation,
            "skeleton": r15_skeleton,
            "framerate": motion.framerate,
        }


class MotionToMHRConverter(BaseMotionConverter):
    """Convert Motion class to MHR format.

    Requires that the Motion object was created from MHR data or has MHR params in `model_data`.
    """

    input_format = "motion"
    output_format = "mhr"

    def convert(self, data: Motion) -> dict[str, Any]:
        if data.body_model_type != "mhr":
            # In future, we could implement retargeting here
            raise NotImplementedError("Can only convert MHR-sourced Motion back to MHR format currently.")

        required_keys = ["model_parameters"]
        for key in required_keys:
            if key not in data.model_data:
                raise ValueError(f"Motion model_data missing required MHR key: {key}")

        return data.model_data


def get_motion_converter(conversion_type: str, **kwargs) -> BaseMotionConverter:
    from .align_converter import ALIGNConverter

    converters = {
        "motion2humanml3d": MotionToHumanML3DConverter,
        "humanml3d2motion": HumanML3DToMotionConverter,
        "format272tomotion": Format272ToMotionConverter,
        "smpl2motion": SMPLToMotionConverter,
        "motion2smpl": MotionToSMPLConverter,
        "motion2gvhmr": MotionToGVHMRConverter,
        "gvhmr2motion": GVHMRToMotionConverter,
        "smplpos2humanml3d": SMPLPositionsToHumanML3DConverter,
        "mhr2motion": MHRToMotionConverter,
        "motion2mhr": MotionToMHRConverter,
        "smpl2r15": SMPLToR15Converter,
        "smpl2format272": SMPLTo272Converter,
        "format272tosmpl": Format272ToSMPLConverter,
        "motion2align": ALIGNConverter,
        "align2motion": ALIGNConverter,
    }

    if conversion_type not in converters:
        available = ", ".join(converters.keys())
        raise ValueError(f"Unknown conversion type: {conversion_type}. Available converters: {available}")

    return converters[conversion_type](**kwargs)


AVAILABLE_CONVERTERS = {
    "gvhmr2motion": "Convert GVHMR SMPL parameters to Motion class [✅ IMPLEMENTED]",
    "motion2gvhmr": "Convert Motion class to GVHMR format [✅ IMPLEMENTED]",
    "motion2smpl": "Convert Motion class to SMPL axis-angle format [✅ IMPLEMENTED]",
    "motion2humanml3d": "Convert Motion to HumanML3D 263-element format [🔄 TO BE IMPLEMENTED]",
    "humanml3d2motion": "Convert HumanML3D format to Motion class [🔄 TO BE IMPLEMENTED]",
    "humanml3d2format272": "Convert HumanML3D 263D to 272D motion format [✅ IMPLEMENTED]",
    "format272tomotion": "Convert 272D motion format to Motion class [✅ IMPLEMENTED]",
    "smpl2motion": "Convert SMPL joint positions to Motion class [✅ IMPLEMENTED]",
    "smplpos2humanml3d": "Convert 22-joint SMPL positions to HumanML3D features [✅ IMPLEMENTED]",
    "mhr2motion": "Convert MHR parameters to Motion class [✅ IMPLEMENTED]",
    "motion2mhr": "Convert Motion class to MHR parameters [✅ IMPLEMENTED]",
    "smpl2r15": "Convert SMPL motion to Roblox R15 format [✅ IMPLEMENTED]",
    "smpl2format272": "Convert SMPL parameters to 272-dim motion format [✅ IMPLEMENTED]",
    "format272tosmpl": "Convert 272-dim motion format to SMPL parameters [✅ IMPLEMENTED]",
    "motion2align": "Convert Motion class to ALIGN 232D features [✅ IMPLEMENTED]",
    "align2motion": "Convert ALIGN 232D features to Motion class [✅ IMPLEMENTED]",
}


def list_converters() -> dict[str, str]:
    return AVAILABLE_CONVERTERS.copy()
