"""Convert HumanML3D to Motion using positions + IK (the correct way)."""

import numpy as np

from ..core.motion import Motion
from ..core.skeleton import Skeleton
from ..utils.humanml_utils import FACE_JOINT_IDX_T2M, T2M_KINEMATIC_CHAIN, T2M_RAW_OFFSETS, HMMLSkeleton
from ..utils.quaternion import quaternion_to_axis_angle
from .base_converter import BaseMotionConverter


class HumanML3DPositionsToMotionConverter(BaseMotionConverter):
    """Convert HumanML3D using positions (RIC) + IK - the ground truth approach.

    HumanML3D stores joint positions (RIC = root-relative in facing-forward frame).
    The rotations in the 263D format are derived features for ML, not direct SMPL rotations.

    Correct approach:
    1. Extract RIC positions (these are the ACTUAL joint positions!)
    2. Reconstruct global positions from RIC + root trajectory
    3. Use inverse kinematics to compute rotations from positions
    4. Create Motion with computed rotations
    """

    input_format = "humanml3d"
    output_format = "motion"
    _saves_objects = True

    def __init__(
        self,
        framerate: float = 20.0,
        body_model_path: str | None = None,
    ):
        super().__init__()
        self.framerate = framerate
        self.body_model_path = body_model_path

        # HumanML3D skeleton for IK
        self.humanml_skeleton = HMMLSkeleton(T2M_RAW_OFFSETS.copy(), T2M_KINEMATIC_CHAIN)

    def convert(self, data: np.ndarray) -> Motion:
        """Convert HumanML3D 263D to Motion using positions + IK.

        Args:
            data: HumanML3D features, shape (T-1, 263)

        Returns:
            Motion object with correct proportions
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Expected numpy ndarray input")
        if data.ndim != 2:
            raise ValueError(f"Expected 2D input (frames-1, 263), got shape {data.shape}")
        if data.shape[1] != 263:
            raise ValueError(f"Expected 263 features per frame, got {data.shape[1]}")

        T_minus_1 = data.shape[0]
        T = T_minus_1 + 1

        # Step 1: Extract positions and root trajectory
        r_velocity_y = data[:, 0:1]  # Root angular velocity (yaw)
        l_velocity_xz = data[:, 1:3]  # Root linear velocity
        root_y = data[:, 3:4]  # Root Y position
        ric_data = data[:, 4:67].reshape(T_minus_1, 21, 3)  # ROOT-RELATIVE POSITIONS (ground truth!)

        # Step 2: Reconstruct root trajectory
        yaw_angles = np.zeros(T, dtype=np.float32)
        for t in range(T_minus_1):
            yaw_angles[t + 1] = yaw_angles[t] + r_velocity_y[t, 0]

        root_positions = np.zeros((T, 3), dtype=np.float32)
        for t in range(T_minus_1):
            cos_yaw = np.cos(-yaw_angles[t])
            sin_yaw = np.sin(-yaw_angles[t])
            vel_global = np.array(
                [
                    cos_yaw * l_velocity_xz[t, 0] - sin_yaw * l_velocity_xz[t, 1],
                    0.0,
                    sin_yaw * l_velocity_xz[t, 0] + cos_yaw * l_velocity_xz[t, 1],
                ],
                dtype=np.float32,
            )
            root_positions[t + 1, [0, 2]] = root_positions[t, [0, 2]] + vel_global[[0, 2]]
        root_positions[1:, 1] = root_y[:, 0]

        # Step 3: Build global positions from RIC
        # RIC is in a "facing forward" frame, so we need to rotate by yaw
        global_positions = np.zeros((T, 22, 3), dtype=np.float32)
        global_positions[:, 0, :] = root_positions  # Root

        for t in range(1, T):  # Start from frame 1 (frame 0 has no RIC)
            yaw = yaw_angles[t]
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)

            for joint_idx in range(21):  # 21 joints in RIC
                # RIC position (root-relative, facing-forward frame)
                x, y, z = ric_data[t - 1, joint_idx]

                # Rotate by yaw to get global direction
                x_global = cos_yaw * x - sin_yaw * z
                z_global = sin_yaw * x + cos_yaw * z

                # Add root position to get global position
                global_positions[t, joint_idx + 1, 0] = x_global + root_positions[t, 0]
                global_positions[t, joint_idx + 1, 1] = y + root_positions[t, 1]
                global_positions[t, joint_idx + 1, 2] = z_global + root_positions[t, 2]

        # Step 4: Use inverse kinematics to compute rotations from positions
        # This is the key: convert positions → quaternions → axis-angles
        quat_rotations = self.humanml_skeleton.inverse_kinematics_np(global_positions, FACE_JOINT_IDX_T2M)

        # Convert quaternions (T, 22, 4) to axis-angles (T, 22, 3)
        joint_rotations_22 = np.zeros((T, 22, 3), dtype=np.float32)
        for t in range(T):
            for j in range(22):
                quat = quat_rotations[t, j]  # (w, x, y, z)
                # Handle zero/invalid quaternions (e.g., from frame 0 or numerical issues)
                quat_norm = np.linalg.norm(quat)
                if quat_norm < 1e-6:
                    # Zero rotation
                    joint_rotations_22[t, j] = np.zeros(3, dtype=np.float32)
                else:
                    # Normalize and convert
                    quat = quat / quat_norm
                    axis_angle = quaternion_to_axis_angle(quat)
                    joint_rotations_22[t, j] = axis_angle

        # Step 5: Pad to 24 joints for SMPL compatibility
        joint_rotations = np.zeros((T, 24, 3), dtype=np.float32)
        joint_rotations[:, :22, :] = joint_rotations_22
        # Joints 22-23 (hands) remain zero

        # Step 6: Create skeleton from actual positions
        skeleton = self._create_skeleton_from_positions(global_positions)

        # Step 7: Create Motion
        motion = Motion(
            joint_rotations=joint_rotations,
            root_translation=root_positions,
            skeleton=skeleton,
            framerate=self.framerate,
            up_axis="y",
            body_model_path=self.body_model_path,
            forward_kinematics="custom",  # Use custom FK with extracted skeleton
            floor_alignment="joint",
        )

        return motion

    def _create_skeleton_from_positions(self, global_positions: np.ndarray) -> Skeleton:
        """Create 24-joint skeleton from 22-joint positions."""
        # Get HumanML3D skeleton template
        from ..core.skeleton import Skeleton

        template_skeleton = Skeleton.from_humanml3d()

        # Compute average bone offsets from positions
        start_frame = min(10, global_positions.shape[0] - 1)
        end_frame = min(50, global_positions.shape[0])

        offsets_22 = np.zeros((22, 3), dtype=np.float32)
        for joint_idx in range(22):
            parent_idx = template_skeleton.parent_indices[joint_idx]
            if parent_idx is None:
                offsets_22[joint_idx] = np.array([0.0, 0.0, 0.0])
            else:
                bone_vectors = []
                for t in range(start_frame, end_frame):
                    bone_vec = global_positions[t, joint_idx] - global_positions[t, parent_idx]
                    bone_vectors.append(bone_vec)
                offsets_22[joint_idx] = np.mean(bone_vectors, axis=0)

        # Extend to 24 joints (add hands)
        joint_names = template_skeleton.joint_names + ("left_hand", "right_hand")
        parent_indices = template_skeleton.parent_indices + (20, 21)
        offsets = np.zeros((24, 3), dtype=np.float32)
        offsets[:22] = offsets_22
        offsets[22] = offsets_22[20] * 0.6  # left_hand
        offsets[23] = offsets_22[21] * 0.6  # right_hand

        return Skeleton(joint_names=joint_names, parent_indices=parent_indices, offsets=offsets)
