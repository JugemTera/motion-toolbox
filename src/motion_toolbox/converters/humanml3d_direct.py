"""Direct converter from HumanML3D 263D format to Motion, bypassing 272D format."""

import numpy as np

from ..core.motion import Motion
from ..core.skeleton import Skeleton
from ..utils.humanml_utils import cont6d_to_matrix_np
from ..utils.quaternion import rotation_matrix_to_axis_angle
from .base_converter import BaseMotionConverter


class HumanML3DDirectToMotionConverter(BaseMotionConverter):
    """Convert HumanML3D 263D format directly to Motion, bypassing 272D.

    HumanML3D 263D format (per frame, actually T-1 frames):
    - [0:1]: Root angular velocity (yaw)
    - [1:3]: Root linear velocity XZ (in character's local frame)
    - [3:4]: Root Y position
    - [4:67]: Root-relative positions (RIC) for 21 joints (21 x 3 = 63D)
    - [67:193]: 6D rotation representations for 21 joints (21 x 6 = 126D)
    - [193:197]: Foot contact labels (4D)
    - [197:263]: Additional features (velocities, etc.)

    Key insight: The rotations in [67:193] are the actual joint rotations we need!
    We just need to:
    1. Reconstruct root position from velocities
    2. Extract 6D rotations → convert to matrices → convert to axis-angles
    3. Handle the root rotation (yaw angle)
    4. Pad to 24 joints for SMPL compatibility
    """

    input_format = "humanml3d"
    output_format = "motion"
    _saves_objects = True

    def __init__(
        self,
        framerate: float = 20.0,
        body_model_path: str | None = None,
        use_smpl_skeleton: bool = False,
    ):
        """Initialize converter.

        Args:
            framerate: Target framerate (HumanML3D is typically 20 FPS)
            body_model_path: Path to SMPL body models (for GLB export)
            use_smpl_skeleton: If True, use standard SMPL skeleton; if False, use HumanML3D skeleton
        """
        super().__init__()
        self.framerate = framerate
        self.body_model_path = body_model_path
        self.use_smpl_skeleton = use_smpl_skeleton

    def convert(self, data: np.ndarray) -> Motion:
        """Convert HumanML3D 263D format to Motion.

        Args:
            data: HumanML3D features, shape (T-1, 263)

        Returns:
            Motion object with proper rotations and positions
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Expected numpy ndarray input")
        if data.ndim != 2:
            raise ValueError(f"Expected 2D input (frames-1, 263), got shape {data.shape}")
        if data.shape[1] != 263:
            raise ValueError(f"Expected 263 features per frame, got {data.shape[1]}")

        T_minus_1 = data.shape[0]
        T = T_minus_1 + 1

        # Extract components from 263D format
        r_velocity_y = data[:, 0:1]  # Root angular velocity (yaw)
        l_velocity_xz = data[:, 1:3]  # Root linear velocity in local frame
        root_y = data[:, 3:4]  # Root Y position
        ric_data = data[:, 4:67].reshape(T_minus_1, 21, 3)  # Root-relative positions
        rot_6d = data[:, 67:193].reshape(T_minus_1, 21, 6)  # 6D rotations (the key data!)

        # Step 1: Reconstruct root trajectory
        # Integrate yaw angle
        yaw_angles = np.zeros(T, dtype=np.float32)
        for t in range(T_minus_1):
            yaw_angles[t + 1] = yaw_angles[t] + r_velocity_y[t, 0]

        # Integrate root position in XZ plane (rotate local velocity by yaw)
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

        # Set Y position from data
        root_positions[1:, 1] = root_y[:, 0]

        # Step 2: Convert 6D rotations to rotation matrices
        # Note: rot_6d is shape (T-1, 21, 6) - we need to handle frame 0
        rot_matrices = np.zeros((T, 22, 3, 3), dtype=np.float32)

        # Frame 0: identity rotations
        rot_matrices[0] = np.eye(3, dtype=np.float32)

        # Frames 1+: convert from 6D
        if T_minus_1 > 0:
            # Convert non-root joints
            rot_matrices_from_6d = cont6d_to_matrix_np(rot_6d)  # Shape: (T-1, 21, 3, 3)
            rot_matrices[1:, 1:, :, :] = rot_matrices_from_6d  # Joints 1-21

            # Root rotation: create from yaw angle
            for t in range(1, T):
                yaw = yaw_angles[t]
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                rot_matrices[t, 0, 0, 0] = cos_yaw
                rot_matrices[t, 0, 0, 2] = sin_yaw
                rot_matrices[t, 0, 1, 1] = 1.0
                rot_matrices[t, 0, 2, 0] = -sin_yaw
                rot_matrices[t, 0, 2, 2] = cos_yaw

        # Step 3: Convert rotation matrices to axis-angles
        # Pad to 24 joints for SMPL compatibility (add hands)
        joint_rotations = np.zeros((T, 24, 3), dtype=np.float32)

        for t in range(T):
            for joint_idx in range(22):
                rot_matrix = rot_matrices[t, joint_idx]
                axis_angle = rotation_matrix_to_axis_angle(rot_matrix)
                joint_rotations[t, joint_idx] = axis_angle

        # Joints 22-23 (hands) remain zero rotation

        # Step 4: Create skeleton
        if self.use_smpl_skeleton:
            skeleton = Skeleton.from_smpl()
        else:
            # Use HumanML3D skeleton with 24 joints (padded with hands)
            humanml_skeleton = Skeleton.from_humanml3d()
            skeleton = self._create_smpl_compatible_skeleton(humanml_skeleton)

        # Step 5: Create Motion object
        motion = Motion(
            joint_rotations=joint_rotations,
            root_translation=root_positions,
            skeleton=skeleton,
            framerate=self.framerate,
            up_axis="y",
            body_model_path=self.body_model_path,
            forward_kinematics="smpl" if self.use_smpl_skeleton else "custom",
            floor_alignment="joint",
        )

        return motion

    def _create_smpl_compatible_skeleton(self, humanml_skeleton: Skeleton) -> Skeleton:
        """Extend 22-joint HumanML3D skeleton to 24-joint SMPL-compatible skeleton."""
        joint_names = humanml_skeleton.joint_names + ("left_hand", "right_hand")
        parent_indices = humanml_skeleton.parent_indices + (20, 21)  # Parents are wrists

        # Copy offsets and add hand offsets
        offsets = np.zeros((24, 3), dtype=np.float32)
        offsets[:22] = humanml_skeleton.offsets

        # Hands: extend from wrists with reasonable length
        left_wrist_offset = humanml_skeleton.offsets[20]
        right_wrist_offset = humanml_skeleton.offsets[21]
        offsets[22] = left_wrist_offset * 0.6  # left_hand
        offsets[23] = right_wrist_offset * 0.6  # right_hand

        return Skeleton(joint_names=joint_names, parent_indices=parent_indices, offsets=offsets)
