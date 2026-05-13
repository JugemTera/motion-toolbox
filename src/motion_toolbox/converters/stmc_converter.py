"""STMC (Multi-Track Timeline Control) motion representation converter.

This module implements the 205-dimensional motion representation from the
"Multi-Track Timeline Control" (STMC) paper. It provides bi-directional conversion
between SMPL parameters (via Motion class) and the STMC feature representation.

Reference: https://github.com/nv-tlabs/stmc

STMC 205-Dimensional Format:
    - [0:2]   (2D):  Root Linear Velocity (XZ plane, local to facing direction)
    - [2:3]   (1D):  Root Angular Velocity (Y-axis only)
    - [3:7]   (4D):  Foot Contact Labels (L_Heel, L_Toe, R_Heel, R_Toe)
    - [7:73]  (66D): Local 3D Joint Positions (22 joints relative to root)
    - [73:205](132D): 6D Continuous Rotations (22 joints)
"""

from pathlib import Path

import numpy as np
import torch

from ..core.motion import Motion
from ..core.skeleton import Skeleton
from ..utils.quaternion import (
    q_inv,
    q_mul,
    q_rotate,
)
from .base_converter import BaseMotionConverter


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D continuous rotation representation to rotation matrix.

    Args:
        rot_6d: 6D rotation, shape (..., 6) representing first two columns of rotation matrix

    Returns:
        Rotation matrix, shape (..., 3, 3)
    """
    # Reshape to get first two columns
    x = rot_6d[..., :3]  # First column
    y = rot_6d[..., 3:]  # Second column (before Gram-Schmidt)

    # Gram-Schmidt orthogonalization
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    y = y - (x * y).sum(axis=-1, keepdims=True) * x
    y = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)

    # Third column is cross product
    z = np.cross(x, y, axis=-1)

    # Stack to form rotation matrix
    return np.stack([x, y, z], axis=-1)  # (..., 3, 3)


def matrix_to_rotation_6d(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D continuous rotation representation.

    Args:
        rot_mat: Rotation matrix, shape (..., 3, 3)

    Returns:
        6D rotation representation, shape (..., 6)
    """
    # Take first two columns
    rot_6d = np.concatenate([rot_mat[..., 0], rot_mat[..., 1]], axis=-1)
    return rot_6d


def axis_angle_to_rotation_6d(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to 6D rotation representation.

    Args:
        axis_angle: Axis-angle rotations, shape (..., 3)

    Returns:
        6D rotation representation, shape (..., 6)
    """
    from scipy.spatial.transform import Rotation as R

    # Copy to ensure writeable array
    axis_angle = np.array(axis_angle)

    # Convert axis-angle to rotation matrix
    rot = R.from_rotvec(axis_angle.reshape(-1, 3))
    rot_mat = rot.as_matrix()

    # Convert to 6D
    rot_6d = matrix_to_rotation_6d(rot_mat)

    # Reshape back
    original_shape = axis_angle.shape[:-1]
    return rot_6d.reshape(original_shape + (6,))


def rotation_6d_to_axis_angle(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to axis-angle.

    Args:
        rot_6d: 6D rotation representation, shape (..., 6)

    Returns:
        Axis-angle rotations, shape (..., 3)
    """
    from scipy.spatial.transform import Rotation as R

    # Convert to rotation matrix
    rot_mat = rotation_6d_to_matrix(rot_6d)

    # Convert to axis-angle
    original_shape = rot_6d.shape[:-1]
    rot_mat_flat = rot_mat.reshape(-1, 3, 3)
    rot = R.from_matrix(rot_mat_flat)
    axis_angle = rot.as_rotvec()

    return axis_angle.reshape(original_shape + (3,))


class STMC205Converter(BaseMotionConverter):
    """Bi-directional converter for STMC 205-dimensional motion representation.

    This converter transforms between Motion objects (SMPL parameters) and the
    STMC 205-dimensional feature representation used in the Multi-Track Timeline
    Control paper.
    """

    input_format = "motion"
    output_format = "stmc205"

    def __init__(
        self,
        smpl_model_path: str | None = None,
        device: str = "cpu",
        foot_contact_vel_threshold: float = 0.002,
        foot_contact_height_threshold: float = 0.02,
    ):
        """Initialize STMC converter.

        Args:
            smpl_model_path: Path to SMPL model files for forward kinematics
            device: Device to run computations on
            foot_contact_vel_threshold: Velocity threshold for foot contact detection (m/frame)
            foot_contact_height_threshold: Height threshold for foot contact detection (m)
        """
        super().__init__()

        if smpl_model_path is None:
            smpl_model_path = str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl/")

        self.smpl_model_path = smpl_model_path
        self.device = device
        self.foot_contact_vel_threshold = foot_contact_vel_threshold
        self.foot_contact_height_threshold = foot_contact_height_threshold

        # Lazy initialization of SMPL model
        self._smpl_model = None

    def _get_smpl_model(self, batch_size: int = 1):
        """Get or create SMPL model."""
        if self._smpl_model is None:
            import smplx

            self._smpl_model = smplx.create(
                model_path=self.smpl_model_path,
                model_type="smpl",
                gender="neutral",
                ext="pkl",
                batch_size=batch_size,
            )
        return self._smpl_model

    def forward(self, motion: Motion) -> torch.Tensor:
        """Convert Motion to STMC 205D representation.

        Args:
            motion: Motion object with SMPL parameters

        Returns:
            STMC features, shape (num_frames, 205)

        Note:
            STMC uses 22 joints. If the motion has more joints, only the first 22 are used.
        """
        num_frames = motion.num_frames
        # STMC uses 22 joints - use first 22 joints of the motion
        num_joints = min(motion.num_joints, 22)

        # Get global joint positions from forward kinematics
        # Use only first 22 joints for STMC
        positions = motion.positions[:, :22, :]  # (T, 22, 3)

        # Compute facing rotation (Y-axis rotation from forward direction)
        facing_rotations = self._compute_facing_rotations(positions)  # (T, 4) quaternions

        # Compute root velocities
        root_linear_vel_local, root_angular_vel = self._compute_root_velocities(
            motion.root_translation, facing_rotations, motion.framerate
        )  # (T, 2), (T, 1)

        # Compute foot contacts
        foot_contacts = self._compute_foot_contacts(positions, motion.framerate)  # (T, 4)

        # Compute local joint positions (relative to root, in root frame)
        # This includes all 22 joints (root will be at origin in local frame)
        local_positions = self._compute_local_positions(
            positions, motion.root_translation, facing_rotations
        )  # (T, 22, 3)
        local_positions_flat = local_positions.reshape(num_frames, -1)  # (T, 22*3)

        # Convert axis-angle rotations to 6D representation (first 22 joints only)
        rotations_6d = axis_angle_to_rotation_6d(motion.joint_rotations[:, :22, :])  # (T, 22, 6)
        rotations_6d_flat = rotations_6d.reshape(num_frames, -1)  # (T, 22*6)

        # Concatenate all features
        features = np.concatenate(
            [
                root_linear_vel_local,  # (T, 2)
                root_angular_vel,  # (T, 1)
                foot_contacts,  # (T, 4)
                local_positions_flat,  # (T, 66)
                rotations_6d_flat,  # (T, 132)
            ],
            axis=-1,
        )  # (T, 205)

        return torch.from_numpy(features).float()

    def inverse(self, features: torch.Tensor, skeleton: Skeleton | None = None) -> Motion:
        """Convert STMC 205D representation back to Motion.

        Args:
            features: STMC features, shape (num_frames, 205) or (batch, num_frames, 205)
            skeleton: Skeleton to use (defaults to SMPL skeleton)

        Returns:
            Motion object reconstructed from features
        """
        # Handle batch dimension
        if features.ndim == 3:
            # Take first batch element
            features = features[0]

        features_np = features.detach().cpu().numpy()
        num_frames = features_np.shape[0]

        # STMC uses 22 joints
        stmc_num_joints = 22

        # Use SMPL skeleton if not provided, but note we'll only reconstruct 22 joints
        if skeleton is None:
            skeleton = Skeleton.from_smpl()

        # We reconstruct motion with 22 joints (STMC standard)
        # If skeleton has more joints, we'll pad with zeros later
        num_joints = stmc_num_joints

        # Extract features
        root_linear_vel_local = features_np[:, 0:2]  # (T, 2)
        root_angular_vel = features_np[:, 2:3]  # (T, 1)
        # foot_contacts = features_np[:, 3:7]           # (T, 4) - not used in inverse
        # local_positions = features_np[:, 7:73]        # (T, 66) - not used in inverse
        rotations_6d_flat = features_np[:, 73:205]  # (T, 132)

        # Recover rotations from 6D representation (22 joints)
        rotations_6d = rotations_6d_flat.reshape(num_frames, stmc_num_joints, 6)
        joint_rotations_22 = rotation_6d_to_axis_angle(rotations_6d)  # (T, 22, 3)

        # If skeleton has more joints, pad with zeros
        if skeleton.num_joints > stmc_num_joints:
            joint_rotations = np.zeros((num_frames, skeleton.num_joints, 3))
            joint_rotations[:, :stmc_num_joints, :] = joint_rotations_22
        else:
            joint_rotations = joint_rotations_22

        # Integrate root angular velocity to get heading
        heading_angles = self._integrate_angular_velocity(root_angular_vel)  # (T,)

        # Create heading quaternions (Y-axis rotations)
        heading_quats = self._angles_to_y_quaternions(heading_angles)  # (T, 4)

        # Rotate local linear velocity by heading to get global velocity
        # Expand XZ velocity to 3D (add zero Y component)
        root_linear_vel_local_3d = np.concatenate(
            [
                root_linear_vel_local[:, 0:1],  # X
                np.zeros((num_frames, 1)),  # Y
                root_linear_vel_local[:, 1:2],  # Z
            ],
            axis=-1,
        )  # (T, 3)

        # Rotate by heading to get global velocity
        root_linear_vel_global = q_rotate(heading_quats, root_linear_vel_local_3d)  # (T, 3)

        # Integrate global velocity to get root translation
        root_translation = self._integrate_linear_velocity(root_linear_vel_global)  # (T, 3)

        # Create Motion object
        motion = Motion(
            joint_rotations=joint_rotations,
            root_translation=root_translation,
            skeleton=skeleton,
            framerate=30.0,  # Default framerate
            body_model_path=self.smpl_model_path,
            floor_alignment="mesh",
        )

        return motion

    def convert(self, data: Motion | torch.Tensor | np.ndarray) -> torch.Tensor | Motion:
        """Convert between Motion and STMC representations.

        Args:
            data: Either a Motion object (forward conversion) or tensor/array (inverse conversion)

        Returns:
            Converted data (tensor if forward, Motion if inverse)
        """
        if isinstance(data, Motion):
            return self.forward(data)
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            return self.inverse(data)
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")

    def _compute_facing_rotations(self, positions: np.ndarray) -> np.ndarray:
        """Compute facing direction quaternions from hip positions.

        Args:
            positions: Joint positions, shape (T, J, 3)

        Returns:
            Facing quaternions (Y-axis rotations), shape (T, 4)
        """
        num_frames = positions.shape[0]

        # SMPL joint indices for hips
        left_hip_idx = 1
        right_hip_idx = 2

        # Compute hip direction
        hip_vector = positions[:, right_hip_idx, :] - positions[:, left_hip_idx, :]  # (T, 3)
        hip_vector[:, 1] = 0  # Zero out Y component
        hip_vector = hip_vector / (np.linalg.norm(hip_vector, axis=-1, keepdims=True) + 1e-8)

        # Forward direction is perpendicular to hip direction
        forward = np.cross(np.array([0, 1, 0]), hip_vector, axis=-1)
        forward = forward / (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8)

        # Compute rotation from Z-axis to forward direction
        z_axis = np.array([0, 0, 1])

        # Compute angle
        cos_angle = np.dot(forward, z_axis)
        sin_angle = np.cross(forward, z_axis)[:, 1]  # Y component of cross product

        # Create Y-axis rotation quaternions
        angle = np.arctan2(sin_angle, cos_angle)
        quaternions = np.zeros((num_frames, 4))
        quaternions[:, 0] = np.cos(angle / 2)  # w
        quaternions[:, 2] = np.sin(angle / 2)  # y (Y-axis rotation)

        return quaternions

    def _compute_root_velocities(
        self,
        root_translation: np.ndarray,
        facing_rotations: np.ndarray,
        framerate: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute root linear and angular velocities.

        Args:
            root_translation: Root positions, shape (T, 3)
            facing_rotations: Facing quaternions, shape (T, 4)
            framerate: Frames per second

        Returns:
            Tuple of (linear_velocity_local, angular_velocity)
            - linear_velocity_local: (T, 2) XZ velocity in local frame
            - angular_velocity: (T, 1) Y-axis angular velocity
        """
        num_frames = root_translation.shape[0]

        # Linear velocity (global frame) - displacement per frame (not per second)
        linear_vel_global = np.zeros_like(root_translation)
        if num_frames > 1:
            linear_vel_global[:-1] = root_translation[1:] - root_translation[:-1]
            linear_vel_global[-1] = linear_vel_global[-2]  # Repeat last

        # Transform to local frame
        facing_inv = q_inv(facing_rotations)
        linear_vel_local_3d = q_rotate(facing_inv, linear_vel_global)

        # Extract XZ components
        linear_vel_local = np.stack(
            [
                linear_vel_local_3d[:, 0],  # X
                linear_vel_local_3d[:, 2],  # Z
            ],
            axis=-1,
        )

        # Angular velocity (Y-axis only) - angle per frame (not per second)
        angular_vel = np.zeros((num_frames, 1))
        if num_frames > 1:
            for i in range(num_frames - 1):
                # Compute relative rotation
                q_rel = q_mul(facing_rotations[i + 1], q_inv(facing_rotations[i]))

                # Extract Y-axis rotation angle
                # For small rotations, angle ≈ 2 * y_component (when w ≈ 1)
                angle = 2.0 * np.arcsin(np.clip(q_rel[2], -1.0, 1.0))  # y component
                angular_vel[i, 0] = angle

            angular_vel[-1] = angular_vel[-2]  # Repeat last

        return linear_vel_local, angular_vel

    def _compute_foot_contacts(self, positions: np.ndarray, framerate: float) -> np.ndarray:
        """Compute binary foot contact labels.

        Args:
            positions: Joint positions, shape (T, J, 3)
            framerate: Frames per second

        Returns:
            Foot contacts, shape (T, 4) for [L_Heel, L_Toe, R_Heel, R_Toe]
        """
        num_frames = positions.shape[0]
        dt = 1.0 / framerate

        # SMPL joint indices (approximate - may need adjustment based on skeleton)
        left_ankle_idx = 7
        left_foot_idx = 10
        right_ankle_idx = 8
        right_foot_idx = 11

        foot_indices = [left_ankle_idx, left_foot_idx, right_ankle_idx, right_foot_idx]
        contacts = np.zeros((num_frames, 4))

        for i, joint_idx in enumerate(foot_indices):
            # Compute velocity
            velocities = np.zeros(num_frames)
            if num_frames > 1:
                vel = np.linalg.norm(positions[1:, joint_idx, :] - positions[:-1, joint_idx, :], axis=-1) / dt
                velocities[:-1] = vel
                velocities[-1] = vel[-1]

            # Get heights
            heights = positions[:, joint_idx, 1]  # Y component

            # Contact = low velocity AND low height
            contacts[:, i] = (
                (velocities < self.foot_contact_vel_threshold) & (heights < self.foot_contact_height_threshold)
            ).astype(np.float32)

        return contacts

    def _compute_local_positions(
        self,
        positions: np.ndarray,
        root_translation: np.ndarray,
        facing_rotations: np.ndarray,
    ) -> np.ndarray:
        """Compute local joint positions relative to root in root frame.

        Args:
            positions: Global joint positions, shape (T, J, 3)
            root_translation: Root positions, shape (T, 3)
            facing_rotations: Facing quaternions, shape (T, 4)

        Returns:
            Local joint positions, shape (T, J, 3)
        """
        # Compute relative positions
        relative_positions = positions - root_translation[:, None, :]  # (T, J, 3)

        # Transform to local frame
        facing_inv = q_inv(facing_rotations)
        local_positions = np.zeros_like(relative_positions)

        for i in range(relative_positions.shape[1]):
            local_positions[:, i, :] = q_rotate(facing_inv, relative_positions[:, i, :])

        return local_positions

    def _integrate_angular_velocity(self, angular_velocity: np.ndarray) -> np.ndarray:
        """Integrate angular velocity to get heading angles.

        Args:
            angular_velocity: Angular velocities, shape (T, 1)

        Returns:
            Heading angles, shape (T,)
        """
        # Angular velocity is delta angle per frame
        # Integrate starting from 0
        angles = np.zeros(angular_velocity.shape[0])
        angles[0] = 0.0  # Start at zero heading

        # Cumulative sum of angular velocities
        for i in range(1, len(angles)):
            angles[i] = angles[i - 1] + angular_velocity[i - 1, 0]

        return angles

    def _integrate_linear_velocity(self, linear_velocity: np.ndarray) -> np.ndarray:
        """Integrate linear velocity to get positions.

        Args:
            linear_velocity: Linear velocities, shape (T, 3)

        Returns:
            Positions, shape (T, 3)
        """
        # Linear velocity is displacement per frame
        # Integrate starting from origin
        positions = np.zeros_like(linear_velocity)
        positions[0] = np.array([0.0, 0.0, 0.0])  # Start at origin

        # Cumulative sum of velocities
        for i in range(1, len(positions)):
            positions[i] = positions[i - 1] + linear_velocity[i - 1]

        return positions

    def _angles_to_y_quaternions(self, angles: np.ndarray) -> np.ndarray:
        """Convert Y-axis rotation angles to quaternions.

        Args:
            angles: Rotation angles around Y-axis, shape (T,)

        Returns:
            Quaternions, shape (T, 4) in (w, x, y, z) format
        """
        num_frames = angles.shape[0]
        quaternions = np.zeros((num_frames, 4))

        half_angles = angles / 2.0
        quaternions[:, 0] = np.cos(half_angles)  # w
        quaternions[:, 2] = np.sin(half_angles)  # y

        return quaternions
