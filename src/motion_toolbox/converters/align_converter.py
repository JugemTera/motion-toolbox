"""ALIGN (232D) feature representation converter.

Based on: "StableMotion: Training Motion Cleanup Models with
          Unpaired Corrupted Data" (arXiv:2505.03154)

This converter transforms SMPL motion data into StableMotion's ALIGN
feature space, which is rotation-invariant and optimized for ML applications.
"""

import logging

import numpy as np
import torch

from ..core.motion import Motion
from ..core.skeleton import Skeleton
from ..utils.euler_utils import axis_angle_rotation, matrix_to_euler_zyx
from ..utils.quaternion import axis_angle_to_matrix, matrix_to_axis_angle
from ..utils.rotation_6d import matrix_to_rotation_6d, rotation_6d_to_matrix
from .base_converter import BaseMotionConverter

logger = logging.getLogger(__name__)


class ALIGNConverter(BaseMotionConverter):
    """Converter for StableMotion's ALIGN feature representation (232D).

    Features:
        - Canonically aligned (rotation-invariant)
        - 6D continuous rotation representation
        - Explicit foot dynamics
        - Motion velocities

    The ALIGN representation is designed for SMPL body models (22 body joints).

    Examples:
        >>> import numpy as np
        >>> from motion_toolbox import Motion
        >>> from motion_toolbox.converters import ALIGNConverter
        >>>
        >>> # Convert Motion to ALIGN features
        >>> converter = ALIGNConverter()
        >>> motion = Motion.from_file("motion.npy")
        >>> features = converter.to_align(motion)  # [T, 232]
        >>>
        >>> # Convert ALIGN features back to Motion
        >>> motion_reconstructed = converter.from_align(features, fps=20.0)
    """

    input_format = "motion"
    output_format = "align"
    FEATURE_DIM = 232
    FOOT_JOINT_INDICES = [7, 10, 6, 9]  # StableMotion: [left_ankle, right_ankle, left_foot, right_foot] (SMPL indices)

    def __init__(self, body_model_path: str | None = None):
        """Initialize ALIGN converter.

        Args:
            body_model_path: Path to SMPL body models (required for FK computation)
        """
        super().__init__()
        self.body_model_path = body_model_path

    def convert(self, data):
        """Convert data to/from ALIGN representation.

        Auto-detects direction based on input type:
        - Motion object → ALIGN features (to_align)
        - numpy array [T, 232] → Motion (from_align)

        Args:
            data: Motion object or ALIGN features array

        Returns:
            ALIGN features or Motion object
        """
        if isinstance(data, Motion):
            return self.to_align(data)
        elif isinstance(data, np.ndarray) and data.shape[-1] == self.FEATURE_DIM:
            return self.from_align(data)
        else:
            raise ValueError(
                f"Expected Motion object or array with shape [..., {self.FEATURE_DIM}], "
                f"got {type(data)} with shape {getattr(data, 'shape', 'N/A')}"
            )

    def to_align(self, motion: Motion) -> np.ndarray:
        """Convert Motion to ALIGN features.

        Args:
            motion: Motion object with SMPL-compatible skeleton

        Returns:
            features: [T, 232] numpy array of ALIGN features

        Raises:
            ValueError: If motion doesn't have required SMPL structure
        """
        # Extract SMPL parameters from motion
        smpl_data = self._extract_smpl_from_motion(motion)

        # Convert to torch tensors (ALIGN uses double precision)
        smpl_tensors = {
            "poses": torch.from_numpy(smpl_data["poses"]).double(),
            "trans": torch.from_numpy(smpl_data["trans"]).double(),
            "joints": torch.from_numpy(smpl_data["joints"]).double(),
        }

        # Convert to ALIGN features
        features = self._smpl_to_align_features(smpl_tensors)

        return features.cpu().numpy()

    def from_align(self, features: np.ndarray, fps: float = 20.0) -> Motion:
        """Convert ALIGN features to Motion.

        Args:
            features: [T, 232] numpy array of ALIGN features
            fps: Frame rate for Motion object (default: 20.0)

        Returns:
            Motion object with SMPL skeleton

        Raises:
            ValueError: If features have wrong shape
        """
        if features.shape[-1] != self.FEATURE_DIM:
            raise ValueError(f"Expected features with last dimension {self.FEATURE_DIM}, got {features.shape}")

        features_torch = torch.from_numpy(features).double()
        smpl_data = self._align_features_to_smpl(features_torch)

        # Convert to Motion
        motion = self._create_motion_from_smpl(smpl_data, fps=fps)

        return motion

    def _smpl_to_align_features(self, smpl_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Core conversion: SMPL → ALIGN features (232D).

        Implements the algorithm from StableMotion paper.

        Args:
            smpl_data: Dict with 'poses' [T, 66], 'trans' [T, 3], 'joints' [T, 24, 3]

        Returns:
            features: [T, 232] tensor
        """
        poses = smpl_data["poses"].clone()  # [T, 66]
        trans = smpl_data["trans"].clone()  # [T, 3]
        joints = smpl_data["joints"].clone()  # [T, 24, 3]

        T = poses.shape[0]

        # =============================================================
        # STEP 1: Ground Removal & Root Extraction
        # =============================================================
        ground = joints[:, :, 2].min()
        joints[:, :, 2] -= ground
        root_grav_axis = joints[:, 0, 2].clone()  # [T]

        # =============================================================
        # STEP 2: Trajectory Extraction
        # =============================================================
        trajectory = joints[:, 0, :2].clone()  # [T, 2]

        # Center joints at pelvis (XY)
        joints[:, :, :2] -= trajectory.unsqueeze(1)
        joints[:, :, 2] -= joints[:, 0:1, 2]

        # Remove pelvis (now all zeros)
        joints = joints[:, 1:]  # [T, 23, 3]

        # Shift trajectory to origin
        trajectory = trajectory - trajectory[0]

        # =============================================================
        # STEP 3: Global Rotation Decomposition (ZYX Euler)
        # =============================================================
        poses_reshaped = poses.reshape(T, 22, 3)

        # Convert axis-angle to rotation matrices
        poses_mat = self._axis_angle_to_matrix_torch(poses_reshaped)  # [T, 22, 3, 3]
        global_orient = poses_mat[:, 0]  # [T, 3, 3]

        # Decompose into Z, Y, X rotations
        global_euler = matrix_to_euler_zyx(global_orient)
        rotZ_angle = global_euler[:, 0]
        rotY_angle = global_euler[:, 1]
        rotX_angle = global_euler[:, 2]

        # Build rotation matrices
        og_rotZ = axis_angle_rotation("Z", rotZ_angle)
        rotY = axis_angle_rotation("Y", rotY_angle)
        rotX = axis_angle_rotation("X", rotX_angle)

        # =============================================================
        # STEP 4: Canonical Frame Alignment
        # =============================================================
        # Compute first frame Z rotation (for alignment)
        init_rotZ = axis_angle_rotation("Z", rotZ_angle[0:1])  # [1, 3, 3]
        init_rotZ = init_rotZ.repeat(T, 1, 1)  # [T, 3, 3]

        # Align trajectory to canonical frame
        trajectory = torch.einsum("tkj,tk->tj", init_rotZ[:, :2, :2], trajectory)

        # Make Z rotation relative to first frame
        rotZ_angle = rotZ_angle - rotZ_angle[0]
        rotZ = axis_angle_rotation("Z", rotZ_angle)

        # Extract 2D representation (cos, sin)
        rotZ_2d = rotZ[:, :2, 0]  # [T, 2]

        # =============================================================
        # STEP 5: Rotate Joints to Local Frame
        # =============================================================
        # Rotate XY components by original Z rotation
        joints_xy = torch.einsum("tkj,tlk->tlj", og_rotZ[:, :2, :2], joints[:, :, :2])
        joints_local = torch.stack([joints_xy[..., 0], joints_xy[..., 1], joints[..., 2]], dim=-1)  # [T, 23, 3]

        # =============================================================
        # STEP 6: Local Pose Representation (6D)
        # =============================================================
        # Global orientation without Z rotation
        global_orient_local = rotY @ rotX  # [T, 3, 3]

        # Combine with body poses
        poses_mat_local = torch.cat([global_orient_local.unsqueeze(1), poses_mat[:, 1:]], dim=1)  # [T, 22, 3, 3]

        # Convert to 6D representation
        poses_local = matrix_to_rotation_6d(poses_mat_local)  # [T, 22, 6]

        # =============================================================
        # STEP 7: Foot Features (Global Coordinates)
        # =============================================================
        foot_joints = joints_local[:, self.FOOT_JOINT_INDICES]  # [T, 4, 3]

        # Transform to global coordinates
        foot_global = foot_joints.clone()
        foot_global[:, :, :2] += trajectory.unsqueeze(1)
        foot_global[:, :, 2] += root_grav_axis.unsqueeze(1)

        # =============================================================
        # STEP 8: Velocity Computation
        # =============================================================
        vel_traj = self._compute_velocity(trajectory)  # [T, 2]
        vel_foot_global = self._compute_velocity(foot_global)  # [T, 4, 3]

        # =============================================================
        # STEP 9: Pack Features
        # =============================================================
        features = torch.cat(
            [
                root_grav_axis.unsqueeze(-1),  # [T, 1]
                trajectory,  # [T, 2]
                rotZ_2d,  # [T, 2]
                poses_local.reshape(T, -1),  # [T, 132]
                joints_local.reshape(T, -1),  # [T, 69]
                foot_global.reshape(T, -1),  # [T, 12]
                vel_foot_global.reshape(T, -1),  # [T, 12]
                vel_traj,  # [T, 2]
            ],
            dim=-1,
        )

        assert features.shape[-1] == 232, f"Expected 232 features, got {features.shape[-1]}"
        return features

    def _align_features_to_smpl(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Core conversion: ALIGN features (232D) → SMPL.

        Implements the inverse algorithm from StableMotion.

        Args:
            features: [T, 232] tensor

        Returns:
            Dict with 'poses' [T, 66], 'trans' [T, 3], 'joints' [T, 24, 3]
        """
        T = features.shape[0]

        # =============================================================
        # STEP 1: Unpack Features
        # =============================================================
        root_grav_axis = features[:, 0]  # [T]
        trajectory = features[:, 1:3]  # [T, 2]
        rotZ_2D = features[:, 3:5]  # [T, 2]
        poses_local = features[:, 5:137].reshape(T, 22, 6)  # [T, 22, 6]
        joints_local = features[:, 137:206].reshape(T, 23, 3)  # [T, 23, 3]
        foot_global = features[:, 206:218].reshape(T, 4, 3)  # [T, 4, 3]
        vel_foot_global = features[:, 218:230].reshape(T, 4, 3)  # [T, 4, 3]
        vel_traj = features[:, 230:232]  # [T, 2]

        # =============================================================
        # STEP 2: Reconstruct Z Rotation from 2D
        # =============================================================
        # Normalize to unit vector
        rotZ_2D = rotZ_2D / torch.norm(rotZ_2D, dim=-1, keepdim=True)
        cos = rotZ_2D[:, 0]
        sin = rotZ_2D[:, 1]

        # Build 3x3 rotation matrix
        zero = torch.zeros_like(cos)
        one = torch.ones_like(cos)
        rotZ = torch.stack([cos, -sin, zero, sin, cos, zero, zero, zero, one], dim=-1).reshape(T, 3, 3)

        # =============================================================
        # STEP 3: Rotate Joints Back to Global Frame
        # =============================================================
        joints_xy = torch.einsum("bjk,blk->blj", rotZ[:, :2, :2], joints_local[:, :, :2])
        joints = torch.stack([joints_xy[..., 0], joints_xy[..., 1], joints_local[..., 2]], dim=-1)  # [T, 23, 3]

        # Add pelvis joint (zeros)
        joints = torch.cat(
            [torch.zeros(T, 1, 3, dtype=joints.dtype, device=joints.device), joints], dim=1
        )  # [T, 24, 3]

        # Add root height and trajectory
        joints[:, :, 2] += root_grav_axis.unsqueeze(-1)
        joints[:, :, :2] += trajectory.unsqueeze(1)

        # =============================================================
        # STEP 4: Reconstruct Translation
        # =============================================================
        trans = torch.cat([trajectory, root_grav_axis.unsqueeze(-1)], dim=-1)  # [T, 3]

        # =============================================================
        # STEP 5: Reconstruct Global Orientation
        # =============================================================
        # Convert 6D poses to rotation matrices
        poses_mat_local = rotation_6d_to_matrix(poses_local)  # [T, 22, 3, 3]
        global_orient_local = poses_mat_local[:, 0]

        # Extract Y and X angles
        global_euler_local = matrix_to_euler_zyx(global_orient_local)
        rotY_angle = global_euler_local[:, 1]
        rotX_angle = global_euler_local[:, 2]

        rotY = axis_angle_rotation("Y", rotY_angle)
        rotX = axis_angle_rotation("X", rotX_angle)

        # Reconstruct full global orientation: Z * Y * X
        global_orient = rotZ @ rotY @ rotX  # [T, 3, 3]

        # =============================================================
        # STEP 6: Combine Poses
        # =============================================================
        poses_mat = torch.cat([global_orient.unsqueeze(1), poses_mat_local[:, 1:]], dim=1)  # [T, 22, 3, 3]

        # Convert to axis-angle
        poses = self._matrix_to_axis_angle_torch(poses_mat)  # [T, 22, 3]
        poses = poses.reshape(T, 66)

        return {"poses": poses, "trans": trans, "joints": joints}

    @staticmethod
    def _compute_velocity(data: torch.Tensor) -> torch.Tensor:
        """Compute forward differences with extrapolation for last frame.

        This matches StableMotion's my_diff() function.

        Args:
            data: [T, ...] input data

        Returns:
            velocity: [T, ...] velocities (same shape as input)
        """
        vel = torch.diff(data, dim=0)  # [T-1, ...]

        # Extrapolate last frame using acceleration
        if vel.shape[0] >= 2:
            acceleration = vel[-1] - vel[-2]
            future_vel = vel[-1] + acceleration
        else:
            future_vel = vel[-1] if vel.shape[0] > 0 else torch.zeros_like(data[0])

        vel = torch.cat([vel, future_vel.unsqueeze(0)], dim=0)  # [T, ...]
        return vel

    def _extract_smpl_from_motion(self, motion: Motion) -> dict[str, np.ndarray]:
        """Extract SMPL parameters from Motion object.

        Args:
            motion: Motion object with SMPL-compatible skeleton

        Returns:
            Dict with 'poses' [T, 66], 'trans' [T, 3], 'joints' [T, 24, 3]
        """
        # Get joint positions via forward kinematics
        joints = motion.positions  # [T, num_joints, 3]

        # Extract rotation parameters (axis-angle)
        joint_rotations = motion.joint_rotations  # [T, num_joints, 3]

        # SMPL format: 22 joints (root + 21 body joints)
        # Use first 22 joints if skeleton has more
        num_joints = min(22, joint_rotations.shape[1])

        # Flatten rotations to match SMPL format [T, 66]
        poses = joint_rotations[:, :num_joints, :].reshape(-1, num_joints * 3)

        # Pad to 66 if needed (22 * 3 = 66)
        if poses.shape[1] < 66:
            padding = np.zeros((poses.shape[0], 66 - poses.shape[1]))
            poses = np.concatenate([poses, padding], axis=1)

        # Translation
        trans = motion.root_translation  # [T, 3]

        # Ensure we have 24 joints for SMPL (pad if necessary)
        if joints.shape[1] < 24:
            padding = np.zeros((joints.shape[0], 24 - joints.shape[1], 3))
            joints = np.concatenate([joints, padding], axis=1)
        else:
            joints = joints[:, :24, :]

        return {"poses": poses, "trans": trans, "joints": joints}

    def _create_motion_from_smpl(self, smpl_data: dict[str, torch.Tensor], fps: float) -> Motion:
        """Create Motion object from SMPL parameters.

        Args:
            smpl_data: Dict with 'poses' [T, 66], 'trans' [T, 3], 'joints' [T, 24, 3]
            fps: Frame rate

        Returns:
            Motion object
        """
        # Convert to numpy
        poses = smpl_data["poses"].cpu().numpy()  # [T, 66]
        trans = smpl_data["trans"].cpu().numpy()  # [T, 3]
        joints = smpl_data["joints"].cpu().numpy() if "joints" in smpl_data else None

        # Reshape poses to [T, 22, 3] and pad to 24 joints for SMPL skeleton
        joint_rotations_22 = poses.reshape(-1, 22, 3)

        # Pad with zeros for hand joints (joints 22-23)
        num_frames = joint_rotations_22.shape[0]
        joint_rotations = np.zeros((num_frames, 24, 3))
        joint_rotations[:, :22, :] = joint_rotations_22

        # Create SMPL skeleton (24 joints)
        skeleton = Skeleton.from_smpl()
        # ALIGN reconstructs SMPL parameters in Z-up coordinate system.
        # Transform translation from Z-up to Y-up: [X, Y, Z] → [X, Z, -Y]
        trans_yup = trans.copy()
        trans_yup[:, [1, 2]] = trans_yup[:, [2, 1]]  # Swap Y and Z
        trans_yup[:, 2] = -trans_yup[:, 2]  # Negate new Z

        # Keep joint rotations as-is - ALIGN outputs SMPL-compatible rotations
        # The body pose rotations are in local joint space and don't need transformation
        joint_rotations_yup = joint_rotations.copy()

        # Create Motion object in Y-up (SMPL's expected coordinate system)
        motion = Motion(
            joint_rotations=joint_rotations_yup,
            root_translation=trans_yup,
            skeleton=skeleton,
            framerate=fps,
            up_axis="y",  # Now Y-up
            body_model_path=self.body_model_path,
            floor_alignment="none",
        )

        # Apply post-correction for floor alignment (Y-up now)
        floor_offset = motion.positions[:, :, 1].min()
        if floor_offset < 0:
            corrected_trans = motion.root_translation.copy()
            corrected_trans[:, 1] -= floor_offset  # Adjust Y axis

            motion = Motion(
                joint_rotations=motion.joint_rotations,
                root_translation=corrected_trans,
                skeleton=motion.skeleton,
                framerate=fps,
                up_axis="y",
                body_model_path=self.body_model_path,
                floor_alignment="none",
            )

        return motion

    @staticmethod
    def _axis_angle_to_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to rotation matrix (torch version).

        Args:
            axis_angle: [..., 3] axis-angle rotations

        Returns:
            matrices: [..., 3, 3] rotation matrices
        """
        # Use scipy via numpy for reliability
        aa_np = axis_angle.detach().cpu().numpy()

        # Handle batch dimensions - reshape to (N, 3) for scipy
        original_shape = aa_np.shape
        aa_flat = aa_np.reshape(-1, 3)

        mat_flat = axis_angle_to_matrix(aa_flat)
        mat_np = mat_flat.reshape(*original_shape[:-1], 3, 3)

        return torch.from_numpy(mat_np).to(axis_angle.device).type(axis_angle.dtype)

    @staticmethod
    def _matrix_to_axis_angle_torch(matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to axis-angle (torch version).

        Args:
            matrix: [..., 3, 3] rotation matrices

        Returns:
            axis_angle: [..., 3] axis-angle rotations
        """
        # Use scipy via numpy for reliability
        mat_np = matrix.detach().cpu().numpy()

        # Handle batch dimensions - reshape to (N, 3, 3) for scipy
        original_shape = mat_np.shape
        mat_flat = mat_np.reshape(-1, 3, 3)

        aa_flat = matrix_to_axis_angle(mat_flat)
        aa_np = aa_flat.reshape(*original_shape[:-2], 3)

        return torch.from_numpy(aa_np).to(matrix.device).type(matrix.dtype)

    # =========================================================================
    # Feature Extraction API (for analysis)
    # =========================================================================

    def extract_root_height(self, features: np.ndarray) -> np.ndarray:
        """Extract root vertical position (gravity axis).

        Args:
            features: [T, 232] ALIGN features

        Returns:
            root_height: [T] vertical position
        """
        return features[:, 0]

    def extract_trajectory(self, features: np.ndarray) -> np.ndarray:
        """Extract root XY trajectory (canonical frame).

        Args:
            features: [T, 232] ALIGN features

        Returns:
            trajectory: [T, 2] XY trajectory
        """
        return features[:, 1:3]

    def extract_heading(self, features: np.ndarray) -> np.ndarray:
        """Extract 2D heading direction (canonical frame).

        Args:
            features: [T, 232] ALIGN features

        Returns:
            heading: [T, 2] heading as (cos, sin)
        """
        return features[:, 3:5]

    def extract_poses_6d(self, features: np.ndarray) -> np.ndarray:
        """Extract body poses in 6D representation.

        Args:
            features: [T, 232] ALIGN features

        Returns:
            poses_6d: [T, 22, 6] body poses
        """
        return features[:, 5:137].reshape(-1, 22, 6)

    def extract_joints_local(self, features: np.ndarray) -> np.ndarray:
        """Extract joint positions (pelvis-relative, canonical frame).

        Args:
            features: [T, 232] ALIGN features

        Returns:
            joints: [T, 23, 3] joint positions
        """
        return features[:, 137:206].reshape(-1, 23, 3)

    def extract_foot_positions(self, features: np.ndarray) -> np.ndarray:
        """Extract global foot positions.

        Args:
            features: [T, 232] ALIGN features

        Returns:
            foot_pos: [T, 4, 3] foot positions
        """
        return features[:, 206:218].reshape(-1, 4, 3)

    def extract_foot_velocities(self, features: np.ndarray) -> np.ndarray:
        """Extract foot velocities.

        Args:
            features: [T, 232] ALIGN features

        Returns:
            foot_vel: [T, 4, 3] foot velocities
        """
        return features[:, 218:230].reshape(-1, 4, 3)

    def extract_trajectory_velocity(self, features: np.ndarray) -> np.ndarray:
        """Extract root trajectory velocity.

        Args:
            features: [T, 232] ALIGN features

        Returns:
            traj_vel: [T, 2] trajectory velocity
        """
        return features[:, 230:232]

    # =========================================================================
    # Analysis Utilities
    # =========================================================================

    def compute_foot_contacts(self, features: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        """Detect foot-ground contacts from velocities.

        Args:
            features: [T, 232] ALIGN features
            threshold: Velocity threshold for contact (m/s, default: 0.05)

        Returns:
            contacts: [T, 4] boolean array (True = contact)
        """
        foot_vel = self.extract_foot_velocities(features)  # [T, 4, 3]
        foot_speed = np.linalg.norm(foot_vel, axis=-1)  # [T, 4]
        contacts = foot_speed < threshold
        return contacts

    def detect_sliding(
        self, features: np.ndarray, contact_threshold: float = 0.05, slide_threshold: float = 0.02
    ) -> np.ndarray:
        """Detect foot sliding (contact + high velocity).

        Args:
            features: [T, 232] ALIGN features
            contact_threshold: Velocity threshold for contact (default: 0.05)
            slide_threshold: Velocity threshold for sliding (default: 0.02)

        Returns:
            sliding: [T, 4] boolean array (True = sliding)
        """
        foot_vel = self.extract_foot_velocities(features)
        foot_speed = np.linalg.norm(foot_vel, axis=-1)

        # Foot is in contact but moving
        is_contact = foot_speed < contact_threshold
        is_moving = foot_speed > slide_threshold
        sliding = is_contact & is_moving

        return sliding
