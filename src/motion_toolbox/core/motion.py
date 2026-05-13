"""Motion data structure for kinematic animation.

This module provides the Motion class that combines skeleton structure with
temporal motion data (rotations, translations) and provides forward kinematics.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from ..utils.quaternion import axis_angle_to_quaternion, normalize_quaternion, quaternion_to_axis_angle
from .forward_kinematics import ForwardKinematics, create_forward_kinematics
from .identity import Identity
from .skeleton import Skeleton


@dataclass(frozen=True)
class Motion:
    """Immutable motion representation with forward kinematics.

    This class combines a skeleton with temporal motion data, storing joint rotations
    as axis-angles (3D vectors where magnitude = rotation angle, direction = rotation axis).
    This representation is more natural for most motion datasets (SMPL, GVHMR, CMU, BVH)
    and avoids conversion errors.

    Args:
        joint_rotations: Joint rotations as axis-angles, shape (num_frames, num_joints, 3)
                        Each vector's magnitude is the rotation angle in radians,
                        and direction is the rotation axis
        root_translation: Root joint translation over time, shape (num_frames, 3)
        skeleton: Skeleton defining the kinematic structure
        framerate: Frames per second (default: 30.0)
        up_axis: Up axis direction - 'y' for Y-up, 'z' for Z-up (default: 'y')
        identity: Optional Identity object containing person-specific attributes like body shape (default: None)
        forward_kinematics: FK implementation instance, method name ("auto", "custom", "smpl"), or None for auto (default: None)
        body_model_path: Path to body models directory (required for SMPL forward kinematics) (default: None)
        floor_alignment: Floor alignment method - "joint" (based on foot joints), "mesh" (based on foot vertices), or "none" (default: "joint")
        model_data: Additional model-specific data (e.g., MHR scalings, face coefficients)
        body_model_type: Identifier for the source body model (e.g., "smpl", "mhr")

    Examples:
        >>> import numpy as np
        >>> import motion_toolbox as mt
        >>>
        >>> # Create a simple motion
        >>> skeleton = mt.create_humanml3d_skeleton()
        >>> num_frames, num_joints = 120, skeleton.num_joints
        >>>
        >>> # Small random rotations (< 0.1 radians)
        >>> rotations = 0.1 * np.random.randn(num_frames, num_joints, 3)
        >>> # Stationary at origin
        >>> translation = np.zeros((num_frames, 3))
        >>>
        >>> body_model_path = "/path/to/body_models"
        >>> motion = Motion(rotations, translation, skeleton, framerate=30.0,
        ...                body_model_path=body_model_path)
        >>> positions = motion.positions  # Cached forward kinematics
        >>> print(f"Motion: {motion.num_frames} frames, {motion.num_joints} joints")
        >>>
        >>> # Create motion with SMPL shape parameters
        >>> identity = Identity(betas=np.array([0.5, -0.2, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]), gender="female")
        >>> motion_with_identity = Motion(rotations, translation, skeleton,
        ...                              framerate=30.0, identity=identity, body_model_path=body_model_path)
    """

    joint_rotations: np.ndarray
    root_translation: np.ndarray
    skeleton: Skeleton
    framerate: float = 30.0
    up_axis: str = "y"
    identity: Identity | None = None  # Person-specific attributes (body shape, gender, etc.)
    forward_kinematics: ForwardKinematics | str | None = None  # FK implementation or method name
    body_model_path: str | None = None  # Path to body models directory
    floor_alignment: str = "joint"  # Floor alignment method: "joint", "mesh", or "none"
    model_data: dict = field(default_factory=dict)  # Additional model-specific data
    body_model_type: str | None = None  # Identifier for the source body model

    # Optional dense velocity data (e.g. from interpolation or editing)
    # Root linear velocity (num_frames, 3) in m/s
    root_linear_velocity: np.ndarray | None = None
    # Joint angular velocities (num_frames, num_joints, 3) in rad/s (axis-angle derivative)
    joint_angular_velocity: np.ndarray | None = None

    # Private cached properties (set by __post_init__)
    _quaternions: np.ndarray | None = None
    _positions: np.ndarray | None = None
    _velocities: np.ndarray | None = None
    _accelerations: np.ndarray | None = None
    _fk_impl: ForwardKinematics | None = None

    def __post_init__(self):
        """Validate motion data consistency."""
        # Validate joint_rotations shape
        if self.joint_rotations.ndim != 3 or self.joint_rotations.shape[2] != 3:
            raise ValueError(
                f"joint_rotations must have shape (num_frames, num_joints, 3), got {self.joint_rotations.shape}"
            )

        # Validate root_translation shape
        if self.root_translation.ndim != 2 or self.root_translation.shape[1] != 3:
            raise ValueError(f"root_translation must have shape (num_frames, 3), got {self.root_translation.shape}")

        # Check that number of frames match
        num_frames_rot = self.joint_rotations.shape[0]
        num_frames_trans = self.root_translation.shape[0]
        if num_frames_rot != num_frames_trans:
            raise ValueError(f"Number of frames must match: rotations={num_frames_rot}, translation={num_frames_trans}")

        # Check that number of joints match skeleton
        num_joints_rot = self.joint_rotations.shape[1]
        if num_joints_rot != self.skeleton.num_joints:
            raise ValueError(
                f"Number of joints must match skeleton: rotations={num_joints_rot}, skeleton={self.skeleton.num_joints}"
            )

        # Validate framerate
        if self.framerate <= 0:
            raise ValueError(f"Framerate must be positive, got {self.framerate}")

        # Validate up_axis
        if self.up_axis.lower() not in ["y", "z"]:
            raise ValueError(f"up_axis must be 'y' or 'z', got {self.up_axis}")

        # Validate identity if provided
        if self.identity is not None and not isinstance(self.identity, Identity):
            raise ValueError(f"identity must be an Identity object, got {type(self.identity)}")

        # Set up forward kinematics implementation
        if self.forward_kinematics is None:
            fk_impl = create_forward_kinematics("auto", model_path=self.body_model_path)
        elif isinstance(self.forward_kinematics, str):
            fk_impl = create_forward_kinematics(self.forward_kinematics, model_path=self.body_model_path)
        else:
            fk_impl = self.forward_kinematics
        object.__setattr__(self, "_fk_impl", fk_impl)

        # Validate velocities if provided
        if self.root_linear_velocity is not None:
            if self.root_linear_velocity.shape != self.root_translation.shape:
                raise ValueError(
                    f"root_linear_velocity shape mismatch: {self.root_linear_velocity.shape} vs {self.root_translation.shape}"
                )

        if self.joint_angular_velocity is not None:
            if self.joint_angular_velocity.shape != self.joint_rotations.shape:
                raise ValueError(
                    f"joint_angular_velocity shape mismatch: {self.joint_angular_velocity.shape} vs {self.joint_rotations.shape}"
                )

        # Check for reasonable axis-angle magnitudes (warn if very large rotations)
        axis_angle_magnitudes = np.linalg.norm(self.joint_rotations, axis=2)
        max_magnitude = np.max(axis_angle_magnitudes)
        if max_magnitude > 10.0:  # > ~570 degrees
            warnings.warn(
                f"Large axis-angle rotations detection (max: {max_magnitude:.2f} rad). "
                "This may indicate incorrect data or units.",
                UserWarning,
            )

        # Cache quaternions
        object.__setattr__(self, "_quaternions", self._compute_quaternions())

        # Compute initial positions
        initial_positions = self._compute_positions()

        # Apply floor alignment if requested
        if self.floor_alignment == "none":
            object.__setattr__(self, "_positions", initial_positions)
        else:
            adjusted_positions, adjusted_root, floor_offset = self._apply_floor_alignment(initial_positions)
            object.__setattr__(self, "root_translation", adjusted_root)
            object.__setattr__(self, "_positions", adjusted_positions)

        # Make arrays read-only
        self.joint_rotations.flags.writeable = False
        self.root_translation.flags.writeable = False

    def _compute_quaternions(self) -> np.ndarray:
        """Compute quaternions from axis-angle rotations.

        Returns:
            Joint rotations as quaternions, shape (num_frames, num_joints, 4)
        """
        num_frames, num_joints = self.joint_rotations.shape[:2]
        quaternions = np.zeros((num_frames, num_joints, 4))

        # Reshape for batch conversion
        axis_angles_flat = self.joint_rotations.reshape(-1, 3)
        quaternions_flat = axis_angle_to_quaternion(axis_angles_flat)
        quaternions = quaternions_flat.reshape(num_frames, num_joints, 4)

        return quaternions

    def _compute_positions(self) -> np.ndarray:
        """Compute joint positions using forward kinematics.

        Returns:
            Joint positions, shape (num_frames, num_joints, 3)
        """
        # Use the configured FK implementation (should be set in __post_init__)
        assert self._fk_impl is not None, "FK implementation not initialized"

        # Pass axis-angles directly to FK implementation
        return self._fk_impl.compute_positions(
            joint_rotations=self.joint_rotations,
            root_translation=self.root_translation,
            skeleton=self.skeleton,
            up_axis=self.up_axis,
            identity=self.identity,
        )

    def _compute_velocities(self) -> np.ndarray:
        """Compute joint velocities using finite differences from positions.

        Uses central differences for interior frames and forward/backward differences
        for the first and last frames respectively.

        Returns:
            Joint velocities, shape (num_frames, num_joints, 3)
        """
        positions = self.positions  # This will use cached positions if available
        num_frames = positions.shape[0]
        dt = 1.0 / self.framerate

        velocities = np.zeros_like(positions)

        if num_frames == 1:
            # Single frame: velocity is zero
            return velocities
        elif num_frames == 2:
            # Two frames: use forward difference for both
            velocities[0] = (positions[1] - positions[0]) / dt
            velocities[1] = (positions[1] - positions[0]) / dt
        else:
            # Forward difference for first frame
            velocities[0] = (positions[1] - positions[0]) / dt

            # Central difference for interior frames
            for i in range(1, num_frames - 1):
                velocities[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)

            # Backward difference for last frame
            velocities[-1] = (positions[-1] - positions[-2]) / dt

        return velocities

    def _compute_accelerations(self) -> np.ndarray:
        """Compute joint accelerations using finite differences from positions.

        Uses second-order finite differences: acceleration = (x[t+1] - 2*x[t] + x[t-1]) / dt²
        For edge frames, uses available data with appropriate finite difference schemes.

        Returns:
            Joint accelerations, shape (num_frames, num_joints, 3)
        """
        positions = self.positions  # This will use cached positions if available
        num_frames = positions.shape[0]
        dt = 1.0 / self.framerate

        accelerations = np.zeros_like(positions)

        if num_frames < 3:
            # Need at least 3 frames to compute acceleration
            return accelerations
        else:
            # Use central differences for interior frames: (x[t+1] - 2*x[t] + x[t-1]) / dt²
            for i in range(1, num_frames - 1):
                accelerations[i] = (positions[i + 1] - 2 * positions[i] + positions[i - 1]) / (dt**2)

            # For first and last frames, use forward/backward differences
            # First frame: approximate using first valid acceleration
            accelerations[0] = accelerations[1]
            # Last frame: approximate using last valid acceleration
            accelerations[-1] = accelerations[-2]

        return accelerations

    def _apply_floor_alignment(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Apply floor alignment and return adjusted root translation, adjusted joint positions and floor offset.

        Args:
            positions: Joint positions, shape (num_frames, num_joints, 3)

        Returns:
            Tuple of (adjusted_positions, adjusted_root_translation, floor_offset)
            - adjusted_positions: shape (num_frames, num_joints, 3)
            - adjusted_root_translation: shape (num_frames, 3)
            - floor_offset: Vertical offset applied to align to floor

        """
        up_idx = 1 if self.up_axis.lower() == "y" else 2  # Y-up: index 1, Z-up: index 2

        if self.floor_alignment == "joint":
            min_up_value = self._get_joint_floor_level(positions, up_idx)
        elif self.floor_alignment == "mesh":
            min_up_value = self._get_mesh_floor_level(up_idx)
        else:
            raise ValueError(f"Unsupported floor alignment method: {self.floor_alignment}")

        # Calculate the offset needed to move character to floor
        floor_offset = min_up_value  # Subtract the minimum value to bring lowest point to 0

        # Adjust root translation to move character to floor
        adjusted_root_translation = self.root_translation.copy()
        adjusted_root_translation[:, up_idx] -= floor_offset

        adjusted_positions = positions.copy()
        adjusted_positions[:, :, up_idx] -= floor_offset
        return adjusted_positions, adjusted_root_translation, floor_offset

    def _get_joint_floor_level(self, positions: np.ndarray, up_idx: int) -> float:
        """Get the minimum floor level based on foot joint positions.

        Args:
            positions: Joint positions, shape (num_frames, num_joints, 3)
            up_idx: Index of the up axis

        Returns:
            Minimum up-axis value for foot joints
        """
        # Prefer SMPL-style foot joints when present; otherwise align using the lowest joint.
        try:
            left_foot_idx = self.skeleton.get_joint_index("left_foot")
            right_foot_idx = self.skeleton.get_joint_index("right_foot")
        except ValueError:
            return float(positions[:, :, up_idx].min())

        return positions[:, [left_foot_idx, right_foot_idx], up_idx].min(axis=0).min()

    def _get_mesh_floor_level(self, up_idx: int) -> float:
        """Get the minimum floor level based on foot mesh vertices.

        Args:
            up_idx: Index of the up axis

        Returns:
            Minimum up-axis value for foot vertices
        """
        # Check if we need SMPL body models
        if self.body_model_path is None:
            raise ValueError(
                "Mesh-based floor alignment requires body_model_path for SMPL forward kinematics. Use floor_alignment='joint' instead."
            )

        # Get SMPL vertices for all frames
        vertices = self._get_smpl_vertices()

        # Find foot vertex indices (approximate foot region)
        # SMPL foot vertices are roughly indices 3387-3407 (left) and 6728-6748 (right)
        # These are the vertices at the bottom of the feet
        left_foot_vertex_indices = list(range(3387, 3408))  # 21 vertices
        right_foot_vertex_indices = list(range(6728, 6749))  # 21 vertices
        foot_vertex_indices = left_foot_vertex_indices + right_foot_vertex_indices

        # Get minimum up-axis value for foot vertices across all frames
        foot_vertices = vertices[:, foot_vertex_indices, :]  # Shape: (frames, 42, 3)
        min_up_value = foot_vertices[:, :, up_idx].min()

        return min_up_value

    def _get_smpl_vertices(self) -> np.ndarray:
        """Get SMPL vertices for all frames.

        Returns:
            SMPL vertices, shape (num_frames, num_vertices, 3)
        """
        import smplx
        import torch

        # Get SMPL parameters
        if self.identity and hasattr(self.identity, "betas") and self.identity.betas is not None:
            betas_np = self.identity.betas
            gender = getattr(self.identity, "gender", "neutral")
        else:
            betas_np = np.zeros(10)
            gender = "neutral"

        # Create SMPL model directly
        smpl_model = smplx.create(
            model_path=self.body_model_path,
            model_type="smpl",
            gender=gender,
            ext="pkl",
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        )

        # Prepare input data
        num_frames, num_joints = self.joint_rotations.shape[:2]

        global_orient = torch.from_numpy(self.joint_rotations[:, 0, :].copy()).float()

        # Body pose: joints 1-23 for SMPL (pad or trim as needed)
        num_body_joints = min(23, num_joints - 1)
        if num_body_joints > 0:
            body_pose_data = self.joint_rotations[:, 1 : 1 + num_body_joints, :].reshape(num_frames, -1)
        else:
            body_pose_data = np.zeros((num_frames, 0))

        # Pad to exactly 69 parameters (23 joints * 3)
        if body_pose_data.shape[1] < 69:
            padding = np.zeros((num_frames, 69 - body_pose_data.shape[1]))
            body_pose_data = np.concatenate([body_pose_data, padding], axis=1)
        else:
            body_pose_data = body_pose_data[:, :69]

        body_pose = torch.from_numpy(body_pose_data.copy()).float()
        betas = torch.from_numpy(betas_np.copy()).float().unsqueeze(0).repeat(num_frames, 1)
        transl = torch.from_numpy(self.root_translation.copy()).float()

        # Run SMPL forward pass to get vertices
        with torch.no_grad():
            smpl_output = smpl_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
            )

        # Extract vertices
        vertices = smpl_output.vertices.cpu().numpy()

        return vertices

    @property
    def num_frames(self) -> int:
        """Number of frames in the motion."""
        return self.joint_rotations.shape[0]

    @property
    def num_joints(self) -> int:
        """Number of joints in the motion."""
        return self.joint_rotations.shape[1]

    @property
    def duration(self) -> float:
        """Duration of motion in seconds."""
        return self.num_frames / self.framerate

    @property
    def fps(self) -> float:
        """Frames per second (alias for framerate)."""
        return self.framerate

    def get_frame(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get joint rotations and root translation for a specific frame.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            Tuple of (joint_rotations, root_translation) for the frame
            joint_rotations: shape (num_joints, 3) - axis-angle rotations
            root_translation: shape (3,)
        """
        if not (0 <= frame_idx < self.num_frames):
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        return self.joint_rotations[frame_idx], self.root_translation[frame_idx]

    @property
    def positions(self) -> np.ndarray:
        """Get joint positions (cached if enabled).

        Returns:
            Joint positions in world coordinates, shape (num_frames, num_joints, 3)
        """
        if self._positions is not None:
            return self._positions
        else:
            # Compute on-demand if not cached
            return self._compute_positions()

    @property
    def quaternions(self) -> np.ndarray:
        """Get quaternion rotations (cached if enabled).

        Returns:
            Joint rotations as quaternions, shape (num_frames, num_joints, 4)
            Quaternions in (w, x, y, z) format
        """
        if self._quaternions is not None:
            return self._quaternions
        else:
            # Compute on-demand if not cached
            return self._compute_quaternions()

    @property
    def velocities(self) -> np.ndarray:
        """Get joint velocities (cached if enabled).

        Velocities are computed using finite differences from joint positions.
        Uses central differences for interior frames and forward/backward differences
        for the first and last frames respectively.

        Returns:
            Joint velocities in world coordinates, shape (num_frames, num_joints, 3)
            Units: meters per second
        """
        if self._velocities is not None:
            return self._velocities
        else:
            # Compute on-demand if not cached
            return self._compute_velocities()

    @property
    def accelerations(self) -> np.ndarray:
        """Get joint accelerations (cached if enabled).

        Accelerations are computed using finite differences from joint positions.
        Uses central differences for interior frames where possible.

        Returns:
            Joint accelerations in world coordinates, shape (num_frames, num_joints, 3)
            Units: meters per second squared
        """
        if self._accelerations is not None:
            return self._accelerations
        else:
            # Compute on-demand if not cached
            return self._compute_accelerations()

    @classmethod
    def from_quaternions(
        cls,
        quaternion_rotations: np.ndarray,
        root_translation: np.ndarray,
        skeleton: Skeleton,
        framerate: float = 30.0,
        up_axis: str = "y",
        identity: Identity | None = None,
        forward_kinematics: ForwardKinematics | str | None = None,
        body_model_path: str | None = None,
        model_data: dict | None = None,
        body_model_type: str | None = None,
        floor_alignment: str = "joint",
    ) -> "Motion":
        """Create Motion from quaternion rotations.

        Args:
            quaternion_rotations: Joint rotations as quaternions, shape (num_frames, num_joints, 4)
            root_translation: Root translation, shape (num_frames, 3)
            skeleton: Skeleton defining kinematic structure
            framerate: Frames per second
            up_axis: Up axis direction
            identity: Optional Identity object
            forward_kinematics: FK implementation or method name
            body_model_path: Path to body models directory
            model_data: Additional model-specific data
            body_model_type: Identifier for the source body model
            floor_alignment: Floor alignment method (``"joint"``, ``"mesh"``, or ``"none"``)

        Returns:
            Motion instance with axis-angle rotations
        """
        if quaternion_rotations.ndim != 3 or quaternion_rotations.shape[2] != 4:
            raise ValueError(
                f"quaternion_rotations must have shape (num_frames, num_joints, 4), got {quaternion_rotations.shape}"
            )
        if root_translation.ndim != 2 or root_translation.shape[1] != 3:
            raise ValueError(f"root_translation must have shape (num_frames, 3), got {root_translation.shape}")

        num_frames, num_joints = quaternion_rotations.shape[:2]
        if root_translation.shape[0] != num_frames:
            raise ValueError(
                f"Number of frames must match: quaternion_rotations={num_frames}, "
                f"root_translation={root_translation.shape[0]}"
            )
        if num_joints != skeleton.num_joints:
            raise ValueError(
                f"Number of joints must match skeleton: quaternion_rotations={num_joints}, "
                f"skeleton={skeleton.num_joints}"
            )

        quat_norms = np.linalg.norm(quaternion_rotations, axis=2)
        if not np.allclose(quat_norms, 1.0, atol=1e-4):
            raise ValueError("All quaternions must be unit quaternions")

        if framerate <= 0:
            raise ValueError(f"Framerate must be positive, got {framerate}")

        # Convert quaternions to axis-angles
        quaternions_flat = quaternion_rotations.reshape(-1, 4)
        axis_angles_flat = quaternion_to_axis_angle(quaternions_flat)
        axis_angle_rotations = axis_angles_flat.reshape(num_frames, num_joints, 3)

        return cls(
            axis_angle_rotations,
            root_translation,
            skeleton,
            framerate,
            up_axis,
            identity,
            forward_kinematics,
            body_model_path,
            floor_alignment=floor_alignment,
            model_data=model_data or {},
            body_model_type=body_model_type,
        )

    def slice_frames(self, start: int, end: int | None = None) -> "Motion":
        """Create a new Motion with a subset of frames.

        Args:
            start: Start frame index (inclusive)
            end: End frame index (exclusive), defaults to end of motion

        Returns:
            New Motion instance with sliced frames
        """
        if end is None:
            end = self.num_frames

        if not (0 <= start < end <= self.num_frames):
            raise ValueError(f"Invalid slice range [{start}, {end}) for motion with {self.num_frames} frames")

        sliced_rotations = self.joint_rotations[start:end].copy()
        sliced_translation = self.root_translation[start:end].copy()

        sliced_linear_vel = (
            self.root_linear_velocity[start:end].copy() if self.root_linear_velocity is not None else None
        )
        sliced_angular_vel = (
            self.joint_angular_velocity[start:end].copy() if self.joint_angular_velocity is not None else None
        )

        # Also slice model_data if it contains array-like data matching frame count
        # This is a simple heuristic - if it's a list/array of same length, slice it.
        sliced_model_data = {}
        for k, v in self.model_data.items():
            if hasattr(v, "__len__") and len(v) == self.num_frames:
                try:
                    sliced_model_data[k] = v[start:end]
                except (TypeError, KeyError):
                    sliced_model_data[k] = v
            else:
                sliced_model_data[k] = v

        return Motion(
            sliced_rotations,
            sliced_translation,
            self.skeleton,
            self.framerate,
            self.up_axis,
            self.identity,
            self.forward_kinematics,
            self.body_model_path,
            self.floor_alignment,
            sliced_model_data,
            self.body_model_type,
            sliced_linear_vel,
            sliced_angular_vel,
        )

    def downsample(self, factor: int) -> "Motion":
        """Downsample motion by taking every Nth frame.

        Args:
            factor: Downsampling factor (take every Nth frame)

        Returns:
            New Motion instance with downsampled frames
        """
        if factor < 1:
            raise ValueError(f"Downsampling factor must be >= 1, got {factor}")

        downsampled_rotations = self.joint_rotations[::factor].copy()
        downsampled_translation = self.root_translation[::factor].copy()
        new_framerate = self.framerate / factor

        ds_linear_vel = self.root_linear_velocity[::factor].copy() if self.root_linear_velocity is not None else None
        ds_angular_vel = (
            self.joint_angular_velocity[::factor].copy() if self.joint_angular_velocity is not None else None
        )

        # Downsample model data
        downsampled_model_data = {}
        for k, v in self.model_data.items():
            if hasattr(v, "__len__") and len(v) == self.num_frames:
                try:
                    downsampled_model_data[k] = v[::factor]
                except (TypeError, KeyError):
                    downsampled_model_data[k] = v
            else:
                downsampled_model_data[k] = v

        return Motion(
            downsampled_rotations,
            downsampled_translation,
            self.skeleton,
            new_framerate,
            self.up_axis,
            self.identity,
            self.forward_kinematics,
            self.body_model_path,
            self.floor_alignment,
            downsampled_model_data,
            self.body_model_type,
            ds_linear_vel,
            ds_angular_vel,
        )

    def normalize_quaternions(self) -> "Motion":
        """Re-build motion with per-joint quaternions normalized to unit length.

        Returns:
            New Motion instance (axis-angle representation is updated consistently).
        """
        q = np.array(self.quaternions, dtype=np.float64, copy=True)
        flat = q.reshape(-1, 4)
        flat = normalize_quaternion(flat)
        aa = quaternion_to_axis_angle(flat).reshape(self.num_frames, self.num_joints, 3)

        return Motion(
            aa,
            np.array(self.root_translation, copy=True),
            self.skeleton,
            self.framerate,
            self.up_axis,
            self.identity,
            self.forward_kinematics,
            self.body_model_path,
            self.floor_alignment,
            dict(self.model_data),
            self.body_model_type,
            np.array(self.root_linear_velocity, copy=True) if self.root_linear_velocity is not None else None,
            np.array(self.joint_angular_velocity, copy=True) if self.joint_angular_velocity is not None else None,
        )

    def to_dict(self) -> dict:
        """Convert motion to dictionary representation.

        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "joint_rotations": self.joint_rotations.tolist(),
            "root_translation": self.root_translation.tolist(),
            "skeleton": self.skeleton.to_dict(),
            "framerate": self.framerate,
            "up_axis": self.up_axis,
            "floor_alignment": self.floor_alignment,
            "model_data": self.model_data,
            "body_model_type": self.body_model_type,
            "body_model_path": self.body_model_path,
            "root_linear_velocity": self.root_linear_velocity.tolist()
            if self.root_linear_velocity is not None
            else None,
            "joint_angular_velocity": self.joint_angular_velocity.tolist()
            if self.joint_angular_velocity is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Motion":
        """Create motion from dictionary representation.

        Args:
            data: Dictionary with motion data

        Returns:
            Motion instance
        """
        joint_rotations = np.array(data["joint_rotations"])
        root_translation = np.array(data["root_translation"])
        skeleton = Skeleton.from_dict(data["skeleton"])
        framerate = data.get("framerate", 30.0)
        up_axis = data.get("up_axis", "y")
        identity = Identity.from_dict(data["identity"]) if "identity" in data else None
        forward_kinematics = data.get("forward_kinematics", None)
        body_model_path = data.get("body_model_path", None)
        model_data = data.get("model_data", None)
        body_model_type = data.get("body_model_type", None)

        root_linear_velocity = (
            np.array(data["root_linear_velocity"]) if data.get("root_linear_velocity") is not None else None
        )
        joint_angular_velocity = (
            np.array(data["joint_angular_velocity"]) if data.get("joint_angular_velocity") is not None else None
        )

        return cls(
            joint_rotations,
            root_translation,
            skeleton,
            framerate,
            up_axis,
            identity,
            forward_kinematics,
            body_model_path,
            floor_alignment=data.get("floor_alignment", "joint"),
            model_data=model_data or {},
            body_model_type=body_model_type,
            root_linear_velocity=root_linear_velocity,
            joint_angular_velocity=joint_angular_velocity,
        )

    @classmethod
    def create_identity(
        cls, skeleton: Skeleton, num_frames: int, framerate: float = 30.0, body_model_path: str | None = None
    ) -> "Motion":
        """Create a motion with identity rotations (T-pose) and zero translation.

        Args:
            skeleton: Skeleton to use
            num_frames: Number of frames
            framerate: Frames per second
            body_model_path: Path to body models directory

        Returns:
            Motion instance with identity pose
        """
        # Zero axis-angles (no rotation)
        joint_rotations = np.zeros((num_frames, skeleton.num_joints, 3))

        # Zero translation
        root_translation = np.zeros((num_frames, 3))

        return cls(
            joint_rotations,
            root_translation,
            skeleton,
            framerate,
            "y",
            None,
            None,
            body_model_path,
            floor_alignment="none",
        )

    # ML Interoperability methods
    def to_torch(self) -> tuple:
        """Convert to PyTorch tensors.

        Returns:
            Tuple of (joint_rotations_tensor, root_translation_tensor)

        Raises:
            ImportError: If PyTorch is not available
        """
        rotations_tensor = torch.from_numpy(np.array(self.joint_rotations, copy=True)).float()
        translation_tensor = torch.from_numpy(np.array(self.root_translation, copy=True)).float()

        return rotations_tensor, translation_tensor

    @classmethod
    def from_torch(
        cls,
        rotations_tensor,
        translation_tensor,
        skeleton: Skeleton,
        framerate: float = 30.0,
        up_axis: str = "y",
        identity: Identity | None = None,
        forward_kinematics: ForwardKinematics | str | None = None,
        body_model_path: str | None = None,
        model_data: dict | None = None,
        body_model_type: str | None = None,
        floor_alignment: str = "joint",
    ) -> "Motion":
        """Create Motion from PyTorch tensors.

        Args:
            rotations_tensor: Joint rotations tensor, shape (num_frames, num_joints, 3) - axis-angles
            translation_tensor: Root translation tensor, shape (num_frames, 3)
            skeleton: Skeleton to use
            framerate: Frames per second
            up_axis: Up axis direction
            identity: Optional Identity object
            forward_kinematics: FK implementation or method name
            body_model_path: Path to body models directory
            model_data: Additional model-specific data
            body_model_type: Identifier for the source body model
            floor_alignment: Floor alignment mode (see :class:`Motion`)

        Returns:
            Motion instance
        """
        rotations_numpy = rotations_tensor.detach().cpu().numpy()
        translation_numpy = translation_tensor.detach().cpu().numpy()

        return cls(
            rotations_numpy,
            translation_numpy,
            skeleton,
            framerate,
            up_axis,
            identity,
            forward_kinematics,
            body_model_path,
            floor_alignment=floor_alignment,
            model_data=model_data or {},
            body_model_type=body_model_type,
        )

    def to_jax(self) -> tuple:
        """Convert to JAX arrays (axis-angle rotations).

        Returns:
            Tuple of (joint_rotations_array, root_translation_array)

        Raises:
            ImportError: If JAX is not available
        """
        try:
            import jax.numpy as jnp
        except ImportError as e:
            raise ImportError("JAX is required for to_jax(). Install with `pip install jax jaxlib`.") from e

        return jnp.array(self.joint_rotations), jnp.array(self.root_translation)

    @classmethod
    def from_jax(
        cls,
        rotations_array,
        translation_array,
        skeleton: Skeleton,
        framerate: float = 30.0,
        up_axis: str = "y",
        identity: Identity | None = None,
        forward_kinematics: ForwardKinematics | str | None = None,
        body_model_path: str | None = None,
        model_data: dict | None = None,
        body_model_type: str | None = None,
        floor_alignment: str = "joint",
    ) -> "Motion":
        """Create Motion from JAX arrays (axis-angle rotations)."""
        rotations_numpy = np.asarray(rotations_array)
        translation_numpy = np.asarray(translation_array)

        return cls(
            rotations_numpy,
            translation_numpy,
            skeleton,
            framerate,
            up_axis,
            identity,
            forward_kinematics,
            body_model_path,
            floor_alignment=floor_alignment,
            model_data=model_data or {},
            body_model_type=body_model_type,
        )

    def __str__(self) -> str:
        """String representation of the motion."""
        return (
            f"Motion({self.num_frames} frames, {self.num_joints} joints, {self.framerate:.1f}fps, {self.duration:.1f}s)"
        )
