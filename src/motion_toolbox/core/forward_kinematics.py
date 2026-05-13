#!/usr/bin/env python3
"""Forward kinematics implementations with pluggable backends.

This module provides a clean class-based interface for different forward kinematics
implementations, allowing easy switching between custom FK and native model FK.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np

from ..utils.quaternion import axis_angle_to_quaternion, q_mul, q_rotate


class ForwardKinematics(ABC):
    """Abstract base class for forward kinematics implementations."""

    @abstractmethod
    def compute_positions(
        self, joint_rotations: np.ndarray, root_translation: np.ndarray, skeleton, up_axis: str = "y", identity=None
    ) -> np.ndarray:
        """Compute joint positions from rotations and translations.

        Args:
            joint_rotations: Joint rotations as axis-angles, shape (num_frames, num_joints, 3)
            root_translation: Root translation, shape (num_frames, 3)
            skeleton: Skeleton with parent indices and offsets
            up_axis: Coordinate system - 'y' for Y-up, 'z' for Z-up
            identity: Person-specific attributes (betas, gender, etc.)

        Returns:
            Joint positions in world coordinates, shape (num_frames, num_joints, 3)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this FK implementation."""
        pass


class CustomForwardKinematics(ForwardKinematics):
    """Basic skeletal forward kinematics implementation."""

    @property
    def name(self) -> str:
        return "Basic Skeletal FK"

    def compute_positions(
        self, joint_rotations: np.ndarray, root_translation: np.ndarray, skeleton, up_axis: str = "y", identity=None
    ) -> np.ndarray:
        """Compute positions using standard hierarchical FK.

        Assumes joint_rotations are local rotations relative to parent.
        """
        num_frames, num_joints, _ = joint_rotations.shape

        # Convert axis-angles to quaternions
        # Flatten for batch processing
        quats = axis_angle_to_quaternion(joint_rotations.reshape(-1, 3)).reshape(num_frames, num_joints, 4)

        # Prepare output arrays
        global_positions = np.zeros((num_frames, num_joints, 3))
        global_rotations = np.zeros((num_frames, num_joints, 4))
        global_rotations[:, :, 0] = 1.0  # Initialize as identity quaternions (w=1, x=0, y=0, z=0)

        # Get parent indices and offsets from skeleton
        parent_indices = skeleton.parent_indices
        offsets = skeleton.offsets

        # Iterate through joints
        # Note: This assumes topological order where parents appear before children.
        # The Skeleton class validation usually ensures parent_index < joint_index.
        for i in range(num_joints):
            parent = parent_indices[i]

            if parent is None:  # Root joint
                # Global rotation is the local rotation
                global_rotations[:, i] = quats[:, i]
                # Global position is root translation + offset (usually 0 for root)
                global_positions[:, i] = root_translation + offsets[i]
            else:
                # Global rotation = Parent Global Rot * Local Rot
                global_rotations[:, i] = q_mul(global_rotations[:, parent], quats[:, i])

                # Global position = Parent Global Pos + Parent Global Rot * Offset
                # We rotate the offset vector by the parent's global rotation
                offset_broadcast = np.tile(offsets[i], (num_frames, 1))
                rotated_offset = q_rotate(global_rotations[:, parent], offset_broadcast)

                global_positions[:, i] = global_positions[:, parent] + rotated_offset

        return global_positions


class SMPLForwardKinematics(ForwardKinematics):
    """SMPL native forward kinematics implementation."""

    def __init__(self, model_path: str):
        """Initialize SMPL FK.

        Args:
            model_path: Path to SMPL models directory (required).
        """
        if not model_path:
            raise ValueError("model_path is required for SMPL forward kinematics")

        self.model_path = model_path
        self._smpl_models = {}  # Cache for loaded models

        # Check if SMPL dependencies are available
        import importlib.util

        self._torch_available = (
            importlib.util.find_spec("smplx") is not None and importlib.util.find_spec("torch") is not None
        )
        if not self._torch_available:
            warnings.warn("torch and smplx not available. SMPL FK will not work.")

    @property
    def name(self) -> str:
        return "SMPL Native FK"

    def compute_positions(
        self, joint_rotations: np.ndarray, root_translation: np.ndarray, skeleton, up_axis: str = "y", identity=None
    ) -> np.ndarray:
        """Compute positions using SMPL's native forward kinematics."""
        if not self._torch_available:
            raise RuntimeError("SMPL FK requires torch and smplx. Install with: pip install torch smplx")

        import torch

        # Get SMPL parameters from identity or use defaults
        if identity and hasattr(identity, "betas") and identity.betas is not None:
            betas_np = identity.betas
            gender = getattr(identity, "gender", "neutral")
        else:
            betas_np = np.zeros(10)
            gender = "neutral"

        # Get or create SMPL model
        try:
            smpl_model = self._get_smpl_model(gender)
        except ValueError as e:
            if "Unknown model type" in str(e):
                # Fallback for different SMPL-X versions or model types
                # Sometimes 'smpl' is not recognized if only 'smplx' is installed or configured?
                # Or maybe it expects 'body' instead of 'smpl'?
                # Try 'smpl' -> 'body' if needed, but standard is 'smpl'.
                # The error "Unknown model type body" suggests something is asking for 'body'
                # OR 'body' is what it got and didn't like.
                # But we passed 'smpl'.
                # Actually the error trace says: ValueError: Unknown model type body, exiting!
                # This often happens in smplx if model_path doesn't contain the expected file structure.
                raise ValueError(
                    f"SMPL model creation failed. Ensure model_path '{self.model_path}' contains SMPL_NEUTRAL.pkl etc. Error: {e}"
                )
            raise e

        # Joint rotations are already in axis-angle format
        num_frames, num_joints = joint_rotations.shape[:2]

        # Ensure the input is axis-angles (should be shape [..., 3])
        if joint_rotations.shape[2] != 3:
            raise ValueError(f"Expected axis-angle rotations with shape [..., 3], got {joint_rotations.shape}")

        global_orient = torch.from_numpy(joint_rotations[:, 0, :]).float()

        # Body pose: joints 1-23 for SMPL (pad or trim as needed)
        num_body_joints = min(23, num_joints - 1)
        if num_body_joints > 0:
            body_pose_data = joint_rotations[:, 1 : 1 + num_body_joints, :].reshape(num_frames, -1)
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
        transl = torch.from_numpy(root_translation.copy()).float()

        # Run SMPL forward pass
        with torch.no_grad():
            smpl_output = smpl_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
            )

        # Extract joint positions
        joint_positions = smpl_output.joints.cpu().numpy()

        # Handle joint count mismatch
        if num_joints != 24:
            joint_positions = self._adjust_joint_count(joint_positions, num_joints)

        return joint_positions

    def _get_smpl_model(self, gender: str):
        """Get cached SMPL model or create new one."""
        if gender in self._smpl_models:
            return self._smpl_models[gender]

        import smplx

        model = smplx.create(
            model_path=self.model_path,
            model_type="smpl",
            gender=gender,
            # ext="pkl", # Let smplx auto-detect extension
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        )

        self._smpl_models[gender] = model
        return model

    def _adjust_joint_count(self, positions: np.ndarray, target_count: int) -> np.ndarray:
        """Adjust joint count to match skeleton."""
        num_frames, current_count = positions.shape[:2]

        if target_count == current_count:
            return positions
        elif target_count < current_count:
            # Trim to target count
            return positions[:, :target_count, :]
        else:
            # Pad with zeros
            padding = np.zeros((num_frames, target_count - current_count, 3))
            return np.concatenate([positions, padding], axis=1)


# Factory function for easy creation
def create_forward_kinematics(method: str = "auto", model_path: str | None = None, **kwargs) -> ForwardKinematics:
    """Create a forward kinematics implementation.

    Args:
        method: FK method - "custom", "smpl", or "auto"
        model_path: Path to SMPL models (required for SMPL FK)
        **kwargs: Additional arguments

    Returns:
        ForwardKinematics implementation
    """
    if method == "custom":
        return CustomForwardKinematics()
    elif method == "smpl":
        if model_path is None:
            raise ValueError("model_path is required when method='smpl'")
        return SMPLForwardKinematics(model_path)
    elif method == "auto":
        # Try SMPL first if path provided, otherwise fall back to Custom
        if model_path is not None:
            try:
                return SMPLForwardKinematics(model_path)
            except (ImportError, RuntimeError):
                # Log warning?
                pass

        # Fallback to Custom/Basic FK
        return CustomForwardKinematics()
    else:
        raise ValueError(f"Unknown FK method: {method}")
