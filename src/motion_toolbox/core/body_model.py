"""Abstract base class for body models.

This module defines the interface that all body models (SMPL, MHR, etc.) must implement
to be compatible with the motion toolbox's visualization and kinematics systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .skeleton import Skeleton

if TYPE_CHECKING:
    from .motion import Motion


class BodyModel(ABC):
    """Abstract interface for 3D body models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the body model."""
        pass

    @abstractmethod
    def get_skeleton(self) -> Skeleton:
        """Get the skeleton structure associated with this body model.

        Returns:
            Skeleton: The skeleton definition.
        """
        pass

    @abstractmethod
    def forward_kinematics(
        self,
        pose_data: dict[str, Any],
        return_vertices: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Compute forward kinematics to get joints and optionally vertices.

        Args:
            pose_data: Dictionary containing model-specific pose parameters
                       (e.g., 'body_pose', 'global_orient', 'transl', 'betas').
            return_vertices: Whether to compute and return mesh vertices.

        Returns:
            Dictionary containing:
            - 'joints': Joint positions (num_frames, num_joints, 3)
            - 'vertices': Mesh vertices (num_frames, num_vertices, 3) (optional)
        """
        pass

    @abstractmethod
    def export_glb(
        self,
        motion: Motion,
        output_path: str,
        **kwargs,
    ) -> None:
        """Export motion to a GLB file using this body model.

        Args:
            motion: The Motion object to export.
            output_path: Path to save the GLB file.
            **kwargs: Additional model-specific export options.
        """
        pass
