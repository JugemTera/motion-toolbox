"""Motion export utilities for 3D human motion animation.

This module provides classes for exporting motion sequences to GLB format.
It delegates the actual export logic to the appropriate BodyModel implementation.
"""

import os

import torch

from ..core.body_model import BodyModel
from ..core.motion import Motion


class GLBExporter:
    """Exports Motion instances to .glb using a BodyModel.

    This class acts as a facade for BodyModel.export_glb, providing
    backward compatibility and easy initialization for SMPL.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "smpl",
        gender: str = "neutral",
        device: str | torch.device = "cpu",
        fps: int = 30,
        ext: str = "npz",
        include_floor: bool = True,
        body_model: BodyModel | None = None,
    ):
        """Initialize GLBExporter.

        Args:
            model_path (str): Directory containing model files (for default SMPL).
            model_type (str): Model type ("smpl", "smplx").
            gender (str): Gender string.
            device (Union[str, torch.device]): Torch device.
            fps (int): Frames per second for animation.
            ext (str): File extension for model files.
            include_floor (bool): Whether to include checkerboard floor.
            body_model (Optional[BodyModel]): Custom body model instance. If provided,
                                              other model-specific args are ignored.
        """
        self.fps = fps
        self.include_floor = include_floor

        if body_model is not None:
            self.body_model = body_model
        else:
            # Default to SMPL for backward compatibility
            # Local import to avoid circular dependency
            from ..core.body_models.smpl import SMPLBodyModel

            if not os.path.isdir(model_path):
                raise ValueError(f"model_path does not exist: {model_path}")

            self.body_model = SMPLBodyModel(
                model_path=model_path,
                model_type=model_type,
                gender=gender,
                device=str(device),
            )

    def export(self, motion: Motion, filename: str):
        """Export a Motion instance as .glb file.

        Args:
            motion: The Motion instance to export.
            filename (str): Output file path.
        """
        self.body_model.export_glb(
            motion=motion,
            output_path=filename,
            fps=self.fps,
            include_floor=self.include_floor,
        )
