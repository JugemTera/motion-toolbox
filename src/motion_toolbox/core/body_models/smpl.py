"""SMPL body model implementation.

This module provides the SMPL body model implementation, including forward kinematics
and GLB export functionality using the smplx library.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...visualization.gltf_utils import build_gltf
from ..body_model import BodyModel
from ..skeleton import Skeleton

if TYPE_CHECKING:
    from ..motion import Motion


class SMPLBodyModel(BodyModel):
    """SMPL body model implementation."""

    def __init__(self, model_path: str, model_type: str = "smpl", gender: str = "neutral", device: str = "cpu"):
        """Initialize SMPL body model.

        Args:
            model_path: Path to SMPL models directory.
            model_type: Model type ("smpl", "smplx", etc.).
            gender: Gender ("neutral", "male", "female").
            device: Torch device.
        """
        self.model_path = model_path
        self.model_type = model_type
        self.gender = gender
        self.device = torch.device(device)
        self._smpl_model = None  # Lazy initialization

        # Check availability
        import importlib.util

        self._smplx_available = importlib.util.find_spec("smplx") is not None
        if not self._smplx_available:
            warnings.warn("smplx not available. SMPL features will not work.")

    @property
    def name(self) -> str:
        return f"SMPL ({self.gender})"

    def get_skeleton(self) -> Skeleton:
        """Get SMPL skeleton."""
        return Skeleton.from_smpl()

    def _get_model(self):
        """Get or create SMPL model instance."""
        if self._smpl_model is not None:
            return self._smpl_model

        if not self._smplx_available:
            raise RuntimeError("smplx not available")

        import smplx

        self._smpl_model = smplx.create(
            self.model_path,
            model_type=self.model_type,
            gender=self.gender,
            ext="pkl",  # Try pkl first
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        ).to(self.device)

        return self._smpl_model

    def forward_kinematics(
        self,
        pose_data: dict[str, Any],
        return_vertices: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Compute forward kinematics."""
        model = self._get_model()

        # Extract parameters
        betas = pose_data.get("betas")
        global_orient = pose_data.get("global_orient")
        body_pose = pose_data.get("body_pose")
        transl = pose_data.get("transl")

        # Move to device
        if betas is not None and not isinstance(betas, torch.Tensor):
            betas = torch.as_tensor(betas).to(self.device).float()
        if global_orient is not None and not isinstance(global_orient, torch.Tensor):
            global_orient = torch.as_tensor(global_orient).to(self.device).float()
        if body_pose is not None and not isinstance(body_pose, torch.Tensor):
            body_pose = torch.as_tensor(body_pose).to(self.device).float()
        if transl is not None and not isinstance(transl, torch.Tensor):
            transl = torch.as_tensor(transl).to(self.device).float()

        # Forward pass
        with torch.no_grad():
            output = model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
            )

        result = {
            "joints": output.joints.cpu().numpy(),
        }

        if return_vertices:
            result["vertices"] = output.vertices.cpu().numpy()

        return result

    def export_glb(
        self,
        motion: Motion,
        output_path: str,
        fps: int = 30,
        include_floor: bool = True,
        **kwargs,
    ) -> None:
        """Export motion to GLB."""
        if not self._smplx_available:
            raise RuntimeError("smplx needed for GLB export")

        # Convert motion to SMPL params
        # We need MotionToSMPLConverter here, but it might import Motion which imports Skeleton...
        # Circular import risk.
        # Assuming the converter handles Motion object correctly.
        from ...converters.motion_converters import MotionToSMPLConverter

        converter = MotionToSMPLConverter()
        try:
            smpl_params = converter.convert(motion)
        except ValueError:
            # If skeleton is not SMPL-compatible, we might fail here.
            # But SMPLBodyModel implies we are dealing with SMPL data or compatible.
            raise

        # Extract SMPL parameters
        betas = smpl_params["betas"]
        body_pose = smpl_params["body_pose"]
        global_orient = smpl_params["global_orient"]
        transl = smpl_params["transl"]

        # Move to device
        betas = betas.to(self.device)
        body_pose = body_pose.to(self.device)
        global_orient = global_orient.to(self.device)
        transl = transl.to(self.device)

        # Ensure betas has correct shape for all frames
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).repeat(motion.num_frames, 1)
        elif betas.shape[0] == 1:
            betas = betas.repeat(motion.num_frames, 1)

        # Get model
        model = self._get_model()

        # Compute vertices and joints for all frames
        with torch.no_grad():
            output = model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
            )
            vertices = output.vertices.cpu().numpy()

            # Get joints using regressor
            # J_regressor: (24, 6890)
            # vertices: (N, 6890, 3)
            # joints: (N, 24, 3)
            joints = torch.einsum("bvx,jv->bjx", output.vertices, model.J_regressor).cpu().numpy()

            # Get initial pose (T-pose / rest pose) for binding
            # We use the first frame's shape (betas) but zero pose
            init_output = model(
                betas=betas[:1],
                global_orient=torch.zeros_like(global_orient[:1]),
                body_pose=torch.zeros_like(body_pose[:1]),
                transl=torch.zeros_like(transl[:1]),
            )
            init_vertices = init_output.vertices.cpu().numpy()
            init_joints = torch.einsum("bvx,jv->bjx", init_output.vertices, model.J_regressor).cpu().numpy()

        # Get model topology data
        faces = model.faces.astype(np.int32)
        parents = model.parents.detach().cpu().numpy()
        lbs_weights = model.lbs_weights.detach().cpu().numpy()

        # Build GLB
        gltf = build_gltf(
            vertices=init_vertices,  # Use T-pose vertices for bind pose, not animated vertices
            init_joints=init_joints,
            faces=faces,
            joints=joints,
            global_orient=global_orient.cpu().numpy(),
            transl=transl.cpu().numpy(),
            body_pose=body_pose.cpu().numpy(),
            parents=parents,
            fps=fps,
            include_floor=include_floor,
            lbs_weights=lbs_weights,
        )

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gltf.save_binary(output_path)
        print(f"Exported animated GLB: {output_path}")
