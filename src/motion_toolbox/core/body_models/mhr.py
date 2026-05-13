"""MHR (Momentum Human Rig) body model implementation.

This module provides the MHR body model implementation, including forward kinematics
and GLB export functionality using the mhr library.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...visualization.constants import JOINT_COUNTS, MHR_FACE_EXPR_DIMS
from ...visualization.gltf_utils import build_gltf
from ..body_model import BodyModel
from ..skeleton import Skeleton

if TYPE_CHECKING:
    from ..motion import Motion


class MHRBodyModel(BodyModel):
    """MHR body model implementation.

    Wraps the facebookresearch/MHR model.
    """

    def __init__(self, model_path: str, lod: int = 1, device: str = "cpu"):
        """Initialize MHR body model.

        Args:
            model_path: Path to MHR assets directory.
            lod: Level of detail (0-6).
            device: Torch device.
        """
        self.model_path = model_path
        self.lod = lod
        self.device = torch.device(device)
        self._mhr_model = None
        self._parents = None

        import importlib.util

        self._mhr_available = importlib.util.find_spec("mhr") is not None
        if not self._mhr_available:
            warnings.warn("mhr package not available. MHR features will not work.")

    @property
    def name(self) -> str:
        return f"MHR (LOD {self.lod})"

    def get_skeleton(self) -> Skeleton:
        """Get MHR skeleton.

        Returns a skeleton compatible with MHR's kinematic structure.
        Attempts to extract hierarchy from the model if loaded.
        """
        # Try to get model to extract parents
        try:
            model = self._get_model()
            if hasattr(model, "parents"):
                parents = model.parents.cpu().numpy() if torch.is_tensor(model.parents) else model.parents
            elif hasattr(model, "kintree_table"):
                parents = model.kintree_table[0].cpu().numpy()
            elif (
                hasattr(model, "character")
                and hasattr(model.character, "skeleton")
                and hasattr(model.character.skeleton, "joint_parents")
            ):
                parents = np.array(model.character.skeleton.joint_parents)
            else:
                parents = None
        except Exception:
            parents = None

        if parents is not None:
            num_joints = len(parents)
            self._parents = parents
            # Convert -1 to None for Skeleton compatibility
            parent_indices = tuple(None if int(p) == -1 else int(p) for p in parents)
            return Skeleton(
                joint_names=tuple(f"joint_{i}" for i in range(num_joints)),
                parent_indices=parent_indices,
                offsets=np.zeros((num_joints, 3)),
            )

        # Fallback if model not loaded or structure unknown
        num_joints = JOINT_COUNTS["mhr"]  # Approximate default
        return Skeleton(
            joint_names=tuple(f"joint_{i}" for i in range(num_joints)),
            parent_indices=tuple([None] + [0] * (num_joints - 1)),
            offsets=np.zeros((num_joints, 3)),
        )

    def _get_model(self):
        """Get or create MHR model instance."""
        if self._mhr_model is not None:
            return self._mhr_model

        if not self._mhr_available:
            raise RuntimeError("mhr package not available")

        from mhr.mhr import MHR

        # Handle custom model path
        if self.model_path and os.path.isdir(self.model_path):
            if os.path.isdir(os.path.join(self.model_path, "assets")):
                assets_dir = os.path.join(self.model_path, "assets")
            elif os.path.isdir(os.path.join(self.model_path, "mhr_assets")):
                assets_dir = os.path.join(self.model_path, "mhr_assets")
            else:
                assets_dir = self.model_path

            os.environ["MHR_ASSETS_DIR"] = assets_dir

            # Pass explicit folder path to avoid using the default (which points to site-packages)
            self._mhr_model = MHR.from_files(folder=Path(assets_dir), device=self.device, lod=self.lod)
        else:
            # Default initialization using package assets
            self._mhr_model = MHR.from_files(device=self.device, lod=self.lod)

        return self._mhr_model

    def forward_kinematics(
        self,
        pose_data: dict[str, Any],
        return_vertices: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Compute forward kinematics."""
        model = self._get_model()

        identity_coeffs = pose_data.get("identity_coeffs")
        model_parameters = pose_data.get("model_parameters")
        face_expr_coeffs = pose_data.get("face_expr_coeffs")

        if identity_coeffs is None:
            batch_size = model_parameters.shape[0] if model_parameters is not None else 1
            identity_coeffs = torch.zeros((batch_size, 45), device=self.device)

        if model_parameters is None:
            raise ValueError("model_parameters required for MHR FK")

        if face_expr_coeffs is None:
            batch_size = model_parameters.shape[0]
            face_expr_coeffs = torch.zeros((batch_size, MHR_FACE_EXPR_DIMS), device=self.device)

        identity_coeffs = torch.as_tensor(identity_coeffs, device=self.device).float()
        model_parameters = torch.as_tensor(model_parameters, device=self.device).float()
        face_expr_coeffs = torch.as_tensor(face_expr_coeffs, device=self.device).float()

        if identity_coeffs.ndim == 1:
            identity_coeffs = identity_coeffs.unsqueeze(0)
        if model_parameters.ndim == 1:
            model_parameters = model_parameters.unsqueeze(0)
        if face_expr_coeffs.ndim == 1:
            face_expr_coeffs = face_expr_coeffs.unsqueeze(0)

        with torch.no_grad():
            vertices, skeleton_state = model(identity_coeffs, model_parameters, face_expr_coeffs)

        result = {}
        if hasattr(skeleton_state, "joint_positions"):
            result["joints"] = skeleton_state.joint_positions.cpu().numpy()
        elif isinstance(skeleton_state, (list, tuple)) and len(skeleton_state) > 1:
            result["joints"] = skeleton_state[1].cpu().numpy()
        elif isinstance(skeleton_state, torch.Tensor):
            result["joints"] = skeleton_state.cpu().numpy()

        if return_vertices:
            result["vertices"] = vertices.cpu().numpy()

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
        if not self._mhr_available:
            raise RuntimeError("mhr needed for GLB export")

        # Check if this is retargeted motion (direct rotation mapping)
        is_retargeted = motion.model_data.get("_retargeted", False)

        if is_retargeted:
            # Use direct skeletal animation with retargeted rotations
            return self._export_glb_retargeted(motion, output_path, fps, include_floor)
        elif motion.body_model_type == "mhr" and "model_parameters" in motion.model_data:
            model_parameters = motion.model_data["model_parameters"]
            identity_coeffs = motion.model_data.get("identity_coeffs")
            face_expr_coeffs = motion.model_data.get("face_expr_coeffs")
        else:
            raise NotImplementedError("Conversion from generic Motion to MHR parameters not yet implemented.")

        # Prepare data
        model_parameters = torch.as_tensor(model_parameters, device=self.device).float()
        batch_size = model_parameters.shape[0]

        if identity_coeffs is None:
            identity_coeffs = torch.zeros((batch_size, 45), device=self.device)
        else:
            identity_coeffs = torch.as_tensor(identity_coeffs, device=self.device).float()

        if face_expr_coeffs is None:
            face_expr_coeffs = torch.zeros((batch_size, MHR_FACE_EXPR_DIMS), device=self.device)
        else:
            face_expr_coeffs = torch.as_tensor(face_expr_coeffs, device=self.device).float()

        # Ensure dimensions
        if model_parameters.ndim == 1:
            model_parameters = model_parameters.unsqueeze(0)
        if identity_coeffs.ndim == 1:
            identity_coeffs = identity_coeffs.repeat(batch_size, 1)
        if face_expr_coeffs.ndim == 1:
            face_expr_coeffs = face_expr_coeffs.repeat(batch_size, 1)

        if identity_coeffs.shape[0] == 1:
            identity_coeffs = identity_coeffs.repeat(batch_size, 1)
        if face_expr_coeffs.shape[0] == 1:
            face_expr_coeffs = face_expr_coeffs.repeat(batch_size, 1)

        model = self._get_model()

        # Run forward pass to get geometry
        with torch.no_grad():
            vertices, skeleton_state = model(identity_coeffs, model_parameters, face_expr_coeffs)

            # Don't compute init pose - use MHR's native bind pose instead!
            # MHR character already has bind_pose vertices and inverse_bind_matrices

        vertices_np = vertices.cpu().numpy()

        # Use MHR's bind pose vertices (what the skin weights are defined against)
        if hasattr(model, "character") and hasattr(model.character, "mesh"):
            init_vertices_raw = model.character.mesh.vertices
            if hasattr(init_vertices_raw, "cpu"):
                init_vertices_raw = init_vertices_raw.cpu().numpy()
            init_vertices_np = np.array(init_vertices_raw)[np.newaxis, ...]  # Add batch dim
        else:
            raise RuntimeError("Could not find MHR character mesh bind pose")

        # MHR uses centimeters, convert to meters for consistency with SMPL
        # SMPL typically has height ~1.7-1.8m, MHR is ~1.47m in cm units
        MHR_SCALE = 0.01  # cm to meters
        vertices_np = vertices_np * MHR_SCALE
        init_vertices_np = init_vertices_np * MHR_SCALE

        # MHR's default facing is backward (-Z), flip to face camera (+Z)
        # This rotates 180° around Y-axis
        vertices_np[:, :, 2] *= -1
        init_vertices_np[:, :, 2] *= -1

        # Get parents hierarchy
        if hasattr(model, "parents"):
            parents = model.parents.cpu().numpy() if torch.is_tensor(model.parents) else model.parents
        elif hasattr(model, "kintree_table"):
            parents = model.kintree_table[0].cpu().numpy()
        elif (
            hasattr(model, "character")
            and hasattr(model.character, "skeleton")
            and hasattr(model.character.skeleton, "joint_parents")
        ):
            parents = np.array(model.character.skeleton.joint_parents)
        else:
            raise RuntimeError("Could not find parents hierarchy in MHR model")

        # Extract bind-pose joint positions from MHR's bind_pose matrices
        # bind_pose is (num_joints, 4, 4) transformation matrices
        if hasattr(model, "character") and hasattr(model.character, "bind_pose"):
            bind_pose_matrices = model.character.bind_pose  # (127, 4, 4)
            # Joint positions are in the translation column of the matrices
            init_joints_np = bind_pose_matrices[:, :3, 3]  # (127, 3)
            init_joints_np = init_joints_np[np.newaxis, ...] * MHR_SCALE  # (1, 127, 3), scale to meters
        else:
            raise RuntimeError("Could not find MHR character bind_pose")

        # For animated joint positions, we still need to compute from skeleton_state
        # Extract local translations and compute global positions
        def compute_global_joint_positions(local_translations, parents):
            """Compute global joint positions from local translations and hierarchy."""
            num_joints = local_translations.shape[0]
            global_positions = np.zeros_like(local_translations)

            for joint_idx in range(num_joints):
                parent_idx = int(parents[joint_idx])
                if parent_idx == -1:
                    global_positions[joint_idx] = local_translations[joint_idx]
                else:
                    global_positions[joint_idx] = global_positions[parent_idx] + local_translations[joint_idx]
            return global_positions

        local_translations = skeleton_state[..., :3].cpu().numpy()  # (batch, num_joints, 3)
        joints_np = np.zeros_like(local_translations)
        for frame_idx in range(local_translations.shape[0]):
            joints_np[frame_idx] = compute_global_joint_positions(local_translations[frame_idx], parents)
        joints_np = joints_np * MHR_SCALE

        # Faces
        if hasattr(model, "faces"):
            faces = model.faces.cpu().numpy().astype(np.int32) if torch.is_tensor(model.faces) else model.faces
        elif hasattr(model, "mesh_faces"):
            faces = model.mesh_faces.cpu().numpy().astype(np.int32)
        elif (
            hasattr(model, "character") and hasattr(model.character, "mesh") and hasattr(model.character.mesh, "faces")
        ):
            faces = np.array(model.character.mesh.faces, dtype=np.int32)
        else:
            raise RuntimeError("Could not find faces in MHR model")

        # Parents already extracted earlier (needed for global joint position computation)

        # Weights - Extract from MHR's SkinWeights structure
        lbs_weights = None

        if hasattr(model, "lbs_weights"):
            lbs_weights = model.lbs_weights.detach().cpu().numpy()
        elif hasattr(model, "weights"):
            lbs_weights = model.weights.detach().cpu().numpy()
        elif hasattr(model, "character") and hasattr(model.character, "skin_weights"):
            # MHR stores skin weights in character.skin_weights (not mesh.skin_weights)
            # It's a pymomentum.geometry.SkinWeights object with sparse storage
            char = model.character
            sw = char.skin_weights

            # Get number of joints from skeleton
            if hasattr(char, "skeleton") and hasattr(char.skeleton, "size"):
                num_joints = char.skeleton.size
                # Convert sparse skin weights to dense (num_vertices, num_joints) matrix
                lbs_weights = sw.to_dense(num_joints)  # Returns numpy array
            else:
                warnings.warn("Could not determine number of joints from skeleton")
                lbs_weights = None

        if lbs_weights is None:
            # Fallback: create rigid binding if skin weights extraction failed
            num_verts = vertices_np.shape[1]  # (batch, verts, 3)
            num_joints = joints_np.shape[1] if joints_np.ndim > 2 else joints_np.shape[0]
            lbs_weights = np.zeros((num_verts, num_joints))
            # Assign each vertex to root joint
            lbs_weights[:, 0] = 1.0
            warnings.warn("Using fallback rigid binding (all vertices to root). Skinning will not be realistic.")

        # MHR uses pose correctives and non-linear deformations beyond simple LBS.
        # The vertices include blendshapes and correctives that can't be replicated
        # with skeletal animation alone. We must use vertex animation (morph targets).

        # Floor alignment: Move lowest vertex to y=0 across all frames
        min_y = vertices_np[:, :, 1].min()
        vertices_np[:, :, 1] -= min_y
        init_vertices_np[:, :, 1] -= min_y

        from ...visualization.gltf_utils_vertex_anim import build_gltf_vertex_animation

        gltf = build_gltf_vertex_animation(
            all_vertices=vertices_np,  # All frame vertices with MHR's deformations
            init_vertices=init_vertices_np,  # MHR's bind pose
            faces=faces,
            fps=fps,
            include_floor=include_floor,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gltf.save_binary(output_path)
        print(f"Exported animated GLB: {output_path}")

    def _export_glb_retargeted(
        self,
        motion: Motion,
        output_path: str,
        fps: int = 30,
        include_floor: bool = True,
    ) -> None:
        """Export retargeted motion using skeletal animation.

        This is used when we have direct joint rotations from retargeting,
        not MHR model_parameters. We use skeletal animation (LBS) instead
        of vertex animation.
        """
        from scipy.spatial.transform import Rotation as R

        # MHR uses centimeters, GLB expects meters
        MHR_SCALE = 0.01

        model = self._get_model()

        # Get joint rotations from model_data (quaternions)
        joint_rotations_quat = motion.model_data["joint_rotations"]  # (num_frames, 127, 4)
        root_translation = motion.model_data["root_translation"]  # (num_frames, 3) in cm

        num_frames = joint_rotations_quat.shape[0]
        num_joints = joint_rotations_quat.shape[1]

        # Get bind pose data
        # Vertices need batch dimension: (1, num_verts, 3)
        init_vertices_np = np.array(model.character.mesh.vertices) * MHR_SCALE
        init_vertices_np = init_vertices_np[np.newaxis, :, :]  # Add batch dim

        init_joints_np = np.array([m[:3, 3] for m in model.character.bind_pose]) * MHR_SCALE
        init_joints_np = init_joints_np[np.newaxis, :, :]  # Add batch dim

        faces = np.array(model.character.mesh.faces).astype(np.int32)
        parents = np.array(model.character.skeleton.joint_parents)

        # Get skin weights
        sw = model.character.skin_weights
        lbs_weights = sw.to_dense(model.character.skeleton.size)

        # Convert quaternions to axis-angle for build_gltf
        # Root rotation (joint 0 in MHR hierarchy - but we use joint 1 'root' for SMPL pelvis)
        # For retargeted motion, joint 0 (body_world) should have identity rotation
        # and joint 1 (root) has the pelvis rotation

        # Find which joint is 'root' (pelvis equivalent)
        root_joint_idx = 1  # 'root' joint in MHR

        global_orient = np.zeros((num_frames, 3))
        for frame in range(num_frames):
            quat = joint_rotations_quat[frame, root_joint_idx]
            if not (abs(quat[3]) > 0.9999 and np.linalg.norm(quat[:3]) < 1e-4):
                global_orient[frame] = R.from_quat(quat).as_rotvec()

        # Body pose: all other joints (excluding body_world at 0 and root at 1)
        # Actually, for proper MHR export, we need rotations for ALL joints
        body_pose = np.zeros((num_frames, (num_joints - 1) * 3))
        for frame in range(num_frames):
            for j in range(1, num_joints):  # Skip body_world (0)
                quat = joint_rotations_quat[frame, j]
                if not (abs(quat[3]) > 0.9999 and np.linalg.norm(quat[:3]) < 1e-4):
                    aa = R.from_quat(quat).as_rotvec()
                    body_pose[frame, (j - 1) * 3 : (j - 1) * 3 + 3] = aa

        # Root translation (convert from cm to meters)
        transl = root_translation * MHR_SCALE

        # joints parameter needs (num_frames, num_joints, 3) for build_gltf to get num_frames
        joints_animated = np.tile(init_joints_np, (num_frames, 1, 1))  # Repeat bind pose for all frames

        gltf = build_gltf(
            vertices=init_vertices_np,
            init_joints=init_joints_np,
            faces=faces,
            joints=joints_animated,
            global_orient=global_orient,
            transl=transl,
            body_pose=body_pose,
            parents=parents,
            fps=fps,
            include_floor=include_floor,
            lbs_weights=lbs_weights,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gltf.save_binary(output_path)
        print(f"Exported retargeted GLB (skeletal): {output_path}")
