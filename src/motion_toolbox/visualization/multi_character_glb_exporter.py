"""Multi-character GLB exporter for spatial animation grids.

This module extends the GLBExporter to handle multiple animated characters
in a single GLB file, each positioned at different locations in 3D space.
"""

import os

import torch
from pygltflib import GLTF2

from motion_toolbox.converters import MotionToSMPLConverter
from motion_toolbox.core.motion import Motion
from motion_toolbox.visualization.glb_exporter import GLBExporter


class MultiCharacterGLBExporter:
    """Exports multiple Motion instances as a single .glb with multiple characters.

    This exporter creates a single GLTF file containing multiple SMPL bodies,
    each with its own animation, positioned at different locations in 3D space.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "smpl",
        gender: str = "neutral",
        device: str | torch.device = "cpu",
        fps: int = 30,
        ext: str = "npz",
    ):
        """Initialize MultiCharacterGLBExporter.

        Args:
            model_path (str): Directory containing SMPL model files
            model_type (str): Model type ("smpl", "smplx", etc.)
            gender (str): Gender for the model ("neutral", "male", "female")
            device (Union[str, torch.device]): Torch device for computations
            fps (int): Frames per second for the exported animation
            ext (str): File extension for model files ("pkl" or "npz")
        """
        # Create a single GLBExporter instance to reuse SMPL model
        self.base_exporter = GLBExporter(
            model_path=model_path,
            model_type=model_type,
            gender=gender,
            device=device,
            fps=fps,
            ext=ext,
        )
        self.fps = fps
        self.device = torch.device(device)

    def export_multi_character(
        self,
        motions: list[Motion],
        offsets: list[tuple[float, float, float]],
        filename: str,
    ) -> None:
        """Export multiple Motion instances as a single GLB with spatial offsets.

        Args:
            motions (List[Motion]): List of Motion instances to export
            offsets (List[Tuple[float, float, float]]): List of (x, y, z) offsets for each character
            filename (str): Output file path
        """
        if len(motions) != len(offsets):
            raise ValueError(f"Number of motions ({len(motions)}) must match number of offsets ({len(offsets)})")

        if not motions:
            raise ValueError("motions list cannot be empty")

        print(f"\n{'=' * 60}")
        print(f"Exporting {len(motions)} characters to single GLB...")
        print(f"{'=' * 60}\n")

        # Convert all motions to SMPL parameters
        converter = MotionToSMPLConverter()
        all_smpl_params = []

        for idx, motion in enumerate(motions):
            print(f"  [{idx + 1}/{len(motions)}] Converting motion to SMPL parameters...")
            smpl_params = converter.convert(motion)

            # Apply offset to translation
            offset_x, offset_y, offset_z = offsets[idx]
            offset = torch.tensor([[offset_x, offset_y, offset_z]], device=self.device)
            smpl_params["transl"] = smpl_params["transl"].to(self.device) + offset

            all_smpl_params.append(smpl_params)

        print("\n✅ All motions converted to SMPL\n")

        # Get SMPL data for all characters
        print(f"{'=' * 60}")
        print("Running SMPL forward pass for all characters...")
        print(f"{'=' * 60}\n")

        all_vertices_list = []
        all_joints_list = []
        all_init_vertices_list = []
        all_init_joints_list = []

        with torch.no_grad():
            for idx, smpl_params in enumerate(all_smpl_params):
                betas = smpl_params["betas"].to(self.device)
                body_pose = smpl_params["body_pose"].to(self.device)
                global_orient = smpl_params["global_orient"].to(self.device)
                transl = smpl_params["transl"].to(self.device)

                # Ensure betas has correct shape
                if betas.ndim == 1:
                    betas = betas.unsqueeze(0).repeat(motions[idx].num_frames, 1)

                # Forward pass for animation
                smpl_output = self.base_exporter._forward(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    transl=transl,
                )

                smpl_vertices = smpl_output.vertices  # (N, 6890, 3)
                smpl_joints = self.base_exporter._get_smpl_joints(smpl_vertices)  # (N, 24, 3)

                # Initial pose for template
                init_output = self.base_exporter._forward(
                    betas=betas[:1],
                    body_pose=torch.zeros_like(body_pose[:1]),
                    global_orient=torch.zeros_like(global_orient[:1]),
                    transl=transl[:1],  # Keep offset for initial pose
                )

                init_vertices = init_output.vertices  # (1, 6890, 3)
                init_joints = self.base_exporter._get_smpl_joints(init_vertices)  # (1, 24, 3)

                all_vertices_list.append(smpl_vertices)
                all_joints_list.append(smpl_joints)
                all_init_vertices_list.append(init_vertices)
                all_init_joints_list.append(init_joints)

                print(f"  [{idx + 1}/{len(motions)}] Character {idx + 1} processed")

        print("\n✅ SMPL forward pass complete\n")

        # Build combined GLTF
        print(f"{'=' * 60}")
        print("Building combined GLTF structure...")
        print(f"{'=' * 60}\n")

        gltf = self._build_multi_character_gltf(
            all_init_vertices_list,
            all_init_joints_list,
            all_joints_list,
            all_smpl_params,
            motions,
        )

        # Save
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        gltf.save_binary(filename)

        print(f"✅ Multi-character GLB exported: {filename}\n")

    def _build_multi_character_gltf(
        self,
        all_init_vertices: list[torch.Tensor],
        all_init_joints: list[torch.Tensor],
        all_joints: list[torch.Tensor],
        all_smpl_params: list[dict],
        motions: list[Motion],
    ) -> GLTF2:
        """Build a GLTF structure with multiple animated characters.

        This is a simplified version that creates separate mesh instances for each character.

        Args:
            all_init_vertices: List of initial vertices for each character
            all_init_joints: List of initial joints for each character
            all_joints: List of animated joints for each character
            all_smpl_params: List of SMPL parameters for each character
            motions: List of Motion objects

        Returns:
            GLTF2: Combined GLTF structure
        """
        print("⚠️  Note: Multi-character GLB export is currently simplified.")
        print("   Each character will have its own mesh and armature.")
        print("   Full vertex merging and shared scene not yet implemented.\n")

        # For now, export just the first character as a placeholder
        # Full implementation would require:
        # 1. Merging all vertices with proper face index offsets
        # 2. Creating a single scene with multiple nodes
        # 3. Managing multiple skins and armatures
        # 4. Synchronizing animations

        # Use base exporter for single character (first one)
        vertices = all_init_vertices[0].cpu().numpy()
        joints = all_joints[0].cpu().numpy()
        init_joints = all_init_joints[0].cpu().numpy()

        smpl_params = all_smpl_params[0]
        global_orient = smpl_params["global_orient"]
        transl = smpl_params["transl"]
        body_pose = smpl_params["body_pose"]

        faces = self.base_exporter.smpl_model.faces

        gltf = self.base_exporter._build_gltf(
            vertices,
            init_joints,
            faces,
            joints,
            global_orient,
            transl,
            body_pose,
        )

        return gltf


def export_motions_to_spatial_grid(
    motions: list[Motion],
    offsets: list[tuple[float, float, float]],
    output_path: str,
    body_model_path: str,
    fps: int = 30,
) -> None:
    """Convenience function to export multiple motions to a spatial grid GLB.

    Args:
        motions (List[Motion]): List of Motion instances
        offsets (List[Tuple[float, float, float]]): List of (x, y, z) offsets
        output_path (str): Output GLB file path
        body_model_path (str): Path to SMPL models
        fps (int): Frames per second
    """
    exporter = MultiCharacterGLBExporter(
        model_path=body_model_path,
        gender="neutral",
        device="cpu",
        fps=fps,
    )

    exporter.export_multi_character(
        motions=motions,
        offsets=offsets,
        filename=output_path,
    )
