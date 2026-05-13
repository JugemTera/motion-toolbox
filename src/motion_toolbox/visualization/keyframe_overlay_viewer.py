"""Keyframe overlay visualization for motion sequences.

This module provides functionality to visualize a motion sequence with overlaid
keyframe poses. Keyframes are positioned at their actual spatial locations in the
sequence, creating a visual "trail" effect that shows the motion progression.
"""

import json
from pathlib import Path

import numpy as np
import smplx
import torch

from motion_toolbox.converters import GVHMRToMotionConverter, MotionToSMPLConverter
from motion_toolbox.core.motion import Motion
from motion_toolbox.visualization.template_renderer import (
    TemplateRenderer,
    render_ground_plane_code,
)


class KeyframeOverlayViewer:
    """Creates a Three.js viewer with animated character and overlaid keyframe poses.

    This class loads a motion sequence, samples keyframes uniformly (or using other
    strategies), and creates an HTML viewer that displays the main animation with
    semi-transparent static keyframe poses overlaid at their actual spatial positions.
    """

    def __init__(
        self,
        num_keyframes: int = 10,
        keyframe_opacity: float = 0.5,
        sampling_strategy: str = "uniform",
        body_model_path: str | None = None,
        material_type: str = "physical",
        mesh_color: str = "blue",
        checkerboard_floor: bool = True,
    ):
        """Initialize the keyframe overlay viewer.

        Args:
            num_keyframes (int): Number of keyframes to sample (default: 10)
            keyframe_opacity (float): Initial opacity for keyframes 0-1 (default: 0.5)
            sampling_strategy (str): Strategy for selecting keyframes (default: "uniform")
            body_model_path (Optional[str]): Path to SMPL body models (required)
            material_type (str): Material type - "phong", "lambert", or "physical" (default: "physical")
            mesh_color (str): Mesh color - "white", "blue", "clay", "grey" (default: "blue")
            checkerboard_floor (bool): If True, use checkerboard pattern for floor (default: True)
        """
        self.num_keyframes = num_keyframes
        self.keyframe_opacity = keyframe_opacity
        self.sampling_strategy = sampling_strategy
        self.body_model_path = body_model_path
        self.material_type = material_type.lower()
        self.mesh_color = mesh_color.lower()
        self.checkerboard_floor = checkerboard_floor

    def create_keyframe_overlay(
        self,
        motion_path: str | Path,
        output_html_path: str | Path,
        title: str = "Keyframe Overlay Viewer",
    ) -> None:
        """Create a keyframe overlay visualization.

        Args:
            motion_path (Union[str, Path]): Path to motion file (.pt file)
            output_html_path (Union[str, Path]): HTML viewer output path
            title (str): Title for the HTML viewer
        """
        if self.body_model_path is None:
            raise ValueError("body_model_path must be provided")

        # Load motion
        print(f"\n{'=' * 60}")
        print("Loading motion sequence...")
        print(f"{'=' * 60}")

        motion = self._load_motion(motion_path)

        print(f"✅ Loaded motion with {motion.num_frames} frames")
        print(f"   Framerate: {motion.framerate} fps")
        print(f"   Duration: {motion.num_frames / motion.framerate:.2f}s\n")

        # Sample keyframes
        print(f"{'=' * 60}")
        print("Sampling keyframes...")
        print(f"{'=' * 60}")

        keyframe_indices = self._sample_keyframes(motion.num_frames, self.num_keyframes)

        print(f"✅ Sampled {len(keyframe_indices)} keyframes using '{self.sampling_strategy}' strategy:")
        for i, idx in enumerate(keyframe_indices):
            time = idx / motion.framerate
            print(f"   Keyframe {i + 1}: Frame {idx} (t={time:.2f}s)")
        print()

        # Pre-compute vertex positions
        print(f"{'=' * 60}")
        print("Pre-computing vertex positions...")
        print(f"{'=' * 60}")

        motion_data = self._extract_motion_data(motion, keyframe_indices)

        print("✅ Pre-computed vertices for all frames")
        print(f"   Animation vertices: {motion_data['animation_vertices'].shape}")
        print(f"   Keyframe vertices: {motion_data['keyframe_vertices'].shape}")
        print(f"   Data size: ~{self._estimate_data_size(motion_data):.1f} MB\n")

        # Generate HTML viewer
        print(f"{'=' * 60}")
        print("Creating Three.js viewer...")
        print(f"{'=' * 60}\n")

        self._generate_keyframe_viewer(
            motion_data=motion_data,
            output_html_path=output_html_path,
            title=title,
            framerate=motion.framerate,
        )

        print(f"✅ HTML viewer created: {output_html_path}\n")
        print(f"{'=' * 60}")
        print("✅ KEYFRAME OVERLAY CREATION COMPLETE")
        print(f"{'=' * 60}\n")

    def _load_motion(self, motion_path: str | Path) -> Motion:
        """Load a motion file (supports Motion and GVHMR formats).

        Args:
            motion_path (Union[str, Path]): Path to motion file

        Returns:
            Motion: Loaded motion object
        """
        motion_path_obj = Path(motion_path)

        # Load file and detect format
        data = torch.load(motion_path)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid motion file format: {motion_path}")

        is_gvhmr = "smpl_params_global" in data
        is_motion = "rotations" in data

        if is_gvhmr:
            # Convert GVHMR to Motion
            converter = GVHMRToMotionConverter(body_model_path=str(self.body_model_path))
            motion = converter.convert(str(motion_path))
            print("   Format: GVHMR → Motion")
        elif is_motion:
            # Load Motion directly
            motion = Motion.from_dict(data)
            print("   Format: Motion")
        else:
            raise ValueError(f"Unknown motion format: {motion_path}")

        return motion

    def _sample_keyframes(self, num_frames: int, num_keyframes: int) -> np.ndarray:
        """Sample keyframe indices from the motion sequence.

        Args:
            num_frames (int): Total number of frames in the motion
            num_keyframes (int): Number of keyframes to sample

        Returns:
            np.ndarray: Array of keyframe indices
        """
        if self.sampling_strategy == "uniform":
            return self._sample_uniform(num_frames, num_keyframes)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _sample_uniform(self, num_frames: int, num_keyframes: int) -> np.ndarray:
        """Sample keyframes uniformly across the motion sequence.

        Args:
            num_frames (int): Total number of frames
            num_keyframes (int): Number of keyframes to sample

        Returns:
            np.ndarray: Array of keyframe indices
        """
        return np.linspace(0, num_frames - 1, num_keyframes, dtype=int)

    def _extract_motion_data(self, motion: Motion, keyframe_indices: np.ndarray) -> dict:
        """Extract and pre-compute vertex positions for animation and keyframes.

        Args:
            motion (Motion): Motion object
            keyframe_indices (np.ndarray): Indices of keyframes

        Returns:
            dict: Dictionary containing pre-computed vertices and metadata
        """
        # Convert Motion to SMPL parameters
        converter = MotionToSMPLConverter()
        smpl_params = converter.convert(motion)

        # Create SMPL model
        device = torch.device("cpu")
        smpl_model = smplx.create(
            self.body_model_path,
            model_type="smpl",
            gender="neutral",
            ext="npz",
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        ).to(device)

        # Extract parameters
        betas = smpl_params["betas"].to(device)
        body_pose = smpl_params["body_pose"].to(device)
        global_orient = smpl_params["global_orient"].to(device)
        transl = smpl_params["transl"].to(device)

        # Ensure betas has correct shape for all frames
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).repeat(motion.num_frames, 1)

        # Pre-compute all vertex positions
        with torch.no_grad():
            smpl_output = smpl_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )
            all_vertices = smpl_output.vertices.cpu().numpy()  # (T, V, 3)

        # Get faces from SMPL model
        faces = smpl_model.faces.astype(np.int32)

        # Extract keyframe data
        keyframe_vertices = all_vertices[keyframe_indices]  # (K, V, 3)
        # Note: keyframe_vertices already include translation from SMPL,
        # so they are positioned at their actual spatial locations

        return {
            "animation_vertices": all_vertices.astype(np.float32),
            "keyframe_vertices": keyframe_vertices.astype(np.float32),
            "keyframe_indices": keyframe_indices.tolist(),
            "faces": faces.tolist(),
            "num_frames": motion.num_frames,
        }

    def _estimate_data_size(self, motion_data: dict) -> float:
        """Estimate the size of motion data in MB.

        Args:
            motion_data (dict): Motion data dictionary

        Returns:
            float: Estimated size in megabytes
        """
        anim_verts = motion_data["animation_vertices"]
        kf_verts = motion_data["keyframe_vertices"]

        # Size in bytes (float32 = 4 bytes)
        size_bytes = (anim_verts.size + kf_verts.size) * 4

        # Convert to MB
        return size_bytes / (1024 * 1024)

    def _generate_keyframe_viewer(
        self,
        motion_data: dict,
        output_html_path: str | Path,
        title: str,
        framerate: float,
    ) -> None:
        """Generate an HTML viewer with Three.js for keyframe overlay.

        Args:
            motion_data (dict): Pre-computed motion data
            output_html_path (Union[str, Path]): Output path for HTML file
            title (str): Title for the viewer
            framerate (float): Animation framerate
        """
        html_path = Path(output_html_path)

        # Color mapping
        color_map = {
            "white": "0xffffff",
            "grey": "0xcccccc",
            "gray": "0xcccccc",
            "blue": "0x6495ED",  # Cornflower blue (original)
            "clay": "0xE5C29F",  # Light clay/tan color
        }
        mesh_color_hex = color_map.get(self.mesh_color, "0xffffff")

        # Generate ground plane code
        ground_plane_code = render_ground_plane_code(self.checkerboard_floor)

        # Prepare motion data for embedding
        # Convert numpy arrays to lists for JSON serialization
        embedded_data = {
            "animationVertices": motion_data["animation_vertices"].tolist(),
            "keyframeVertices": motion_data["keyframe_vertices"].tolist(),
            "keyframeIndices": motion_data["keyframe_indices"],
            "faces": motion_data["faces"],
            "numFrames": motion_data["num_frames"],
            "framerate": framerate,
        }

        # Prepare template context
        context = {
            "TITLE": title,
            "MOTION_DATA": json.dumps(embedded_data),
            "MATERIAL_TYPE": self.material_type,
            "MESH_COLOR_HEX": mesh_color_hex,
            "GROUND_PLANE_CODE": ground_plane_code,
            "KEYFRAME_OPACITY": self.keyframe_opacity,
        }

        # Render template
        renderer = TemplateRenderer()
        html_content = renderer.render("keyframe_overlay_viewer.html", context)

        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_content)


def create_keyframe_overlay(
    motion_path: str | Path,
    output_html_path: str | Path,
    title: str = "Keyframe Overlay Viewer",
    num_keyframes: int = 10,
    keyframe_opacity: float = 0.5,
    sampling_strategy: str = "uniform",
    body_model_path: str | None = None,
    material_type: str = "physical",
    mesh_color: str = "blue",
    checkerboard_floor: bool = True,
) -> None:
    """Convenience function to create a keyframe overlay visualization.

    Args:
        motion_path (Union[str, Path]): Path to motion file (.pt file)
        output_html_path (Union[str, Path]): HTML viewer output path
        title (str): Title for HTML viewer
        num_keyframes (int): Number of keyframes to sample (default: 10)
        keyframe_opacity (float): Initial opacity for keyframes 0-1 (default: 0.5)
        sampling_strategy (str): Strategy for keyframe sampling (default: "uniform")
        body_model_path (Optional[str]): Path to SMPL models
        material_type (str): Material type - "phong", "lambert", or "physical" (default: "physical")
        mesh_color (str): Mesh color - "white", "blue", "clay", "grey" (default: "blue")
        checkerboard_floor (bool): If True, use checkerboard pattern for floor (default: True)
    """
    viewer = KeyframeOverlayViewer(
        num_keyframes=num_keyframes,
        keyframe_opacity=keyframe_opacity,
        sampling_strategy=sampling_strategy,
        body_model_path=body_model_path,
        material_type=material_type,
        mesh_color=mesh_color,
        checkerboard_floor=checkerboard_floor,
    )

    viewer.create_keyframe_overlay(
        motion_path=motion_path,
        output_html_path=output_html_path,
        title=title,
    )
