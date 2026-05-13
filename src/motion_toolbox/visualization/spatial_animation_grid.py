"""Spatial animation grid for displaying multiple motions in a single 3D scene.

This module provides functionality to load multiple motion sequences and arrange them
in a 3D spatial grid within a single Three.js viewer. Each character is loaded as a
separate GLB file and positioned at grid locations in 3D space. All characters share
the same camera and playback controls for synchronized visualization.
"""

import base64
import json
import math
from pathlib import Path

import numpy as np

from motion_toolbox.converters import GVHMRToMotionConverter, MHRToMotionConverter
from motion_toolbox.core.motion import Motion
from motion_toolbox.visualization.glb_exporter import GLBExporter
from motion_toolbox.visualization.template_renderer import (
    TemplateRenderer,
    render_ground_plane_code,
)


class SpatialAnimationGrid:
    """Creates a Three.js viewer with multiple animated characters in a 3D spatial grid.

    This class loads multiple motion sequences, exports each as a separate GLB file,
    and creates an HTML viewer that loads and positions all characters in a single
    3D scene using Three.js.
    """

    def __init__(
        self,
        spacing: float = 2.0,
        use_bounding_box: bool = False,
        body_model_path: str | None = None,
        body_model_type: str = "smpl",
        embed_glbs: bool = True,
        show_ground: bool = True,
        material_type: str = "physical",
        mesh_color: str = "blue",
        checkerboard_floor: bool = True,
    ):
        """Initialize the spatial animation grid.

        Args:
            spacing (float): Fixed spacing between characters in meters (used if use_bounding_box=False)
            use_bounding_box (bool): If True, calculate spacing based on motion bounding boxes
            body_model_path (Optional[str]): Path to body models (required for GLB export)
            body_model_type (str): Type of body model ("smpl", "mhr", etc.) (default: "smpl")
            embed_glbs (bool): If True, embed GLB files as base64 in HTML (default: True)
            show_ground (bool): If True, show ground plane in Three.js viewer (default: True)
            material_type (str): Material type - "phong", "lambert", or "physical" (default: "physical")
            mesh_color (str): Mesh color - "white", "blue", "clay", "grey" (default: "blue")
            checkerboard_floor (bool): If True, use checkerboard pattern for floor (default: True)
        """
        self.spacing = spacing
        self.use_bounding_box = use_bounding_box
        self.body_model_path = body_model_path
        self.body_model_type = body_model_type.lower()
        self.embed_glbs = embed_glbs
        self.show_ground = show_ground
        self.material_type = material_type.lower()
        self.mesh_color = mesh_color.lower()
        self.checkerboard_floor = checkerboard_floor

    def create_spatial_grid(
        self,
        motion_paths: list[str | Path],
        output_dir: str | Path,
        output_html_path: str | Path,
        title: str = "Spatial Animation Grid",
        grid_cols: int | None = None,
        labels: list[str] | None = None,
        details: list[dict[str, str]] | None = None,
    ) -> None:
        """Create a spatial grid of animated characters using Three.js.

        Args:
            motion_paths (List[Union[str, Path]]): Paths to motion files (.pt files)
            output_dir (Union[str, Path]): Directory to save individual GLB files
            output_html_path (Union[str, Path]): HTML viewer output path
            title (str): Title for the HTML viewer
            grid_cols (Optional[int]): Number of columns in grid (auto-calculated if None)
            labels (Optional[List[str]]): Short labels for each animation (displayed above character)
            details (Optional[List[Dict[str, str]]]): Detailed info for each animation (shown on hover/click)
        """
        if not motion_paths:
            raise ValueError("motion_paths must contain at least one motion file")

        if self.body_model_path is None:
            raise ValueError("body_model_path must be provided for GLB export")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load all motions (supports Motion, GVHMR, and MHR formats)
        print(f"\n{'=' * 60}")
        print(f"Loading {len(motion_paths)} motion sequences...")
        print(f"{'=' * 60}")

        motions = []
        motion_names = []

        # Lazy initializers for converters
        gvhmr_converter = None
        mhr_converter = None

        for idx, motion_path in enumerate(motion_paths):
            try:
                import torch

                motion_path_obj = Path(motion_path)
                print(f"  [{idx + 1}/{len(motion_paths)}] Loading {motion_path_obj.name}...")

                # Load file and detect format
                data = torch.load(motion_path, map_location="cpu", weights_only=False)

                if not isinstance(data, dict):
                    print(f"  ⚠️  Skipping {motion_path_obj.name}: invalid format")
                    continue

                is_gvhmr = "smpl_params_global" in data
                is_motion = (
                    "joint_rotations" in data
                )  # Changed from 'rotations' to 'joint_rotations' to match Motion.to_dict
                is_mhr = "model_parameters" in data

                if is_gvhmr:
                    # Convert GVHMR to Motion
                    if gvhmr_converter is None:
                        gvhmr_converter = GVHMRToMotionConverter(body_model_path=str(self.body_model_path))
                    motion = gvhmr_converter.convert(data)
                    print("      Format: GVHMR → Motion")
                elif is_mhr:
                    # Convert MHR to Motion
                    if mhr_converter is None:
                        mhr_converter = MHRToMotionConverter()
                    motion = mhr_converter.convert(data)
                    # Force body model type if detected as MHR input
                    if self.body_model_type == "smpl":
                        print("      Note: MHR input detected, but body_model_type is 'smpl'.")
                    print("      Format: MHR → Motion")
                elif is_motion:
                    # Load Motion directly
                    motion = Motion.from_dict(data)
                    print("      Format: Motion")
                else:
                    print(f"  ⚠️  Skipping {motion_path_obj.name}: unknown format")
                    continue

                motions.append(motion)
                motion_names.append(motion_path_obj.stem)

            except Exception as e:
                print(f"  ⚠️  Failed to load {Path(motion_path).name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        if not motions:
            raise RuntimeError("No motions were successfully loaded")

        print(f"✅ Loaded {len(motions)} motions successfully\n")

        # Calculate grid layout
        num_motions = len(motions)
        if grid_cols is None:
            grid_cols = int(math.ceil(math.sqrt(num_motions)))

        grid_rows = int(math.ceil(num_motions / grid_cols))

        print(f"{'=' * 60}")
        print(f"Grid Layout: {grid_rows} rows × {grid_cols} columns")
        print(f"{'=' * 60}\n")

        # Calculate spacing
        if self.use_bounding_box:
            spacing = self._calculate_bounding_box_spacing(motions)
            print(f"Using bounding box spacing: {spacing:.2f}m\n")
        else:
            spacing = self.spacing
            print(f"Using fixed spacing: {spacing:.2f}m\n")

        # Export each motion as a separate GLB
        print(f"{'=' * 60}")
        print("Exporting individual GLB files...")
        print(f"{'=' * 60}")

        # Initialize exporter with appropriate BodyModel
        if self.body_model_type == "mhr":
            from ..core.body_models.mhr import MHRBodyModel

            body_model = MHRBodyModel(model_path=str(self.body_model_path), lod=1)
        else:
            from ..core.body_models.smpl import SMPLBodyModel

            body_model = SMPLBodyModel(model_path=str(self.body_model_path))

        glb_files = []
        exporter = GLBExporter(
            model_path=str(self.body_model_path),  # Kept for compatibility, but body_model overrides
            body_model=body_model,
            include_floor=False,  # Don't include floor in GLB - we'll add it in Three.js
        )

        for idx, (motion, name) in enumerate(zip(motions, motion_names)):
            glb_filename = f"character_{idx:03d}_{name}.glb"
            glb_path = output_dir / glb_filename

            # Check if motion type matches exporter type
            if motion.body_model_type and motion.body_model_type != self.body_model_type:
                print(
                    f"  ⚠️  Warning: Motion {name} type ({motion.body_model_type}) mismatch with exporter ({self.body_model_type})"
                )

            print(f"  [{idx + 1}/{len(motions)}] Exporting {glb_filename}...")
            try:
                exporter.export(motion, str(glb_path))
                glb_files.append(glb_filename)
            except Exception as e:
                print(f"  ❌ Failed to export {glb_filename}: {e}")

        print(f"\n✅ Exported {len(glb_files)} GLB files\n")

        if not glb_files:
            raise RuntimeError("No GLB files were generated.")

        # Calculate grid positions
        print(f"{'=' * 60}")
        print("Calculating spatial positions...")
        print(f"{'=' * 60}")

        positions = []
        # Only create positions for successfully exported files
        for idx in range(len(glb_files)):
            row = idx // grid_cols
            col = idx % grid_cols

            # Calculate offset (centered grid)
            offset_x = (col - (grid_cols - 1) / 2) * spacing
            offset_z = (row - (grid_rows - 1) / 2) * spacing

            print(f"  Character {idx + 1}: Position ({offset_x:+.1f}, 0.0, {offset_z:+.1f})")
            positions.append((offset_x, 0.0, offset_z))

        print("\n✅ All positions calculated\n")

        # Create HTML viewer with Three.js
        print(f"{'=' * 60}")
        print("Creating Three.js viewer...")
        if self.embed_glbs:
            print("  Mode: Embedded GLBs (base64)")
        else:
            print("  Mode: File references")
        print(f"{'=' * 60}\n")

        self._generate_threejs_viewer(
            glb_files=glb_files,
            positions=positions,
            output_html_path=output_html_path,
            title=title,
            glb_dir=output_dir,
            embed_glbs=self.embed_glbs,
            material_type=self.material_type,
            mesh_color=self.mesh_color,
            checkerboard_floor=self.checkerboard_floor,
            labels=labels,
            details=details,
        )

        print(f"✅ HTML viewer created: {output_html_path}\n")
        print(f"{'=' * 60}")
        print("✅ SPATIAL GRID CREATION COMPLETE")
        print(f"{'=' * 60}\n")

    def _generate_threejs_viewer(
        self,
        glb_files: list[str],
        positions: list[tuple],
        output_html_path: str | Path,
        title: str,
        glb_dir: Path,
        embed_glbs: bool = True,
        material_type: str = "phong",
        mesh_color: str = "white",
        checkerboard_floor: bool = False,
        labels: list[str] | None = None,
        details: list[dict[str, str]] | None = None,
    ) -> None:
        """Generate an HTML viewer with Three.js for the spatial grid.

        Args:
            glb_files (List[str]): List of GLB filenames (relative to glb_dir)
            positions (List[tuple]): List of (x, y, z) positions for each character
            output_html_path (Union[str, Path]): Output path for HTML file
            title (str): Title for the viewer
            glb_dir (Path): Directory containing GLB files
            embed_glbs (bool): If True, embed GLB files as base64 data URIs
            material_type (str): Material type - "phong", "lambert", or "physical"
            mesh_color (str): Mesh color - "white", "blue", "clay", "grey"
            checkerboard_floor (bool): If True, use checkerboard pattern for floor
            labels (Optional[List[str]]): Short labels for each animation
            details (Optional[List[Dict[str, str]]]): Detailed info for each animation
        """
        html_path = Path(output_html_path)

        if embed_glbs:
            # Embed GLB files as base64 data URIs
            print("  Embedding GLB files as base64...")
            glb_data_uris = []
            for idx, glb_filename in enumerate(glb_files):
                glb_path = glb_dir / glb_filename
                with open(glb_path, "rb") as f:
                    glb_bytes = f.read()
                    glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
                    data_uri = f"data:model/gltf-binary;base64,{glb_base64}"
                    glb_data_uris.append(data_uri)
                    print(f"    [{idx + 1}/{len(glb_files)}] Embedded {glb_filename} ({len(glb_bytes):,} bytes)")

            # Create scene data with data URIs
            scene_data = []
            for idx, (uri, pos) in enumerate(zip(glb_data_uris, positions)):
                item = {"glb": uri, "position": list(pos)}
                if labels and idx < len(labels):
                    item["label"] = labels[idx]
                if details and idx < len(details):
                    item["details"] = details[idx]
                scene_data.append(item)
        else:
            # Use file references
            try:
                glb_rel_paths = [str((glb_dir / glb).relative_to(html_path.parent)) for glb in glb_files]
            except ValueError:
                # If relative path fails, use absolute paths
                glb_rel_paths = [str(glb_dir / glb) for glb in glb_files]

            # Create scene data with file paths
            scene_data = []
            for idx, (glb, pos) in enumerate(zip(glb_rel_paths, positions)):
                item = {"glb": glb, "position": list(pos)}
                if labels and idx < len(labels):
                    item["label"] = labels[idx]
                if details and idx < len(details):
                    item["details"] = details[idx]
                scene_data.append(item)

        # Color mapping
        color_map = {
            "white": "0xffffff",
            "grey": "0xcccccc",
            "gray": "0xcccccc",
            "blue": "0x6495ED",  # Cornflower blue (original)
            "clay": "0xE5C29F",  # Light clay/tan color
        }
        mesh_color_hex = color_map.get(mesh_color, "0xffffff")

        # Generate ground plane code based on checkerboard setting
        ground_plane_code = render_ground_plane_code(checkerboard_floor)

        # Prepare template context
        context = {
            "TITLE": title,
            "SCENE_DATA": json.dumps(scene_data, indent=8),
            "MATERIAL_TYPE": material_type,
            "MESH_COLOR_HEX": mesh_color_hex,
            "GROUND_PLANE_CODE": ground_plane_code,
        }

        # Render template
        renderer = TemplateRenderer()
        html_content = renderer.render("spatial_grid_viewer.html", context)

        output_html_path = Path(output_html_path)
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        output_html_path.write_text(html_content)

    def _calculate_bounding_box_spacing(self, motions: list[Motion]) -> float:
        """Calculate spacing based on motion bounding boxes.

        Args:
            motions (List[Motion]): List of motion sequences

        Returns:
            float: Recommended spacing in meters
        """
        max_extent = 0.0

        for motion in motions:
            # Get positions for all joints across all frames
            # positions property computes positions using FK
            positions = motion.positions  # (T, J, 3)

            # Calculate bounding box
            min_pos = positions.min(axis=(0, 1))
            max_pos = positions.max(axis=(0, 1))

            extent = np.linalg.norm(max_pos - min_pos)
            max_extent = max(max_extent, extent)

        # Add 50% margin
        return max_extent * 1.5


def create_spatial_animation_grid(
    motion_paths: list[str | Path],
    output_dir: str | Path,
    output_html_path: str | Path,
    title: str = "Spatial Animation Grid",
    spacing: float = 2.0,
    use_bounding_box: bool = False,
    grid_cols: int | None = None,
    body_model_path: str | None = None,
    body_model_type: str = "smpl",
    embed_glbs: bool = True,
    material_type: str = "physical",
    mesh_color: str = "blue",
    checkerboard_floor: bool = True,
    labels: list[str] | None = None,
    details: list[dict[str, str]] | None = None,
) -> None:
    """Convenience function to create a spatial animation grid with Three.js.

    Args:
        motion_paths (List[Union[str, Path]]): Paths to motion files (.pt files)
        output_dir (Union[str, Path]): Directory to save individual GLB files
        output_html_path (Union[str, Path]): HTML viewer output path
        title (str): Title for HTML viewer
        spacing (float): Spacing between characters in meters
        use_bounding_box (bool): Calculate spacing from bounding boxes
        grid_cols (Optional[int]): Number of columns (auto-calculated if None)
        body_model_path (Optional[str]): Path to body models
        body_model_type (str): Type of body model ("smpl", "mhr")
        embed_glbs (bool): If True, embed GLBs as base64 in HTML (default: True)
        material_type (str): Material type - "phong", "lambert", or "physical" (default: "physical")
        mesh_color (str): Mesh color - "white", "blue", "clay", "grey" (default: "blue")
        checkerboard_floor (bool): If True, use checkerboard pattern for floor (default: True)
        labels (Optional[List[str]]): Short labels for each animation (displayed above character)
        details (Optional[List[Dict[str, str]]]): Detailed info for each animation (shown on hover/click)
    """
    grid = SpatialAnimationGrid(
        spacing=spacing,
        use_bounding_box=use_bounding_box,
        body_model_path=body_model_path,
        body_model_type=body_model_type,
        embed_glbs=embed_glbs,
        material_type=material_type,
        mesh_color=mesh_color,
        checkerboard_floor=checkerboard_floor,
    )

    grid.create_spatial_grid(
        motion_paths=motion_paths,
        output_dir=output_dir,
        output_html_path=output_html_path,
        title=title,
        grid_cols=grid_cols,
        labels=labels,
        details=details,
    )


def create_spatial_animation_grid_from_config(config_path: str):
    """Create a spatial animation grid from a configuration file.

    Args:
        config_path (str): Path to YAML or JSON configuration file

    Example:
        >>> create_spatial_animation_grid_from_config("examples/simple_grid.yaml")
    """
    from motion_toolbox.visualization.grid_config import GridConfig

    # Load configuration
    config = GridConfig.from_file(config_path)

    # Extract animation paths, labels, and details
    motion_paths = []
    labels = []
    details = []

    for anim in config.animations:
        if isinstance(anim, dict):
            motion_paths.append(anim["path"])
            if "label" in anim:
                labels.append(anim["label"])
            if "details" in anim:
                details.append(anim["details"])
        else:
            # Simple string path
            motion_paths.append(anim)

    # Use empty lists if no labels/details were provided
    labels = labels if labels else None
    details = details if details else None

    # Create output path
    output_path = Path(config.output_dir) / config.output_html
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect body model type from config if available, otherwise default to smpl
    # GridConfig might need an update to support body_model_type, currently we default to "smpl"
    # or we check if config has that attribute (it's dynamic)
    body_model_type = getattr(config, "body_model_type", "smpl")

    # Create grid
    create_spatial_animation_grid(
        motion_paths=motion_paths,
        output_dir=config.output_dir,
        output_html_path=str(output_path),
        title=config.title,
        spacing=config.spacing,
        use_bounding_box=False,
        grid_cols=config.grid_cols,
        body_model_path=config.body_model_path,
        body_model_type=body_model_type,
        embed_glbs=config.embed_glbs,
        material_type=config.material_type,
        mesh_color=config.mesh_color,
        checkerboard_floor=config.checkerboard_floor,
        labels=labels,
        details=details,
    )

    print("\n✅ Grid created successfully!")
    print(f"   Output: {output_path.absolute()}")
    return output_path
