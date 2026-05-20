#!/usr/bin/env python3
"""GVHMR Interactive 3D Motion Visualization Script.

Loads GVHMR data, converts it to Motion class, and creates interactive 3D visualizations.
This script demonstrates the complete pipeline from GVHMR to Motion to GLB/HTML visualization.

Features:
- Converts GVHMR output to Motion objects
- Exports motion as GLB files with SMPL body mesh
- Creates interactive HTML viewers with playback controls
- Processes all frames for complete motion visualization

Usage:
    python scripts/gvhmr_motion_visualization.py

Requirements:
    - SMPL body models (downloaded and set up)
    - GLB export dependencies (smplx, pygltflib)
"""

import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Import from the motion_toolbox package
from motion_toolbox.core.motion import Motion
from motion_toolbox.visualization import GLB_AVAILABLE, GLBExporter, MotionVisualizer


def load_and_convert_gvhmr(
    data_path: str, max_frames: int | None = None, body_model_path: str | None = None
) -> Motion | None:
    """Load GVHMR file and convert to Motion.

    Args:
        data_path: Path to GVHMR .pt file
        max_frames: Maximum number of frames to load (None for all)
        body_model_path: Path to SMPL body models for forward kinematics

    Returns:
        Motion object or None if loading failed
    """
    logger.info(f"Loading GVHMR file: {Path(data_path).name}")

    try:
        # Load the data
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        logger.info(f"Data type: {type(data)}")

        if isinstance(data, dict) and "smpl_params_global" in data:
            smpl_params = data["smpl_params_global"]
            logger.info(f"SMPL params keys: {list(smpl_params.keys())}")

            # Log shapes
            body_pose = smpl_params["body_pose"]
            global_orient = smpl_params["global_orient"]
            transl = smpl_params["transl"]

            logger.info(f"Body pose shape: {body_pose.shape}")
            logger.info(f"Global orient shape: {global_orient.shape}")
            logger.info(f"Translation shape: {transl.shape}")

            # Use the GVHMR converter
            # Get converter with proper configuration
            from motion_toolbox.converters import GVHMRToMotionConverter

            converter = GVHMRToMotionConverter(max_frames=max_frames, body_model_path=body_model_path)
            motion = converter.convert(data)

            logger.info(f"✅ Created Motion: {motion.num_frames} frames, {motion.num_joints} joints")
            return motion

        else:
            logger.error(f"Expected dict with 'smpl_params_global', got {type(data)}")
            if isinstance(data, dict):
                logger.error(f"Available keys: {list(data.keys())}")
            return None

    except Exception as e:
        logger.error(f"Error loading GVHMR file: {e}")
        return None


def visualize_motion(motion: Motion, output_dir: str, filename: str, body_model_path: str | None = None):
    """Create interactive 3D motion visualization.

    Args:
        motion: Motion object to visualize
        output_dir: Output directory for files
        filename: Base filename for outputs
        body_model_path: Path to SMPL body models for GLB export
    """
    logger.info(f"Creating interactive 3D visualization for {filename}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not GLB_AVAILABLE:
        logger.warning("GLB export not available - skipping 3D visualization")
        logger.info("💡 Install GLB dependencies for interactive 3D visualization")
        return

    if not body_model_path or not Path(body_model_path).exists():
        logger.warning("Body models not available - skipping 3D visualization")
        logger.info("💡 Set up SMPL body models for 3D mesh visualization")
        return

    try:
        # Create GLB exporter
        logger.info("🎭 Exporting motion to GLB format...")
        exporter = GLBExporter(model_path=body_model_path, gender="neutral")

        # Export motion to GLB
        glb_path = output_path / f"{filename}_animation.glb"
        exporter.export(motion, str(glb_path))
        logger.info(f"✅ GLB animation created: {glb_path}")

        # Create interactive HTML viewer
        logger.info("🌐 Creating interactive HTML viewer...")
        motion_visualizer = MotionVisualizer()
        html_path = output_path / f"{filename}_interactive.html"

        motion_visualizer.create_standalone_html(
            glb_path=str(glb_path),
            output_path=str(html_path),
            title=f"{filename} - Motion Animation ({motion.num_frames} frames)",
            width="100%",
            height="700px",
        )
        logger.info(f"✅ Interactive viewer created: {html_path}")
        logger.info(f"🎬 Open {html_path} in your browser to view the 3D animation!")

    except Exception as e:
        logger.error(f"Failed to create 3D visualization: {e}")
        logger.info("💡 Ensure SMPL body models are available and GLB dependencies are installed")


def main():
    """Main script function."""
    logger.info("🎬 GVHMR Interactive Motion Visualization Script")
    logger.info("=" * 55)

    # Configuration
    sample_data_dir = Path("motion_toolbox/sample_data/gvhmr")
    output_dir = Path("motion_toolbox/outputs/interactive_visualizations")
    max_frames = None  # Process all frames for full visualization

    # Body model path for 3D visualization
    body_model_path = Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl"

    # Check if body models are available
    if not body_model_path.exists():
        logger.warning(f"Body models not found at: {body_model_path}")
        logger.info("💡 Download SMPL models and run setup script for 3D visualization")

    # Process all GVHMR files
    gvhmr_files = list(sample_data_dir.glob("*.pt"))

    if not gvhmr_files:
        logger.error(f"No GVHMR files found in {sample_data_dir}")
        logger.info(f"💡 Make sure GVHMR data files are in: {sample_data_dir}")
        return

    logger.info(f"Found {len(gvhmr_files)} GVHMR files")
    logger.info(f"Output directory: {output_dir}")

    successful_visualizations = 0

    for gvhmr_file in gvhmr_files:
        logger.info(f"\n🎯 Processing: {gvhmr_file.name}")
        logger.info("-" * 40)

        # Load and convert
        motion = load_and_convert_gvhmr(
            str(gvhmr_file),
            max_frames=max_frames,
            body_model_path=str(body_model_path) if body_model_path.exists() else None,
        )

        if motion is None:
            logger.warning(f"Skipping {gvhmr_file.name} due to loading error")
            continue

        # Create filename from gvhmr file
        base_filename = f"gvhmr_{gvhmr_file.stem}"

        # Visualize
        try:
            visualize_motion(
                motion,
                str(output_dir),
                base_filename,
                body_model_path=str(body_model_path) if body_model_path.exists() else None,
            )
            successful_visualizations += 1
        except Exception as e:
            logger.error(f"Failed to visualize {gvhmr_file.name}: {e}")

    logger.info("\n" + "=" * 55)
    logger.info(f"✅ {successful_visualizations}/{len(gvhmr_files)} visualizations completed!")
    if successful_visualizations > 0:
        logger.info(f"📁 Interactive 3D animations available in: {output_dir}")
        logger.info("🌐 Open the HTML files in your browser to view animations!")
    else:
        logger.warning("No visualizations were created. Check body model setup.")


if __name__ == "__main__":
    main()
