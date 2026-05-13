#!/usr/bin/env python3
"""Universal Motion Visualization Script.

Loads motion data in multiple formats (GVHMR, SMPL), converts to Motion class,
and creates interactive 3D visualizations.

Supported formats:
- GVHMR: .pt files with SMPL parameters (smpl_params_global, etc.)
- SMPL: .npz files with pose_body, root_orient, trans, etc.

Features:
- Auto-detects input format based on file extension and content
- Converts motion data to Motion objects
- Exports motion as GLB files with SMPL body mesh
- Creates interactive HTML viewers with playback controls
- Processes all frames for complete motion visualization

Usage:
    python scripts/motion_visualization.py --data-file sample_data/gvhmr/gvhmr_1.pt
    python scripts/motion_visualization.py --data-file sample_data/smpl/metrabs_1.npz

Requirements:
    - SMPL body models (downloaded and set up)
    - GLB export dependencies (smplx, pygltflib)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motion_toolbox.converters.motion_converters import GVHMRToMotionConverter, SMPLToMotionConverter
from motion_toolbox.core.motion import Motion
from motion_toolbox.visualization.glb_exporter import GLBExporter
from motion_toolbox.visualization.html_viewer import MotionVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def detect_format(data_file: str) -> str:
    """Detect the format of the input data file.

    Args:
        data_file: Path to the data file

    Returns:
        Format string: "gvhmr", "smpl", or "unknown"
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Check file extension first
    ext = Path(data_file).suffix.lower()

    if ext == ".pt":
        # Likely GVHMR format
        try:
            data = torch.load(data_file, map_location="cpu")
            if isinstance(data, dict) and "smpl_params_global" in data:
                return "gvhmr"
            else:
                logger.warning(f"PT file {data_file} doesn't contain expected GVHMR keys")
                return "unknown"
        except Exception as e:
            logger.warning(f"Failed to load PT file {data_file}: {e}")
            return "unknown"

    elif ext == ".npz":
        # Likely SMPL format
        try:
            data = np.load(data_file)
            required_keys = ["pose_body", "root_orient", "trans"]
            if all(key in data for key in required_keys):
                return "smpl"
            else:
                logger.warning(f"NPZ file {data_file} doesn't contain expected SMPL keys: {required_keys}")
                return "unknown"
        except Exception as e:
            logger.warning(f"Failed to load NPZ file {data_file}: {e}")
            return "unknown"

    logger.warning(f"Unknown file extension: {ext}")
    return "unknown"


def load_motion_data(data_file: str, body_model_path: str | None = None) -> Motion:
    """Load motion data from file and convert to Motion object.

    Args:
        data_file: Path to the data file
        body_model_path: Path to SMPL body models directory

    Returns:
        Motion object

    Raises:
        ValueError: If format is unsupported or conversion fails
    """
    # Detect format
    format_type = detect_format(data_file)
    logger.info(f"Detected format: {format_type}")

    if format_type == "gvhmr":
        # Load GVHMR data
        logger.info("Loading GVHMR data...")
        converter = GVHMRToMotionConverter(body_model_path=body_model_path)
        motion = converter.convert(data_file)
        logger.info(f"✓ GVHMR motion loaded: {motion.num_frames} frames, {motion.skeleton.num_joints} joints")

    elif format_type == "smpl":
        # Load SMPL data
        logger.info("Loading SMPL data...")
        converter = SMPLToMotionConverter(body_model_path=body_model_path)
        motion = converter.convert(data_file)
        logger.info(f"✓ SMPL motion loaded: {motion.num_frames} frames, {motion.skeleton.num_joints} joints")

    else:
        raise ValueError(f"Unsupported format: {format_type}. Supported formats: GVHMR (.pt), SMPL (.npz)")

    return motion


def create_output_paths(data_file: str, output_dir: str = "outputs/visualization") -> tuple[str, str]:
    """Create output paths for GLB and HTML files.

    Args:
        data_file: Input data file path
        output_dir: Output directory

    Returns:
        Tuple of (glb_path, html_path)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get base name without extension
    base_name = Path(data_file).stem

    glb_path = os.path.join(output_dir, f"{base_name}_motion.glb")
    html_path = os.path.join(output_dir, f"{base_name}_viewer.html")

    return glb_path, html_path


def export_to_glb(motion: Motion, glb_path: str, body_model_path: str) -> None:
    """Export motion to GLB file.

    Args:
        motion: Motion object to export
        glb_path: Output GLB file path
        body_model_path: Path to SMPL body models
    """
    logger.info("🎬 Exporting motion to GLB...")

    try:
        exporter = GLBExporter(model_path=body_model_path)
        exporter.export(motion, glb_path)
        logger.info(f"✓ GLB exported: {glb_path}")

    except Exception as e:
        logger.error(f"❌ GLB export failed: {e}")
        raise


def create_html_viewer(glb_path: str, html_path: str, motion: Motion) -> None:
    """Create HTML viewer for the GLB file.

    Args:
        glb_path: Path to GLB file
        html_path: Output HTML file path
        motion: Motion object for metadata
    """
    logger.info("🌐 Creating HTML viewer...")

    try:
        visualizer = MotionVisualizer()

        # Motion information
        motion_info = {
            "frames": motion.num_frames,
            "duration": f"{motion.duration:.2f}s",
            "fps": motion.fps,
            "joints": motion.skeleton.num_joints,
            "format": getattr(motion, "metadata", {}).get("format", "motion"),
            "up_axis": motion.up_axis,
        }

        visualizer.create_standalone_html(
            glb_path=glb_path,
            output_path=html_path,
            title=f"Motion Visualization - {Path(glb_path).stem}",
        )

        logger.info(f"✓ HTML viewer created: {html_path}")

    except Exception as e:
        logger.error(f"❌ HTML viewer creation failed: {e}")
        raise


def main():
    """Main visualization pipeline."""
    parser = argparse.ArgumentParser(description="Universal Motion Visualization")
    parser.add_argument(
        "--data-file", type=str, required=True, help="Path to motion data file (.pt for GVHMR, .npz for SMPL)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/visualization", help="Output directory for GLB and HTML files"
    )
    parser.add_argument(
        "--body-model-path",
        type=str,
        default=str(Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl"),
        help="Path to SMPL body models directory",
    )
    parser.add_argument("--skip-glb", action="store_true", help="Skip GLB export (only load and validate data)")

    args = parser.parse_args()

    logger.info("🚀 Universal Motion Visualization Pipeline")
    logger.info("=" * 60)

    try:
        # Load motion data
        logger.info(f"📁 Loading motion data: {args.data_file}")
        motion = load_motion_data(args.data_file, args.body_model_path)

        logger.info("📊 Motion Summary:")
        logger.info(f"  Duration: {motion.duration:.2f}s at {motion.fps}fps")
        logger.info(f"  Frames: {motion.num_frames}")
        logger.info(f"  Joints: {motion.skeleton.num_joints}")
        logger.info(f"  Format: {getattr(motion, 'metadata', {}).get('format', 'motion')}")
        logger.info(f"  Up axis: {motion.up_axis}")

        if args.skip_glb:
            logger.info("✓ Data loaded successfully (GLB export skipped)")
            return

        # Create output paths
        glb_path, html_path = create_output_paths(args.data_file, args.output_dir)

        # Check if body models exist
        if not os.path.exists(args.body_model_path):
            logger.warning(f"⚠️  Body model path not found: {args.body_model_path}")
            logger.warning("GLB export requires SMPL body models. Download from:")
            logger.warning("https://smpl.is.tue.mpg.de/download.php")
            return

        # Export to GLB
        export_to_glb(motion, glb_path, args.body_model_path)

        # Create HTML viewer
        create_html_viewer(glb_path, html_path, motion)

        logger.info("\n🎉 Visualization complete!")
        logger.info(f"📁 GLB file: {glb_path}")
        logger.info(f"🌐 HTML viewer: {html_path}")
        logger.info(f"👁️  Open {html_path} in a web browser to view the motion")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
