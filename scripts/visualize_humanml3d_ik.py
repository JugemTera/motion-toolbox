#!/usr/bin/env python3
"""Generate HTML visualizations for HumanML3D using IK-based converter (positions → rotations)."""

import logging
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from motion_toolbox.converters.humanml3d_positions import HumanML3DPositionsToMotionConverter
from motion_toolbox.visualization import GLBExporter, MotionVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_sample(sample_path: Path, output_dir: Path, body_model_path: str):
    """Process a single HumanML3D sample using IK-based converter."""
    sample_name = sample_path.stem
    logger.info(f"Processing {sample_name}...")

    # Load text description
    text_path = sample_path.parent / f"{sample_name}.txt"
    if text_path.exists():
        with open(text_path) as f:
            description = f.readline().split("#")[0]
    else:
        description = "No description available"

    # Load and denormalize
    data_263 = np.load(sample_path)
    mean = np.load(sample_path.parent / "Mean.npy")
    std = np.load(sample_path.parent / "Std.npy")
    data_263 = data_263 * std + mean

    # Convert using IK-based approach (positions → rotations)
    converter = HumanML3DPositionsToMotionConverter(framerate=20.0, body_model_path=body_model_path)
    motion = converter.convert(data_263)

    # Verify height
    positions = motion.positions
    mid_frame = len(positions) // 2
    height = np.linalg.norm(positions[mid_frame, 15] - positions[mid_frame, 0])

    # Create visualizations
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export GLB
    exporter = GLBExporter(model_path=body_model_path, gender="neutral")
    glb_path = output_dir / f"{sample_name}.glb"
    exporter.export(motion, str(glb_path))

    # Create HTML viewer
    visualizer = MotionVisualizer()
    html_path = output_dir / f"{sample_name}.html"
    visualizer.create_standalone_html(
        glb_path=str(glb_path),
        output_path=str(html_path),
        title=f"HumanML3D: {sample_name} - {description} [IK-based]",
        width="100%",
        height="700px",
    )

    logger.info(f"✅ Created: {html_path}")
    logger.info(f"   Description: {description}")
    logger.info(f"   Height: {height:.4f}m, Frames: {motion.num_frames}")

    return html_path


def main():
    """Generate visualizations for HumanML3D samples using IK converter."""
    sample_dir = Path("motion_toolbox/sample_data/humanml3d")
    output_dir = Path("motion_toolbox/outputs/humanml3d_ik_visualizations")
    body_model_path = str(Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl")

    # Get all sample files
    sample_files = sorted([f for f in sample_dir.glob("*.npy") if f.name not in ["Mean.npy", "Std.npy"]])

    logger.info("=" * 70)
    logger.info("HumanML3D → Motion using IK-based converter")
    logger.info(f"Found {len(sample_files)} samples")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    html_files = []
    for sample_path in sample_files:
        try:
            html_path = process_sample(sample_path, output_dir, body_model_path)
            html_files.append(html_path)
        except Exception as e:
            logger.error(f"Failed to process {sample_path.name}: {e}")
            import traceback

            traceback.print_exc()

    logger.info("=" * 70)
    logger.info(f"✅ Generated {len(html_files)} visualizations")
    logger.info("\n📁 Open these files in your browser:")
    for html_file in html_files:
        logger.info(f"   file://{html_file.absolute()}")


if __name__ == "__main__":
    main()
