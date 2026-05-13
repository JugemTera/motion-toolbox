#!/usr/bin/env python3
"""Test script for debugging HumanML3D visualization issues.

This script loads HumanML3D data, converts it through the pipeline,
and creates visualizations to help identify where the distortion occurs.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from motion_toolbox.converters import (
    Format272ToMotionConverter,
    HumanML3DTo272Converter,
)
from motion_toolbox.core.motion import Motion
from motion_toolbox.visualization import GLB_AVAILABLE, GLBExporter, MotionVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_humanml3d_sample(file_path: str, mean_path: str = None, std_path: str = None) -> np.ndarray:
    """Load a HumanML3D sample file and optionally denormalize it.

    Args:
        file_path: Path to the .npy file containing 263D features
        mean_path: Optional path to Mean.npy for denormalization
        std_path: Optional path to Std.npy for denormalization

    Returns:
        Numpy array of shape (frames, 263)
    """
    logger.info(f"Loading HumanML3D data from: {file_path}")
    data = np.load(file_path)
    logger.info(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
    logger.info(f"Data range: [{data.min():.4f}, {data.max():.4f}], mean: {data.mean():.4f}")

    # Denormalize if mean/std provided
    if mean_path and std_path:
        mean = np.load(mean_path)
        std = np.load(std_path)
        logger.info(f"Denormalizing with mean shape: {mean.shape}, std shape: {std.shape}")
        data = data * std + mean
        logger.info(f"After denormalization - range: [{data.min():.4f}, {data.max():.4f}]")

    return data


def analyze_263d_format(data: np.ndarray):
    """Analyze the 263D HumanML3D format to understand the data structure.

    Format breakdown:
    - [0]: Root velocity Y
    - [1:3]: Local velocity XZ
    - [3:4]: Root Y position
    - [4:67]: Root-relative positions (RIC) for 21 joints (63D)
    - [67:193]: 6D rotation representations for 21 joints (126D)
    - [193:197]: Foot contact labels (4D)
    - [197:263]: Additional features (66D)
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYZING 263D FORMAT")
    logger.info("=" * 60)

    T = data.shape[0]

    # Root motion
    root_v_y = data[:, 0]
    local_v_xz = data[:, 1:3]
    root_y = data[:, 3]

    logger.info("\nRoot Motion:")
    logger.info(f"  Root velocity Y range: [{root_v_y.min():.4f}, {root_v_y.max():.4f}]")
    logger.info(f"  Local velocity XZ range: [{local_v_xz.min():.4f}, {local_v_xz.max():.4f}]")
    logger.info(f"  Root Y range: [{root_y.min():.4f}, {root_y.max():.4f}]")

    # Positions (RIC - root-relative)
    ric_data = data[:, 4:67].reshape(T, 21, 3)
    logger.info("\nRoot-relative positions (RIC):")
    logger.info(f"  Shape: {ric_data.shape}")
    logger.info(f"  Range: [{ric_data.min():.4f}, {ric_data.max():.4f}]")
    logger.info(f"  Mean distance from root: {np.linalg.norm(ric_data, axis=-1).mean():.4f}")

    # Rotations (6D representation)
    rot_data = data[:, 67:193].reshape(T, 21, 6)
    logger.info("\nRotations (6D):")
    logger.info(f"  Shape: {rot_data.shape}")
    logger.info(f"  Range: [{rot_data.min():.4f}, {rot_data.max():.4f}]")

    # Foot contacts
    foot_contacts = data[:, 193:197]
    logger.info("\nFoot contacts:")
    logger.info(f"  Shape: {foot_contacts.shape}")
    logger.info(f"  Range: [{foot_contacts.min():.4f}, {foot_contacts.max():.4f}]")

    return {
        "root_v_y": root_v_y,
        "local_v_xz": local_v_xz,
        "root_y": root_y,
        "ric_data": ric_data,
        "rot_data": rot_data,
        "foot_contacts": foot_contacts,
    }


def convert_humanml3d_to_272(data: np.ndarray) -> np.ndarray:
    """Convert HumanML3D 263D to 272D format."""
    logger.info("\n" + "=" * 60)
    logger.info("CONVERTING: HumanML3D (263D) -> 272D")
    logger.info("=" * 60)

    converter = HumanML3DTo272Converter()
    data_272 = converter.convert(data)

    logger.info(f"Output shape: {data_272.shape}")
    logger.info(f"Output range: [{data_272.min():.4f}, {data_272.max():.4f}]")

    return data_272


def analyze_272d_format(data: np.ndarray):
    """Analyze the 272D motion representation."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYZING 272D FORMAT")
    logger.info("=" * 60)

    T = data.shape[0]

    # Root velocities
    root_vel_xz = data[:, 0:2]
    logger.info("\nRoot velocity XZ:")
    logger.info(f"  Range: [{root_vel_xz.min():.4f}, {root_vel_xz.max():.4f}]")

    # Heading (6D)
    heading_6d = data[:, 2:8]
    logger.info("\nHeading (6D):")
    logger.info(f"  Range: [{heading_6d.min():.4f}, {heading_6d.max():.4f}]")

    # Local positions
    local_pos = data[:, 8:74].reshape(T, 22, 3)
    logger.info("\nLocal positions:")
    logger.info(f"  Shape: {local_pos.shape}")
    logger.info(f"  Range: [{local_pos.min():.4f}, {local_pos.max():.4f}]")
    logger.info(f"  Mean distance from root: {np.linalg.norm(local_pos[:, 1:], axis=-1).mean():.4f}")

    # Local velocities
    local_vel = data[:, 74:140].reshape(T, 22, 3)
    logger.info("\nLocal velocities:")
    logger.info(f"  Range: [{local_vel.min():.4f}, {local_vel.max():.4f}]")

    # Local rotations (6D)
    local_rot = data[:, 140:272].reshape(T, 22, 6)
    logger.info("\nLocal rotations (6D):")
    logger.info(f"  Range: [{local_rot.min():.4f}, {local_rot.max():.4f}]")


def convert_272_to_motion(data: np.ndarray, body_model_path: str = None) -> Motion:
    """Convert 272D to Motion object."""
    logger.info("\n" + "=" * 60)
    logger.info("CONVERTING: 272D -> Motion")
    logger.info("=" * 60)

    converter = Format272ToMotionConverter(
        framerate=20.0,  # HumanML3D uses 20 FPS
        body_model_path=body_model_path,
    )
    motion = converter.convert(data)

    logger.info("Motion created:")
    logger.info(f"  Num frames: {motion.num_frames}")
    logger.info(f"  Num joints: {motion.num_joints}")
    logger.info(f"  Framerate: {motion.framerate}")
    logger.info(f"  Skeleton: {motion.skeleton.__class__.__name__}")

    # Analyze the resulting motion
    positions = motion.positions
    logger.info("\nGlobal joint positions:")
    logger.info(f"  Shape: {positions.shape}")
    logger.info(f"  Range: [{positions.min():.4f}, {positions.max():.4f}]")

    # Check bone lengths
    root_to_head = np.linalg.norm(positions[:, 15, :] - positions[:, 0, :], axis=-1)
    logger.info(f"  Root to head distance: {root_to_head.mean():.4f} ± {root_to_head.std():.4f} m")

    # Check if character is grounded
    min_y = positions[:, :, 1].min(axis=1)
    logger.info(f"  Min Y per frame: mean={min_y.mean():.4f}, min={min_y.min():.4f}, max={min_y.max():.4f}")

    return motion


def visualize_motion(motion: Motion, output_dir: Path, filename: str, body_model_path: str = None):
    """Create visualizations of the motion."""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot joint trajectories
    logger.info("Creating joint trajectory plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    positions = motion.positions

    # Root trajectory (XZ plane)
    axes[0, 0].plot(positions[:, 0, 0], positions[:, 0, 2], "b-", linewidth=2)
    axes[0, 0].scatter(positions[0, 0, 0], positions[0, 0, 2], c="g", s=100, marker="o", label="Start")
    axes[0, 0].scatter(positions[-1, 0, 0], positions[-1, 0, 2], c="r", s=100, marker="X", label="End")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Z (m)")
    axes[0, 0].set_title("Root Trajectory (Top-Down View)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis("equal")

    # Height over time
    axes[0, 1].plot(positions[:, 0, 1], label="Root")
    axes[0, 1].plot(positions[:, 15, 1], label="Head")  # Joint 15 is head
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Y (m)")
    axes[0, 1].set_title("Height Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Bone lengths over time (should be constant)
    bone_pairs = [(0, 15), (0, 1), (0, 2), (1, 4), (2, 5)]  # pelvis-head, pelvis-l_hip, pelvis-r_hip, etc.
    for idx, (j1, j2) in enumerate(bone_pairs):
        bone_length = np.linalg.norm(positions[:, j2, :] - positions[:, j1, :], axis=-1)
        axes[1, 0].plot(bone_length, label=f"{motion.skeleton.joint_names[j1]}-{motion.skeleton.joint_names[j2]}")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Bone Length (m)")
    axes[1, 0].set_title("Bone Lengths Over Time (Should Be Constant)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 3D skeleton at frame 0
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
    frame_idx = 0
    frame_pos = positions[frame_idx]

    # Plot joints
    ax_3d.scatter(frame_pos[:, 0], frame_pos[:, 2], frame_pos[:, 1], c="b", s=50)

    # Plot bones
    if hasattr(motion.skeleton, "get_bone_connections"):
        connections = motion.skeleton.get_bone_connections()
        for j1, j2 in connections:
            ax_3d.plot(
                [frame_pos[j1, 0], frame_pos[j2, 0]],
                [frame_pos[j1, 2], frame_pos[j2, 2]],
                [frame_pos[j1, 1], frame_pos[j2, 1]],
                "r-",
                linewidth=2,
            )

    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Z (m)")
    ax_3d.set_zlabel("Y (m)")
    ax_3d.set_title(f"3D Skeleton (Frame {frame_idx})")

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                frame_pos[:, 0].max() - frame_pos[:, 0].min(),
                frame_pos[:, 2].max() - frame_pos[:, 2].min(),
                frame_pos[:, 1].max() - frame_pos[:, 1].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (frame_pos[:, 0].max() + frame_pos[:, 0].min()) * 0.5
    mid_z = (frame_pos[:, 2].max() + frame_pos[:, 2].min()) * 0.5
    mid_y = (frame_pos[:, 1].max() + frame_pos[:, 1].min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_z - max_range, mid_z + max_range)
    ax_3d.set_zlim(mid_y - max_range, mid_y + max_range)

    plt.tight_layout()
    plot_path = output_dir / f"{filename}_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved analysis plot: {plot_path}")
    plt.close()

    # 2. Create GLB visualization if available
    if GLB_AVAILABLE and body_model_path and Path(body_model_path).exists():
        logger.info("Creating GLB visualization...")
        try:
            exporter = GLBExporter(model_path=body_model_path, gender="neutral")
            glb_path = output_dir / f"{filename}.glb"
            exporter.export(motion, str(glb_path))
            logger.info(f"Saved GLB: {glb_path}")

            # Create HTML viewer
            visualizer = MotionVisualizer()
            html_path = output_dir / f"{filename}.html"
            visualizer.create_standalone_html(
                glb_path=str(glb_path),
                output_path=str(html_path),
                title=f"HumanML3D Test: {filename}",
                width="100%",
                height="700px",
            )
            logger.info(f"Saved HTML viewer: {html_path}")
        except Exception as e:
            logger.error(f"Failed to create GLB visualization: {e}")
    else:
        logger.warning("GLB visualization not available (missing dependencies or body models)")


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("HumanML3D VISUALIZATION TEST")
    logger.info("=" * 60)

    # Paths
    sample_dir = Path("motion_toolbox/sample_data/humanml3d")
    output_dir = Path("motion_toolbox/outputs/humanml3d_debug")
    body_model_path = Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl"

    # Load sample files
    sample_files = sorted(sample_dir.glob("*.npy"))
    sample_files = [f for f in sample_files if f.name not in ["Mean.npy", "Std.npy"]]

    if not sample_files:
        logger.error(f"No sample files found in {sample_dir}")
        return

    logger.info(f"Found {len(sample_files)} sample files")

    # Test with the first sample
    sample_file = sample_files[0]
    logger.info(f"\nTesting with: {sample_file.name}")

    # Load and denormalize
    mean_path = sample_dir / "Mean.npy"
    std_path = sample_dir / "Std.npy"

    data_263 = load_humanml3d_sample(
        str(sample_file), str(mean_path) if mean_path.exists() else None, str(std_path) if std_path.exists() else None
    )

    # Analyze 263D format
    analyze_263d_format(data_263)

    # Convert to 272D
    data_272 = convert_humanml3d_to_272(data_263)

    # Analyze 272D format
    analyze_272d_format(data_272)

    # Convert to Motion
    motion = convert_272_to_motion(data_272, body_model_path=str(body_model_path) if body_model_path.exists() else None)

    # Visualize
    visualize_motion(
        motion, output_dir, sample_file.stem, body_model_path=str(body_model_path) if body_model_path.exists() else None
    )

    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Check outputs in: {output_dir}")


if __name__ == "__main__":
    main()
