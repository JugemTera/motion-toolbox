#!/usr/bin/env python3
"""Test script for HumanML3D ↔ SMPL bidirectional converter.

This script:
1. Loads HumanML3D data from new_joint_vecs
2. Recovers joint positions using recover_from_ric
3. Fits SMPL parameters using SMPLify3D
4. Creates side-by-side skeleton visualization for comparison
5. Optionally tests round-trip conversion

Usage:
    python scripts/test_humanml3d_smpl_converter.py --sample 000010
"""

import sys
from pathlib import Path

# Add parent to path for imports
MODULE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(MODULE_DIR.parent))

import argparse

import numpy as np


def load_humanml3d_data(sample_id: str, data_dir: str) -> tuple:
    """Load HumanML3D sample data.

    Args:
        sample_id: Sample ID (e.g., "000010")
        data_dir: Path to sample data directory

    Returns:
        Tuple of (data_263d, joints_22, mean, std)
    """
    data_dir = Path(data_dir)

    # Load 263D data
    data_263d_path = data_dir / f"{sample_id}_263d.npy"
    if not data_263d_path.exists():
        data_263d_path = data_dir / f"{sample_id}.npy"

    if not data_263d_path.exists():
        raise FileNotFoundError(f"263D data not found: {data_263d_path}")

    data_263d = np.load(data_263d_path)
    print(f"Loaded 263D data: shape {data_263d.shape}")

    # Load joints if available
    joints_path = data_dir / f"{sample_id}_joints.npy"
    joints_22 = None
    if joints_path.exists():
        joints_22 = np.load(joints_path)
        print(f"Loaded joints: shape {joints_22.shape}")

    # Load mean/std if available
    mean_path = data_dir / "Mean.npy"
    std_path = data_dir / "Std.npy"
    mean = np.load(mean_path) if mean_path.exists() else None
    std = np.load(std_path) if std_path.exists() else None

    if mean is not None:
        print(f"Loaded normalization: mean shape {mean.shape}")

    return data_263d, joints_22, mean, std


def run_recover_from_ric(data_263d: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Run recover_from_ric (HumanML3D RIC recovery).

    Args:
        data_263d: Normalized 263D data
        mean: Mean for denormalization
        std: Std for denormalization

    Returns:
        Recovered joint positions (T, 22, 3)
    """
    from motion_toolbox.converters.humanml3d_to_smpl import recover_from_ric

    # Denormalize
    if mean is not None and std is not None:
        data_263d = data_263d * std + mean

    # Recover positions
    positions = recover_from_ric(data_263d, joints_num=22)

    print(f"Recovered positions: shape {positions.shape}")
    print(f"  Y range: [{positions[:, :, 1].min():.3f}, {positions[:, :, 1].max():.3f}]")
    print(
        f"  Root trajectory range: X [{positions[:, 0, 0].min():.2f}, {positions[:, 0, 0].max():.2f}], "
        f"Z [{positions[:, 0, 2].min():.2f}, {positions[:, 0, 2].max():.2f}]"
    )

    return positions


def create_skeleton_visualization(
    positions: np.ndarray,
    output_path: str,
    title: str = "Skeleton Visualization",
) -> str:
    """Create skeleton visualization.

    Args:
        positions: Joint positions (T, 22, 3)
        output_path: Output HTML path

    Returns:
        Path to generated HTML file
    """
    from motion_toolbox.visualization.skeleton_comparison import create_single_skeleton_html

    return create_single_skeleton_html(
        positions,
        output_path,
        title=title,
        fps=20,
    )


def run_smpl_fitting(
    positions: np.ndarray,
    output_dir: str,
    sample_id: str,
    device: str = "cuda:0",
) -> dict:
    """Test SMPL fitting via SMPLify3D.

    Args:
        positions: Joint positions (T, 22, 3)
        output_dir: Output directory
        sample_id: Sample ID for naming
        device: Device for optimization

    Returns:
        SMPL parameters dict
    """
    from motion_toolbox.converters.humanml3d_to_smpl import HumanML3DToSMPLConverter

    print("\nFitting SMPL parameters with SMPLify3D...")
    print(f"  Device: {device}")
    print(f"  Frames: {len(positions)}")

    converter = HumanML3DToSMPLConverter(
        device=device,
        num_smplify_iters=30,  # Fast for testing (MDM uses 100-150)
        use_batched=False,  # Sequential but more stable
    )

    result = converter.convert_with_positions(positions)

    print("\nSMPL fitting results:")
    print(f"  global_orient: {result['global_orient'].shape}")
    print(f"  body_pose: {result['body_pose'].shape}")
    print(f"  transl: {result['transl'].shape}")
    print(f"  betas: {result['betas'].shape}")

    # Save SMPL params
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{sample_id}_smpl_params.npz",
        global_orient=result["global_orient"],
        body_pose=result["body_pose"],
        transl=result["transl"],
        betas=result["betas"],
    )
    print(f"  Saved to: {output_dir / f'{sample_id}_smpl_params.npz'}")

    return result


def create_comparison_visualization(
    positions_original: np.ndarray,
    smpl_joints: np.ndarray,
    output_path: str,
    sample_id: str,
) -> str:
    """Create side-by-side comparison visualization.

    Args:
        positions_original: Original HumanML3D positions (T, 22, 3)
        smpl_joints: SMPL fitted joints (T, 22, 3)
        output_path: Output HTML path
        sample_id: Sample ID for title

    Returns:
        Path to generated HTML file
    """
    from motion_toolbox.visualization.skeleton_comparison import create_skeleton_comparison_html

    return create_skeleton_comparison_html(
        positions_original,
        smpl_joints,
        output_path,
        title=f"HumanML3D vs SMPL - {sample_id}",
        left_label="Original (recover_from_ric)",
        right_label="SMPL (SMPLify3D)",
        fps=20,
    )


def main():
    parser = argparse.ArgumentParser(description="Test HumanML3D ↔ SMPL converter")
    parser.add_argument("--sample", type=str, default="000010", help="Sample ID to test")
    parser.add_argument("--data-dir", type=str, default="sample_data/humanml3d", help="Path to HumanML3D sample data")
    parser.add_argument("--output-dir", type=str, default="outputs/humanml3d_smpl_test", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for SMPL fitting")
    parser.add_argument("--skip-smpl", action="store_true", help="Skip SMPL fitting (only test recover_from_ric)")
    args = parser.parse_args()

    print("=" * 60)
    print("HumanML3D ↔ SMPL Converter Test")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading sample: {args.sample}")
    data_263d, joints_22, mean, std = load_humanml3d_data(args.sample, args.data_dir)

    # Test recover_from_ric
    print("\n[2] Testing recover_from_ric...")
    positions_recovered = run_recover_from_ric(data_263d, mean, std)

    # Compare with ground truth if available
    if joints_22 is not None:
        # Adjust frame count (263D has T-1 frames)
        joints_22_trimmed = joints_22[: len(positions_recovered)]
        error = np.abs(positions_recovered - joints_22_trimmed).mean()
        max_error = np.abs(positions_recovered - joints_22_trimmed).max()
        print("\n  Comparison with ground truth joints:")
        print(f"    Mean error: {error:.4f}m")
        print(f"    Max error: {max_error:.4f}m")
        print("\n  Note: Will use ground truth joints for SMPL fitting")

    # Create skeleton visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton_html = create_skeleton_visualization(
        positions_recovered,
        str(output_dir / f"{args.sample}_skeleton.html"),
        title=f"HumanML3D Skeleton - {args.sample}",
    )
    print(f"\n[3] Created skeleton visualization: {skeleton_html}")

    if not args.skip_smpl:
        # Test SMPL fitting - use ground truth joints if available
        print("\n[4] Testing SMPL fitting...")
        positions_for_fitting = joints_22 if joints_22 is not None else positions_recovered
        try:
            smpl_result = run_smpl_fitting(
                positions_for_fitting,
                str(output_dir),
                args.sample,
                args.device,
            )

            # Get SMPL joints for comparison
            smpl_joints = smpl_result["joints"][:, :22]

            # Create comparison visualization
            comparison_html = create_comparison_visualization(
                positions_for_fitting,
                smpl_joints,
                str(output_dir / f"{args.sample}_comparison.html"),
                args.sample,
            )
            print(f"\n[5] Created comparison visualization: {comparison_html}")

            # Compute fitting error
            fitting_error = np.abs(smpl_joints - positions_for_fitting).mean()
            max_fitting_error = np.abs(smpl_joints - positions_for_fitting).max()
            print("\n  SMPL fitting error:")
            print(f"    Mean error: {fitting_error:.4f}m ({fitting_error * 100:.2f}cm)")
            print(f"    Max error: {max_fitting_error:.4f}m ({max_fitting_error * 100:.2f}cm)")

        except Exception as e:
            print(f"\n  SMPL fitting failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
