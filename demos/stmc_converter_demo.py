"""Demo script for STMC converter validation.

This script demonstrates the bi-directional conversion between Motion (SMPL) and STMC 205D representation.
It validates the reconstruction quality by:
1. Converting a Motion to STMC features (forward)
2. Converting STMC features back to Motion (inverse)
3. Computing MSE between original and reconstructed root translations
4. Visualizing both motions side-by-side
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add motion_toolbox to path for imports
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from motion_toolbox.converters.stmc_converter import STMC205Converter
from motion_toolbox.core.identity import Identity
from motion_toolbox.core.motion import Motion
from motion_toolbox.core.skeleton import Skeleton
from motion_toolbox.visualization.html_viewer import create_motion_visualization


def load_sample_motion(sample_path: str, body_model_path: str) -> Motion:
    """Load a sample motion from file.
    
    Args:
        sample_path: Path to sample motion file (.pt or .npz)
        body_model_path: Path to SMPL body models
    
    Returns:
        Motion object
    """
    if sample_path.endswith('.pt'):
        data = torch.load(sample_path, weights_only=False)
        
        # Handle different data formats
        if isinstance(data, dict):
            # Check if it's GVHMR format
            if 'smpl_params_global' in data:
                smpl_data = data['smpl_params_global']
                global_orient = smpl_data['global_orient']
                body_pose = smpl_data['body_pose']
                transl = smpl_data['transl']
            else:
                # SMPL format
                global_orient = data['global_orient']  # (T, 3)
                body_pose = data['body_pose']  # (T, 69) or (T, 63)
                transl = data['transl']  # (T, 3)
            
            # Concatenate to get full pose
            if isinstance(global_orient, torch.Tensor):
                global_orient = global_orient.cpu().numpy()
            if isinstance(body_pose, torch.Tensor):
                body_pose = body_pose.cpu().numpy()
            if isinstance(transl, torch.Tensor):
                transl = transl.cpu().numpy()
            
            # Combine into joint rotations
            num_frames = global_orient.shape[0]
            num_body_joints = body_pose.shape[1] // 3
            
            # Create joint rotations with proper shape for SMPL (24 joints)
            skeleton = Skeleton.from_smpl()
            joint_rotations = np.zeros((num_frames, skeleton.num_joints, 3))
            joint_rotations[:, 0, :] = global_orient
            
            # Fill in body joints (handle cases where we have fewer than 23 body joints)
            available_body_joints = min(num_body_joints, skeleton.num_joints - 1)
            joint_rotations[:, 1:available_body_joints + 1, :] = body_pose.reshape(num_frames, num_body_joints, 3)[:, :available_body_joints, :]
            
            # Get identity if available
            identity = None
            # Get identity if available
            identity = None
            if 'betas' in data and data['betas'] is not None:
                betas = data['betas']
                if isinstance(betas, torch.Tensor):
                    betas = betas.cpu().numpy()
                if betas.ndim == 2:
                    betas = betas[0]
                identity = Identity(betas=betas)
            
            motion = Motion(
                joint_rotations=joint_rotations,
                root_translation=transl,
                skeleton=skeleton,
                framerate=30.0,
                identity=identity,
                body_model_path=body_model_path,
                floor_alignment="mesh",
            )
            
    elif sample_path.endswith('.npz'):
        data = np.load(sample_path)
        
        # Handle npz format
        if 'poses' in data:
            poses = data['poses']  # (T, 72) or (T, 69)
            transl = data['trans'] if 'trans' in data else data.get('transl', np.zeros((poses.shape[0], 3)))
            
            num_frames = poses.shape[0]
            num_joints = poses.shape[1] // 3
            joint_rotations = poses.reshape(num_frames, num_joints, 3)
            
            identity = None
            if 'betas' in data:
                identity = Identity(betas=data['betas'])
            
            skeleton = Skeleton.from_smpl()
            
            motion = Motion(
                joint_rotations=joint_rotations[:, :skeleton.num_joints, :],
                root_translation=transl,
                skeleton=skeleton,
                framerate=30.0,
                identity=identity,
                body_model_path=body_model_path,
                floor_alignment="mesh",
            )
    else:
        raise ValueError(f"Unsupported file format: {sample_path}")
    
    return motion


def compute_metrics(original_motion: Motion, reconstructed_motion: Motion):
    """Compute reconstruction metrics.
    
    Args:
        original_motion: Original motion
        reconstructed_motion: Reconstructed motion
    
    Returns:
        Dictionary of metrics
    """
    # Root translation MSE
    root_mse = np.mean((original_motion.root_translation - reconstructed_motion.root_translation) ** 2)
    root_rmse = np.sqrt(root_mse)
    
    # Joint rotation MSE (axis-angle)
    rotation_mse = np.mean((original_motion.joint_rotations - reconstructed_motion.joint_rotations) ** 2)
    rotation_rmse = np.sqrt(rotation_mse)
    
    # Position MSE (from forward kinematics)
    pos_mse = np.mean((original_motion.positions - reconstructed_motion.positions) ** 2)
    pos_rmse = np.sqrt(pos_mse)
    
    return {
        'root_translation_mse': root_mse,
        'root_translation_rmse': root_rmse,
        'joint_rotation_mse': rotation_mse,
        'joint_rotation_rmse': rotation_rmse,
        'position_mse': pos_mse,
        'position_rmse': pos_rmse,
    }


def main():
    parser = argparse.ArgumentParser(description='STMC Converter Demo')
    parser.add_argument(
        '--input',
        type=str,
        default='sample_data/smpl/gvhmr_gt_4d004.pt',
        help='Input motion file (.pt or .npz)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--body-model-path',
        type=str,
        default=None,
        help='Path to SMPL body models (default: ~/.motion_toolbox/datasets/body_models_clean_pkl/)'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=None,
        help='Number of frames to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Set default body model path
    if args.body_model_path is None:
        args.body_model_path = str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl/")
    
    print("=" * 80)
    print("STMC 205D Converter Validation")
    print("=" * 80)
    
    # Load sample motion
    print(f"\n1. Loading motion from: {args.input}")
    motion_original = load_sample_motion(args.input, args.body_model_path)
    
    # Slice if requested
    if args.num_frames is not None and args.num_frames < motion_original.num_frames:
        motion_original = motion_original.slice_frames(0, args.num_frames)
    
    print(f"   - Frames: {motion_original.num_frames}")
    print(f"   - Joints: {motion_original.num_joints}")
    print(f"   - Duration: {motion_original.duration:.2f}s")
    print(f"   - FPS: {motion_original.framerate}")
    
    # Initialize converter
    print("\n2. Initializing STMC converter")
    converter = STMC205Converter(
        smpl_model_path=args.body_model_path,
        device='cpu',
    )
    
    # Forward conversion (Motion -> STMC features)
    print("\n3. Converting Motion to STMC 205D features (forward)")
    features = converter.forward(motion_original)
    print(f"   - Features shape: {features.shape}")
    print(f"   - Expected: ({motion_original.num_frames}, 205)")
    
    # Verify feature dimensions
    assert features.shape == (motion_original.num_frames, 205), "Feature shape mismatch!"
    
    # Print feature statistics
    print("\n4. Feature statistics:")
    feature_ranges = {
        'Root Linear Vel (XZ)': features[:, 0:2],
        'Root Angular Vel (Y)': features[:, 2:3],
        'Foot Contacts': features[:, 3:7],
        'Local Positions': features[:, 7:73],
        'Rotations 6D': features[:, 73:205],
    }
    
    for name, feat in feature_ranges.items():
        print(f"   {name:25s}: min={feat.min().item():.4f}, max={feat.max().item():.4f}, mean={feat.mean().item():.4f}")
    
    # Inverse conversion (STMC features -> Motion)
    print("\n5. Converting STMC features back to Motion (inverse)")
    motion_reconstructed = converter.inverse(features, skeleton=motion_original.skeleton)
    print(f"   - Frames: {motion_reconstructed.num_frames}")
    print(f"   - Joints: {motion_reconstructed.num_joints}")
    
    # Compute metrics
    print("\n6. Computing reconstruction metrics:")
    metrics = compute_metrics(motion_original, motion_reconstructed)
    
    print("   Root Translation:")
    print(f"     - MSE:  {metrics['root_translation_mse']:.6f} m²")
    print(f"     - RMSE: {metrics['root_translation_rmse']:.6f} m")
    print("   Joint Rotations:")
    print(f"     - MSE:  {metrics['joint_rotation_mse']:.6f} rad²")
    print(f"     - RMSE: {metrics['joint_rotation_rmse']:.6f} rad")
    print("   Joint Positions (FK):")
    print(f"     - MSE:  {metrics['position_mse']:.6f} m²")
    print(f"     - RMSE: {metrics['position_rmse']:.6f} m")
    
    # Critical validation check
    print("\n7. Validation status:")
    if metrics['root_translation_rmse'] < 0.05:  # Less than 5cm error is acceptable
        print(f"   ✓ PASS: Root translation error is acceptable ({metrics['root_translation_rmse']*100:.2f} cm)")
    else:
        print(f"   ⚠ WARNING: Root translation error is high ({metrics['root_translation_rmse']*100:.2f} cm)")
    
    if metrics['joint_rotation_rmse'] < 0.01:  # Less than 0.01 radians (~0.6 degrees)
        print(f"   ✓ PASS: Joint rotation error is excellent ({np.degrees(metrics['joint_rotation_rmse']):.2f}°)")
    else:
        print(f"   ⚠ WARNING: Joint rotation error is high ({np.degrees(metrics['joint_rotation_rmse']):.2f}°)")
    
    # Visualize both motions
    print("\n8. Generating visualizations:")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export motions to GLB first
    from motion_toolbox.visualization.glb_exporter import GLBExporter
    
    exporter = GLBExporter(model_path=args.body_model_path)
    
    # Export original
    original_glb = output_dir / "stmc_original.glb"
    print(f"   - Exporting original motion to: {original_glb}")
    exporter.export(motion_original, str(original_glb))
    
    # Export reconstructed
    reconstructed_glb = output_dir / "stmc_reconstructed.glb"
    print(f"   - Exporting reconstructed motion to: {reconstructed_glb}")
    exporter.export(motion_reconstructed, str(reconstructed_glb))
    
    # Create HTML viewers
    original_output = output_dir / "stmc_original.html"
    print(f"   - Creating HTML viewer: {original_output}")
    create_motion_visualization(str(original_glb), str(original_output), title="Original Motion")
    
    reconstructed_output = output_dir / "stmc_reconstructed.html"
    print(f"   - Creating HTML viewer: {reconstructed_output}")
    create_motion_visualization(str(reconstructed_glb), str(reconstructed_output), title="STMC Reconstructed")
    
    # Create side-by-side comparison
    comparison_output = output_dir / "stmc_comparison.html"
    print(f"   - Comparison view: {comparison_output}")
    
    # Simple side-by-side HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>STMC Converter Comparison</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: white;
        }}
        h1 {{
            text-align: center;
        }}
        .container {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }}
        .viewer {{
            flex: 1;
            border: 2px solid #333;
            border-radius: 8px;
            overflow: hidden;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
        }}
        .label {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
            background: #2a2a2a;
            border-bottom: 2px solid #444;
        }}
        .metrics {{
            margin: 20px auto;
            max-width: 600px;
            padding: 20px;
            background: #2a2a2a;
            border-radius: 8px;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #444;
        }}
    </style>
</head>
<body>
    <h1>STMC Converter: Original vs Reconstructed</h1>
    
    <div class="metrics">
        <h2>Reconstruction Metrics</h2>
        <div class="metric-row">
            <span>Root Translation RMSE:</span>
            <span>{metrics['root_translation_rmse']:.6f} m ({metrics['root_translation_rmse']*100:.2f} cm)</span>
        </div>
        <div class="metric-row">
            <span>Joint Rotation RMSE:</span>
            <span>{metrics['joint_rotation_rmse']:.6f} rad ({np.degrees(metrics['joint_rotation_rmse']):.2f}°)</span>
        </div>
        <div class="metric-row">
            <span>Position RMSE:</span>
            <span>{metrics['position_rmse']:.6f} m ({metrics['position_rmse']*100:.2f} cm)</span>
        </div>
    </div>
    
    <div class="container">
        <div class="viewer">
            <div class="label">Original Motion</div>
            <iframe src="stmc_original.html"></iframe>
        </div>
        <div class="viewer">
            <div class="label">STMC Reconstructed (205D)</div>
            <iframe src="stmc_reconstructed.html"></iframe>
        </div>
    </div>
</body>
</html>
"""
    
    with open(comparison_output, 'w') as f:
        f.write(html_content)
    
    print("\n" + "=" * 80)
    print(f"Done! Open {comparison_output} in your browser to view the comparison.")
    print("=" * 80)


if __name__ == '__main__':
    main()
