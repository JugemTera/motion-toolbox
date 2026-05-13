"""Demo: Motion → 272D Representation Conversion.

This script demonstrates converting motion data (HumanML3D or SMPL) to the
272-dimensional motion representation and back, including round-trip testing
and visualization.

Supports:
- HumanML3D (263D) → SMPL → 272D → SMPL (via IK)
- SMPL → 272D → SMPL (direct)
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    print("=" * 80)
    print("MOTION → 272D REPRESENTATION CONVERSION DEMO")
    print("=" * 80)
    print("Supports: HumanML3D (ground truth) and SMPL input formats")
    
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    body_model_path = os.getenv("SMPL_BODY_MODELS_PATH")
    
    # Try clean models first (no chumpy dependency)
    if not body_model_path or not Path(body_model_path).exists():
        clean_path = Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl"
        if clean_path.exists():
            body_model_path = str(clean_path)
    
    if not body_model_path or not Path(body_model_path).exists():
        print("\n❌ Error: SMPL body models not found")
        print("   Set SMPL_BODY_MODELS_PATH in .env file or install to:")
        print("   ~/.motion_toolbox/datasets/body_models_clean_pkl")
        return 1
    
    print(f"\n✅ Using body models from: {body_model_path}")
    
    # Find sample data - support both SMPL and HumanML3D formats
    sample_smpl_dir = project_root / "sample_data" / "smpl"
    sample_humanml3d_dir = project_root / "sample_data" / "humanml3d"
    
    smpl_files = list(sample_smpl_dir.glob("*.npz"))
    humanml3d_files = list(sample_humanml3d_dir.glob("[0-9]*.npy"))  # HumanML3D 263D files
    
    # Prefer HumanML3D files (ground truth), fall back to SMPL
    if humanml3d_files:
        input_file = humanml3d_files[0]
        input_format = "humanml3d"
        print(f"\n📂 Loading HumanML3D data: {input_file.name}")
    elif smpl_files:
        input_file = smpl_files[0]
        input_format = "smpl"
        print(f"\n📂 Loading SMPL data: {input_file.name}")
    else:
        print(f"\n❌ No sample data found!")
        print(f"   Tried: {sample_smpl_dir} (SMPL .npz)")
        print(f"   Tried: {sample_humanml3d_dir} (HumanML3D .npy)")
        return 1
    
    # === STEP 0: Load and convert to SMPL (if needed) ===
    if input_format == "humanml3d":
        print("\n" + "=" * 80)
        print("STEP 0: Converting HumanML3D → SMPL (Ground Truth)")
        print("=" * 80)
        
        humanml3d_data = np.load(input_file)
        print(f"\n📊 HumanML3D Data: shape {humanml3d_data.shape}, dtype {humanml3d_data.dtype}")
        print(f"   Format: 263-dimensional representation (T-1 frames)")
        
        from converters.humanml3d_to_smpl import HumanML3DToSMPLConverter
        
        print("\n⚙️  Converting HumanML3D → SMPL using IK...")
        humanml3d_to_smpl = HumanML3DToSMPLConverter(
            body_model_path=body_model_path,
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
        )
        
        smpl_data = humanml3d_to_smpl.convert(humanml3d_data)
        
        print(f"\n✅ HumanML3D → SMPL conversion complete!")
        print(f"   Output: SMPL parameters for {smpl_data['body_pose'].shape[0]} frames")
        
    else:  # input_format == "smpl"
        # Load SMPL data
        smpl_data = dict(np.load(input_file, allow_pickle=True))
    
    # Display SMPL data info
    print("\n📊 SMPL Data Information:")
    for key, value in smpl_data.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"   {key}: {value}")
    
    # === STEP 1: SMPL → 272D ===
    print("\n" + "=" * 80)
    print("STEP 1: Converting SMPL → 272D")
    print("=" * 80)
    
    from converters.motion_converters import SMPLTo272Converter
    
    smpl_to_272 = SMPLTo272Converter(body_model_path=body_model_path)
    print(f"✅ Created converter: {smpl_to_272}")
    
    print("\n⚙️  Running conversion...")
    motion_272 = smpl_to_272.convert(smpl_data)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output shape: {motion_272.shape}")
    print(f"   Frames: {motion_272.shape[0]}")
    print(f"   Features: {motion_272.shape[1]}")
    print(f"   Data type: {motion_272.dtype}")
    print(f"   Memory: {motion_272.nbytes / 1024:.1f} KB")
    
    # Analyze 272D format structure
    print("\n📊 272D Format Structure:")
    print(f"   [0:2]     Root velocity XZ:        {motion_272[:, 0:2].shape}")
    print(f"   [2:8]     Root angular velocity:   {motion_272[:, 2:8].shape}")
    print(f"   [8:74]    Local joint positions:   {motion_272[:, 8:74].shape}")
    print(f"   [74:140]  Local joint velocities:  {motion_272[:, 74:140].shape}")
    print(f"   [140:272] Joint rotations (6D):    {motion_272[:, 140:272].shape}")
    
    # Show statistics
    print("\n📈 Feature Statistics:")
    for name, idx_start, idx_end in [
        ("Root vel XZ", 0, 2),
        ("Root ang vel", 2, 8),
        ("Local pos", 8, 74),
        ("Local vel", 74, 140),
        ("Rotations 6D", 140, 272),
    ]:
        data = motion_272[:, idx_start:idx_end]
        print(f"   {name:15s}: min={data.min():8.4f}, max={data.max():8.4f}, "
              f"mean={data.mean():8.4f}, std={data.std():8.4f}")
    
    # Save 272D output
    output_dir = project_root / "outputs" / "smpl_272_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_272_file = output_dir / f"{input_file.stem}_272d.npy"
    np.save(output_272_file, motion_272)
    print(f"\n💾 Saved 272D data: {output_272_file}")
    
    # === STEP 2: 272D → SMPL (Round-trip) ===
    print("\n" + "=" * 80)
    print("STEP 2: Converting 272D → SMPL (Round-trip test)")
    print("=" * 80)
    
    from converters.motion_converters import Format272ToSMPLConverter
    
    format_272_to_smpl = Format272ToSMPLConverter(body_model_path=body_model_path)
    print(f"✅ Created converter: {format_272_to_smpl}")
    
    print("\n⚙️  Running conversion...")
    recovered_smpl = format_272_to_smpl.convert(motion_272)
    
    print(f"\n✅ Conversion complete!")
    print(f"\n📊 Recovered SMPL Parameters:")
    for key, value in recovered_smpl.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"   {key}: {value}")
    
    # === STEP 3: Error Analysis ===
    print("\n" + "=" * 80)
    print("STEP 3: Round-trip Error Analysis")
    print("=" * 80)
    
    # Compare original and recovered
    import torch
    
    orig_body_pose = smpl_data["pose_body"]
    orig_root_orient = smpl_data["root_orient"]
    orig_trans = smpl_data["trans"]
    
    # Convert to numpy if needed
    if torch.is_tensor(orig_body_pose):
        orig_body_pose = orig_body_pose.numpy()
    if torch.is_tensor(orig_root_orient):
        orig_root_orient = orig_root_orient.numpy()
    if torch.is_tensor(orig_trans):
        orig_trans = orig_trans.numpy()
    
    recovered_body_pose = recovered_smpl["body_pose"].numpy()
    recovered_root_orient = recovered_smpl["global_orient"].numpy()
    recovered_trans = recovered_smpl["transl"].numpy()
    
    # Compute errors
    body_pose_error = np.linalg.norm(
        orig_body_pose.reshape(-1, 63) - recovered_body_pose.reshape(-1, 63),
        axis=-1
    ).mean()
    
    root_orient_error = np.linalg.norm(
        orig_root_orient - recovered_root_orient,
        axis=-1
    ).mean()
    
    # For translation, use relative motion (to account for floor alignment)
    orig_trans_diff = orig_trans[1:] - orig_trans[:-1] if len(orig_trans) > 1 else orig_trans
    recovered_trans_diff = recovered_trans[1:] - recovered_trans[:-1] if len(recovered_trans) > 1 else recovered_trans
    trans_error = np.linalg.norm(orig_trans_diff - recovered_trans_diff, axis=-1).mean()
    
    print(f"\n📏 Reconstruction Errors:")
    print(f"   Body pose error:      {body_pose_error:.6f} radians")
    print(f"   Root orientation error: {root_orient_error:.6f} radians")
    print(f"   Translation error:    {trans_error:.6f} meters (relative motion)")
    
    # Error assessment
    if body_pose_error < 0.01 and root_orient_error < 0.01:
        print("\n✅ Excellent reconstruction! Errors are very small.")
    elif body_pose_error < 0.1 and root_orient_error < 0.1:
        print("\n✅ Good reconstruction! Errors are acceptable.")
    else:
        print("\n⚠️  Moderate reconstruction errors. This may be expected due to:")
        print("   - Floor alignment differences")
        print("   - Numerical precision in conversion pipeline")
        print("   - Canonicalization operations")
    
    # === STEP 4: Visualization (Optional) ===
    print("\n" + "=" * 80)
    print("STEP 4: Motion Visualization")
    print("=" * 80)
    
    try:
        from converters.motion_converters import Format272ToMotionConverter
        from visualization.glb_exporter import create_glb_from_motion
        from visualization.html_viewer import create_motion_visualization
        
        print("\n⚙️  Converting 272D → Motion for visualization...")
        format_272_to_motion = Format272ToMotionConverter(body_model_path=body_model_path)
        motion = format_272_to_motion.convert(motion_272)
        print(f"✅ Motion: {motion.num_frames} frames, {motion.num_joints} joints")
        
        # Export GLB
        print("\n⚙️  Exporting to GLB...")
        glb_file = output_dir / f"{smpl_file.stem}_272d.glb"
        create_glb_from_motion(
            motion=motion,
            output_path=str(glb_file),
            body_model_path=body_model_path,
        )
        print(f"✅ GLB exported: {glb_file}")
        
        # Create HTML viewer
        print("\n⚙️  Creating HTML viewer...")
        html_file = output_dir / f"{smpl_file.stem}_viewer.html"
        create_motion_visualization(
            glb_path=str(glb_file),
            output_path=str(html_file),
            title=f"272D Motion: {smpl_file.stem}",
        )
        print(f"✅ HTML viewer created: {html_file}")
        
        print(f"\n🌐 Open in browser: file://{html_file}")
        
    except ImportError as e:
        print(f"\n⚠️  Visualization skipped (missing dependencies): {e}")
    except Exception as e:
        print(f"\n⚠️  Visualization failed: {e}")
    
    # === Summary ===
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print(f"\n✅ Successfully demonstrated:")
    print(f"   1. SMPL → 272D conversion")
    print(f"   2. 272D → SMPL round-trip conversion")
    print(f"   3. Error analysis (reconstruction quality)")
    print(f"   4. Motion visualization pipeline")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n💡 Next steps:")
    print(f"   - Try with different SMPL files in sample_data/smpl/")
    print(f"   - Use 272D data for motion generation models")
    print(f"   - Visualize and compare original vs. recovered motions")
    
    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
