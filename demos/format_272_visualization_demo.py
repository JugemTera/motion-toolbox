"""Demo: 272D Motion Representation Visualization.

This script demonstrates loading and visualizing 272-dimensional motion data
using the existing Motion pipeline and visualization tools.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    print("=" * 80)
    print("272D MOTION REPRESENTATION VISUALIZATION DEMO")
    print("=" * 80)
    
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
    
    # Find 272D sample data
    sample_272_dir = project_root / "sample_data" / "format_272"
    format_272_files = sorted(sample_272_dir.glob("[0-9]*.npy"))
    
    if not format_272_files:
        print(f"\n⚠️  No 272D sample data found in {sample_272_dir}")
        print("   Generating sample data from SMPL...")
        
        # Try to generate from SMPL
        sample_smpl_dir = project_root / "sample_data" / "smpl"
        smpl_files = list(sample_smpl_dir.glob("*.npz"))
        
        if not smpl_files:
            print(f"\n❌ No SMPL data found either in {sample_smpl_dir}")
            print("   Please add SMPL data or download 272D samples")
            print("   Run: python scripts/download_272_samples.py")
            return 1
        
        # Generate 272D from SMPL
        from converters.motion_converters import SMPLTo272Converter
        
        print(f"\n⚙️  Converting SMPL → 272D...")
        converter = SMPLTo272Converter(body_model_path=body_model_path)
        
        smpl_file = smpl_files[0]
        smpl_data = dict(np.load(smpl_file, allow_pickle=True))
        motion_272 = converter.convert(smpl_data)
        
        # Save generated 272D data
        output_file = sample_272_dir / f"generated_{smpl_file.stem}.npy"
        np.save(output_file, motion_272)
        print(f"✅ Generated and saved: {output_file}")
        
        format_272_files = [output_file]
    
    # Select files to visualize (up to 3)
    files_to_visualize = format_272_files[:3]
    
    print(f"\n📂 Found {len(format_272_files)} 272D file(s), visualizing {len(files_to_visualize)}:")
    for f in files_to_visualize:
        data = np.load(f)
        print(f"   • {f.name}: {data.shape[0]} frames, {data.shape[1]} features")
    
    # === STEP 1: Convert 272D to Motion ===
    print("\n" + "=" * 80)
    print("STEP 1: Converting 272D → Motion")
    print("=" * 80)
    
    from converters.motion_converters import Format272ToMotionConverter
    
    converter = Format272ToMotionConverter(
        body_model_path=body_model_path,
        framerate=30.0
    )
    print(f"✅ Created converter: {converter}")
    
    motions = []
    for file_path in files_to_visualize:
        print(f"\n⚙️  Converting {file_path.name}...")
        motion_272 = np.load(file_path)
        motion = converter.convert(motion_272)
        motions.append((file_path.stem, motion))
        print(f"   ✅ Motion: {motion.num_frames} frames, {motion.num_joints} joints, "
              f"{motion.framerate:.1f} fps")
    
    # === STEP 2: Export to GLB ===
    print("\n" + "=" * 80)
    print("STEP 2: Exporting Motions to GLB")
    print("=" * 80)
    
    output_dir = project_root / "outputs" / "format_272_viz"
    glb_dir = output_dir / "glbs"
    glb_dir.mkdir(parents=True, exist_ok=True)
    
    from visualization.glb_exporter import create_glb_from_motion
    
    glb_files = []
    for name, motion in motions:
        print(f"\n⚙️  Exporting {name} to GLB...")
        glb_file = glb_dir / f"{name}.glb"
        
        try:
            create_glb_from_motion(
                motion=motion,
                output_path=str(glb_file),
                body_model_path=body_model_path,
            )
            print(f"   ✅ GLB exported: {glb_file}")
            glb_files.append((name, glb_file))
        except Exception as e:
            print(f"   ❌ GLB export failed: {e}")
    
    # === STEP 3: Create Visualizations ===
    print("\n" + "=" * 80)
    print("STEP 3: Creating HTML Visualizations")
    print("=" * 80)
    
    from visualization.html_viewer import create_motion_visualization
    
    # Create individual viewers
    print("\n📄 Creating individual HTML viewers...")
    for name, glb_file in glb_files:
        html_file = output_dir / f"{name}_viewer.html"
        
        create_motion_visualization(
            glb_path=str(glb_file),
            output_path=str(html_file),
            title=f"272D Motion: {name}",
        )
        print(f"   ✅ {html_file.name}")
    
    # Create grid viewer if multiple files
    if len(glb_files) > 1:
        print("\n📊 Creating spatial grid viewer...")
        
        try:
            from visualization.spatial_animation_grid import create_spatial_animation_grid
            
            grid_html = output_dir / "grid_viewer.html"
            
            # Prepare motion list with paths
            motion_list = []
            for name, motion in motions:
                # We need to pass file paths, not Motion objects
                # So we'll temporarily save as .pkl
                motion_file = glb_dir / f"{name}_motion.pkl"
                import pickle
                with open(motion_file, 'wb') as f:
                    pickle.dump(motion, f)
                motion_list.append(str(motion_file))
            
            create_spatial_animation_grid(
                motion_paths=motion_list,
                output_dir=glb_dir,
                output_html_path=grid_html,
                title="272D Motion Visualization Grid",
                body_model_path=body_model_path,
                spacing=2.5,
            )
            print(f"   ✅ {grid_html.name}")
            print(f"\n🌐 Grid viewer: file://{grid_html}")
            
        except Exception as e:
            print(f"   ⚠️  Grid viewer creation failed: {e}")
    
    # === STEP 4: Motion Analysis ===
    print("\n" + "=" * 80)
    print("STEP 4: Motion Analysis")
    print("=" * 80)
    
    for name, motion in motions:
        print(f"\n📊 {name}:")
        print(f"   Duration: {motion.duration:.2f} seconds")
        print(f"   Frames: {motion.num_frames}")
        print(f"   Joints: {motion.num_joints}")
        print(f"   Frame rate: {motion.framerate:.1f} fps")
        
        # Analyze motion characteristics
        positions = motion.positions
        velocities = motion.velocities
        
        # Root motion
        root_pos = positions[:, 0, :]
        root_displacement = np.linalg.norm(root_pos[-1] - root_pos[0])
        print(f"   Root displacement: {root_displacement:.2f} meters")
        
        # Average velocity
        avg_velocity = np.linalg.norm(velocities, axis=-1).mean()
        print(f"   Avg joint velocity: {avg_velocity:.4f} m/s")
        
        # Height range
        height_min = positions[:, :, 1].min()
        height_max = positions[:, :, 1].max()
        print(f"   Height range: {height_min:.3f} to {height_max:.3f} meters")
    
    # === Summary ===
    print("\n" + "=" * 80)
    print("VISUALIZATION SUMMARY")
    print("=" * 80)
    print(f"\n✅ Successfully visualized {len(motions)} motion(s)")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n🌐 View HTML files:")
    for name, _ in glb_files:
        html_file = output_dir / f"{name}_viewer.html"
        print(f"   • file://{html_file}")
    
    if len(glb_files) > 1:
        grid_html = output_dir / "grid_viewer.html"
        if grid_html.exists():
            print(f"   • file://{grid_html} (Grid View)")
    
    print(f"\n💡 Tips:")
    print(f"   - Use mouse to rotate, zoom, and pan the 3D view")
    print(f"   - Hover over the viewer to see playback controls")
    print(f"   - Compare multiple motions using the grid viewer")
    print(f"   - Add more 272D files to sample_data/format_272/ for visualization")
    
    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
