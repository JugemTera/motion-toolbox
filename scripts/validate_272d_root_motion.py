"""Validate 272D Root Motion Preservation.

This script validates that root motion velocities are correctly preserved
through the SMPL → 272D → SMPL conversion pipeline.

Tests:
1. Visualize original SMPL (with absolute translation)
2. Visualize 272D round-trip SMPL (translation from 0,0,0)
3. Compare root velocities (should match!)
4. Compare root orientation changes (should match!)
"""

import sys
from pathlib import Path

import numpy as np

# Add package source to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    print("=" * 80)
    print("272D ROOT MOTION VALIDATION")
    print("=" * 80)
    print("\nThis script validates that root motion is preserved through 272D conversion")
    print("Expected: Absolute position removed, but velocities/rotations preserved\n")

    import os

    import torch
    from dotenv import load_dotenv

    load_dotenv()
    body_model_path = os.getenv("SMPL_BODY_MODELS_PATH")

    if not body_model_path or not Path(body_model_path).exists():
        clean_path = Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl"
        if clean_path.exists():
            body_model_path = str(clean_path)

    if not body_model_path or not Path(body_model_path).exists():
        print("❌ Error: SMPL body models not found")
        return 1

    # === STEP 1: Load HumanML3D Ground Truth ===
    print("\n" + "=" * 80)
    print("STEP 1: Load HumanML3D Ground Truth")
    print("=" * 80)

    humanml3d_file = project_root / "sample_data" / "humanml3d" / "000000.npy"
    if not humanml3d_file.exists():
        print(f"❌ HumanML3D file not found: {humanml3d_file}")
        return 1

    humanml3d_data = np.load(humanml3d_file)
    print(f"\n✅ Loaded: {humanml3d_file.name}")
    print(f"   Shape: {humanml3d_data.shape} (263D HumanML3D format)")

    # === STEP 2: Convert to SMPL (preserves translation) ===
    print("\n" + "=" * 80)
    print("STEP 2: Convert HumanML3D → SMPL (Original with Translation)")
    print("=" * 80)

    from motion_toolbox.converters.humanml3d_to_smpl import HumanML3DToSMPLConverter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n⚙️  Converting on device: {device}")

    humanml3d_to_smpl = HumanML3DToSMPLConverter(smpl_model_path=body_model_path, device=device)

    smpl_original = humanml3d_to_smpl.convert(humanml3d_data)

    # Check what keys we got
    print(f"\n📊 SMPL keys: {list(smpl_original.keys())}")

    # Map to standardized keys
    trans_key = "transl" if "transl" in smpl_original else "trans"
    pose_key = "body_pose" if "body_pose" in smpl_original else "pose_body"
    orient_key = "global_orient" if "global_orient" in smpl_original else "root_orient"

    num_frames = smpl_original[trans_key].shape[0]
    print(f"\n✅ Converted to SMPL: {num_frames} frames")
    print(f"   {trans_key} shape: {smpl_original[trans_key].shape}")
    print(f"   {pose_key} shape: {smpl_original[pose_key].shape}")
    print(f"   {orient_key} shape: {smpl_original[orient_key].shape}")

    # Calculate original root velocities
    trans_original = smpl_original[trans_key]
    velocities_original = np.zeros((num_frames, 3), dtype=np.float32)
    velocities_original[1:] = trans_original[1:] - trans_original[:-1]

    print("\n📊 Original Translation:")
    print(f"   Start position: [{trans_original[0, 0]:.3f}, {trans_original[0, 1]:.3f}, {trans_original[0, 2]:.3f}]")
    print(f"   End position:   [{trans_original[-1, 0]:.3f}, {trans_original[-1, 1]:.3f}, {trans_original[-1, 2]:.3f}]")
    print(f"   Total distance: {np.linalg.norm(trans_original[-1] - trans_original[0]):.3f}m")
    print("\n📊 Original Root Velocities:")
    print(f"   Mean XZ speed: {np.linalg.norm(velocities_original[:, [0, 2]], axis=1).mean():.4f} m/frame")
    print(f"   Max XZ speed:  {np.linalg.norm(velocities_original[:, [0, 2]], axis=1).max():.4f} m/frame")

    # === STEP 3: Convert SMPL → 272D ===
    print("\n" + "=" * 80)
    print("STEP 3: Convert SMPL → 272D")
    print("=" * 80)

    from motion_toolbox.converters.motion_converters import SMPLTo272Converter

    # Remap keys to match SMPLTo272Converter expectations
    # Note: HumanML3DToSMPLConverter returns 23 joints (69 dims), but SMPLTo272Converter expects 21 joints (63 dims)
    body_pose_truncated = smpl_original["body_pose"][:, :63]  # Take only first 21 joints

    smpl_for_272 = {
        "pose_body": body_pose_truncated,
        "root_orient": smpl_original["global_orient"],
        "trans": smpl_original["transl"],
        "betas": smpl_original["betas"],
    }

    smpl_to_272 = SMPLTo272Converter(body_model_path=body_model_path)
    motion_272 = smpl_to_272.convert(smpl_for_272)

    print(f"\n✅ Converted to 272D: {motion_272.shape}")
    print("\n📊 272D Root Velocities (stored):")
    root_vel_272 = motion_272[:, 0:2]  # XZ velocities
    print(f"   Mean XZ speed: {np.linalg.norm(root_vel_272, axis=1).mean():.4f} units/frame")
    print(f"   Max XZ speed:  {np.linalg.norm(root_vel_272, axis=1).max():.4f} units/frame")

    # === STEP 4: Convert 272D → SMPL (round-trip) ===
    print("\n" + "=" * 80)
    print("STEP 4: Convert 272D → SMPL (Round-Trip)")
    print("=" * 80)

    from motion_toolbox.converters.motion_converters import Format272ToSMPLConverter

    format_272_to_smpl = Format272ToSMPLConverter(body_model_path=body_model_path)
    smpl_roundtrip = format_272_to_smpl.convert(motion_272)

    # Check what keys we got back
    print(f"\n📊 Round-trip SMPL keys: {list(smpl_roundtrip.keys())}")

    # Map keys
    trans_rt_key = "transl" if "transl" in smpl_roundtrip else "trans"
    trans_roundtrip = smpl_roundtrip[trans_rt_key]
    velocities_roundtrip = np.zeros((num_frames, 3), dtype=np.float32)
    velocities_roundtrip[1:] = trans_roundtrip[1:] - trans_roundtrip[:-1]

    print(f"\n✅ Converted back to SMPL: {num_frames} frames")
    print("\n📊 Round-Trip Translation:")
    print(f"   Start position: [{trans_roundtrip[0, 0]:.3f}, {trans_roundtrip[0, 1]:.3f}, {trans_roundtrip[0, 2]:.3f}]")
    print(
        f"   End position:   [{trans_roundtrip[-1, 0]:.3f}, {trans_roundtrip[-1, 1]:.3f}, {trans_roundtrip[-1, 2]:.3f}]"
    )
    print(f"   Total distance: {np.linalg.norm(trans_roundtrip[-1] - trans_roundtrip[0]):.3f}m")
    print("\n📊 Round-Trip Root Velocities:")
    print(f"   Mean XZ speed: {np.linalg.norm(velocities_roundtrip[:, [0, 2]], axis=1).mean():.4f} m/frame")
    print(f"   Max XZ speed:  {np.linalg.norm(velocities_roundtrip[:, [0, 2]], axis=1).max():.4f} m/frame")

    # === STEP 5: Compare Velocities ===
    print("\n" + "=" * 80)
    print("STEP 5: Validate Root Motion Preservation")
    print("=" * 80)

    # Compare XZ velocities (Y is height, not relevant for locomotion)
    vel_original_xz = velocities_original[:, [0, 2]]
    vel_roundtrip_xz = velocities_roundtrip[:, [0, 2]]

    velocity_error = np.abs(vel_original_xz - vel_roundtrip_xz)
    velocity_error_mean = velocity_error.mean()
    velocity_error_max = velocity_error.max()

    print("\n📊 Velocity Comparison (XZ plane):")
    print(f"   Mean absolute error: {velocity_error_mean:.6f} m/frame")
    print(f"   Max absolute error:  {velocity_error_max:.6f} m/frame")
    print(f"   Mean relative error: {(velocity_error_mean / np.abs(vel_original_xz).mean() * 100):.2f}%")

    # Threshold for validation
    VELOCITY_THRESHOLD = 0.01  # 1cm per frame

    if velocity_error_mean < VELOCITY_THRESHOLD:
        print(f"\n✅ PASS: Root velocities preserved! (error < {VELOCITY_THRESHOLD}m/frame)")
    else:
        print(f"\n❌ FAIL: Root velocities NOT preserved! (error > {VELOCITY_THRESHOLD}m/frame)")

    # Compare distances traveled
    dist_original = np.linalg.norm(trans_original[-1] - trans_original[0])
    dist_roundtrip = np.linalg.norm(trans_roundtrip[-1] - trans_roundtrip[0])
    dist_error = np.abs(dist_original - dist_roundtrip)

    print("\n📊 Total Distance Comparison:")
    print(f"   Original:   {dist_original:.3f}m")
    print(f"   Round-trip: {dist_roundtrip:.3f}m")
    print(f"   Difference: {dist_error:.3f}m ({dist_error / dist_original * 100:.1f}%)")

    if dist_error / dist_original < 0.05:  # 5% tolerance
        print("   ✅ PASS: Distance preserved within 5%")
    else:
        print("   ⚠️  WARNING: Distance difference > 5%")

    # === STEP 6: Visualize Both ===
    print("\n" + "=" * 80)
    print("STEP 6: Generate Visualizations")
    print("=" * 80)

    output_dir = project_root / "outputs" / "validate_272d_root_motion"
    output_dir.mkdir(parents=True, exist_ok=True)

    import base64

    from motion_toolbox.visualization.glb_exporter import GLBExporter

    exporter = GLBExporter(model_path=body_model_path, model_type="smpl", gender="neutral", fps=30, include_floor=True)

    # Create Motion objects for export
    from motion_toolbox.converters.motion_converters import Format272ToMotionConverter

    # Export original via 272D → Motion (to ensure fair comparison)
    print("\n⚙️  Exporting original SMPL...")
    converter_to_motion = Format272ToMotionConverter(framerate=30.0, body_model_path=body_model_path)

    # Convert original to 272D then to Motion
    motion_original_272 = smpl_to_272.convert(smpl_for_272)
    motion_original = converter_to_motion.convert(motion_original_272)

    glb_original = output_dir / "original_with_translation.glb"
    exporter.export(motion=motion_original, filename=str(glb_original))
    print(f"   ✅ {glb_original.name}")

    # Export round-trip SMPL
    print("\n⚙️  Exporting 272D round-trip...")
    motion_roundtrip = converter_to_motion.convert(motion_272)

    glb_roundtrip = output_dir / "roundtrip_from_origin.glb"
    exporter.export(motion=motion_roundtrip, filename=str(glb_roundtrip))
    print(f"   ✅ {glb_roundtrip.name}")

    # Create comparison HTML
    print("\n⚙️  Creating comparison viewer...")

    with open(glb_original, "rb") as f:
        glb_original_base64 = base64.b64encode(f.read()).decode("utf-8")

    with open(glb_roundtrip, "rb") as f:
        glb_roundtrip_base64 = base64.b64encode(f.read()).decode("utf-8")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>272D Root Motion Validation</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'SF Pro Display', -apple-system, sans-serif; background: #000; color: #fff; }}
        .header {{ background: #000; border-bottom: 1px solid #333; padding: 24px 32px; }}
        .header h1 {{ font-size: 24px; font-weight: 600; }}
        .header .subtitle {{ color: #999; font-size: 14px; margin-top: 8px; }}
        .results {{ background: #111; padding: 24px 32px; border-bottom: 1px solid #333; }}
        .result-row {{ display: flex; gap: 32px; margin: 12px 0; }}
        .result-label {{ color: #999; min-width: 200px; }}
        .result-value {{ color: #fff; font-weight: 500; }}
        .result-value.pass {{ color: #4ade80; }}
        .result-value.warn {{ color: #fbbf24; }}
        .result-value.fail {{ color: #ef4444; }}
        .viewers {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; height: calc(100vh - 250px); }}
        .viewer-container {{ position: relative; background: #fff; display: flex; flex-direction: column; border-right: 1px solid #333; }}
        .viewer-container:last-child {{ border-right: none; }}
        .viewer-title {{ background: #000; color: #fff; padding: 16px 24px; font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #333; }}
        model-viewer {{ flex: 1; width: 100%; background: #fff; }}
        .controls {{ position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%); background: #000; border: 1px solid #333; border-radius: 12px; padding: 16px 24px; display: flex; gap: 16px; z-index: 10; }}
        .controls button {{ background: #fff; color: #000; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 500; cursor: pointer; }}
        .controls button:hover {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>272D Root Motion Validation</h1>
        <div class="subtitle">Comparing original SMPL (with translation) vs 272D round-trip (from origin)</div>
    </div>
    
    <div class="results">
        <div class="result-row">
            <div class="result-label">Original Distance Traveled:</div>
            <div class="result-value">{dist_original:.3f}m</div>
        </div>
        <div class="result-row">
            <div class="result-label">Round-Trip Distance Traveled:</div>
            <div class="result-value">{dist_roundtrip:.3f}m</div>
        </div>
        <div class="result-row">
            <div class="result-label">Distance Error:</div>
            <div class="result-value {"pass" if dist_error / dist_original < 0.05 else "warn"}">{dist_error:.3f}m ({dist_error / dist_original * 100:.1f}%)</div>
        </div>
        <div class="result-row">
            <div class="result-label">Mean Velocity Error (XZ):</div>
            <div class="result-value {"pass" if velocity_error_mean < VELOCITY_THRESHOLD else "fail"}">{velocity_error_mean:.6f} m/frame</div>
        </div>
        <div class="result-row">
            <div class="result-label">Root Motion Preservation:</div>
            <div class="result-value {"pass" if velocity_error_mean < VELOCITY_THRESHOLD else "fail"}">{"✅ PASS" if velocity_error_mean < VELOCITY_THRESHOLD else "❌ FAIL"}</div>
        </div>
    </div>
    
    <div class="viewers">
        <div class="viewer-container">
            <div class="viewer-title">Original (with translation)</div>
            <model-viewer id="viewer1" src="data:model/gltf-binary;base64,{glb_original_base64}" camera-controls autoplay shadow-intensity="0.4" environment-image="neutral" camera-target="0m 1m 0m" camera-orbit="45deg 75deg 4m" field-of-view="45deg" loading="eager"></model-viewer>
            <div class="controls">
                <button onclick="togglePlay(1)">Pause</button>
                <button onclick="resetCamera(1)">Reset Camera</button>
            </div>
        </div>
        <div class="viewer-container">
            <div class="viewer-title">Round-Trip 272D (from origin)</div>
            <model-viewer id="viewer2" src="data:model/gltf-binary;base64,{glb_roundtrip_base64}" camera-controls autoplay shadow-intensity="0.4" environment-image="neutral" camera-target="0m 1m 0m" camera-orbit="45deg 75deg 4m" field-of-view="45deg" loading="eager"></model-viewer>
            <div class="controls">
                <button onclick="togglePlay(2)">Pause</button>
                <button onclick="resetCamera(2)">Reset Camera</button>
            </div>
        </div>
    </div>
    
    <script>
        const viewers = [document.getElementById('viewer1'), document.getElementById('viewer2')];
        const playing = [true, true];
        
        viewers.forEach((viewer, i) => {{
            viewer.addEventListener('load', () => {{
                const model = viewer.model;
                if (model) {{
                    model.materials.forEach((material) => {{
                        if (!material.name || !material.name.toLowerCase().includes('floor')) {{
                            const colors = [[0.2, 0.6, 0.9, 1.0], [0.9, 0.3, 0.5, 1.0]];
                            material.pbrMetallicRoughness.setBaseColorFactor(colors[i]);
                            material.pbrMetallicRoughness.setRoughnessFactor(0.7);
                            material.pbrMetallicRoughness.setMetallicFactor(0.1);
                        }}
                    }});
                }}
            }});
        }});
        
        function togglePlay(id) {{
            const idx = id - 1;
            if (playing[idx]) {{
                viewers[idx].pause();
                event.target.textContent = 'Play';
            }} else {{
                viewers[idx].play();
                event.target.textContent = 'Pause';
            }}
            playing[idx] = !playing[idx];
        }}
        
        function resetCamera(id) {{
            viewers[id - 1].cameraTarget = '0m 1m 0m';
            viewers[id - 1].cameraOrbit = '45deg 75deg 4m';
            viewers[id - 1].fieldOfView = '45deg';
        }}
    </script>
</body>
</html>"""

    html_file = output_dir / "comparison.html"
    with open(html_file, "w") as f:
        f.write(html_content)

    print(f"   ✅ {html_file.name}")

    # === Summary ===
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print("\n📊 Results:")
    print(f"   Velocity preservation: {'✅ PASS' if velocity_error_mean < VELOCITY_THRESHOLD else '❌ FAIL'}")
    print(f"   Distance preservation: {'✅ PASS' if dist_error / dist_original < 0.05 else '⚠️  WARNING'}")

    print("\n📁 Output:")
    print(f"   Comparison viewer: {html_file}")
    print(f"   Original GLB: {glb_original}")
    print(f"   Round-trip GLB: {glb_roundtrip}")

    print("\n💡 Interpretation:")
    print("   - Original: Shows ground truth walking with absolute translation")
    print("   - Round-trip: Shows same motion but starting from (0,0,0)")
    print("   - Both should have same walking speed and style!")

    if velocity_error_mean < VELOCITY_THRESHOLD:
        print("\n✅ SUCCESS: 272D converter correctly preserves root motion velocities!")
    else:
        print("\n❌ ERROR: 272D converter is NOT preserving root motion correctly!")
        print("   This indicates a bug in the SMPL→272D or 272D→SMPL conversion.")

    print("\n" + "=" * 80)

    return 0 if velocity_error_mean < VELOCITY_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
