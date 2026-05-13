"""Visualize IK solver output compared to ground truth joints.

This helps debug why IK initialization isn't helping by visualizing
what poses the IK actually produces.
"""

import argparse

import numpy as np
import smplx
import torch

from motion_toolbox.converters.joints2smpl.fast_ik_init import fast_smpl_ik, get_smpl_rest_offsets
from motion_toolbox.converters.joints2smpl.src import config


def forward_kinematics_smpl(pose: torch.Tensor, betas: torch.Tensor, smpl_model) -> torch.Tensor:
    """Get joint positions from SMPL pose parameters.

    Args:
        pose: SMPL pose (B, 72) in axis-angle
        betas: SMPL shape (B, 10)
        smpl_model: SMPL model instance

    Returns:
        Joint positions (B, 45, 3)
    """
    batch_size = pose.shape[0]

    with torch.no_grad():
        output = smpl_model(
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:72],
            betas=betas,
        )

    return output.joints


def visualize_ik_vs_gt(input_file: str, output_file: str, frame_idx: int = 0, num_opt_iters: int = 50):
    """Visualize IK solver output vs ground truth vs optimized result.

    Args:
        input_file: Path to HumanML3D joints file (.npy)
        output_file: Output HTML file path
        frame_idx: Which frame to visualize
        num_opt_iters: Number of optimization iterations to run
    """
    # Load ground truth joints
    gt_joints = np.load(input_file)
    if gt_joints.shape[1] != 22:
        raise ValueError(f"Expected 22 joints, got {gt_joints.shape[1]}")

    print(f"Loaded {gt_joints.shape[0]} frames with {gt_joints.shape[1]} joints")
    print(f"Visualizing frame {frame_idx}")

    # Select single frame
    frame_joints = gt_joints[frame_idx]  # Shape: (22, 3)

    # Extend to 24 joints for SMPL (zero-pad hands)
    joints_24 = np.zeros((24, 3), dtype=np.float32)
    joints_24[:22] = frame_joints

    # Convert to torch
    joints_torch = torch.from_numpy(joints_24).float().unsqueeze(0)  # (1, 24, 3)

    # Get SMPL offsets
    device = "cpu"
    smpl_offsets = get_smpl_rest_offsets(config.SMPL_MODEL_DIR, device=device)

    print("\nSMPL bone offsets (first 10):")
    for i in range(10):
        print(f"  Joint {i}: offset={smpl_offsets[i].numpy()}, norm={torch.norm(smpl_offsets[i]):.4f}")

    # Run IK
    print("\nRunning IK solver...")
    ik_pose = fast_smpl_ik(joints_torch, smpl_offsets)  # (1, 24, 3) axis-angle

    print(f"IK pose shape: {ik_pose.shape}")
    print(f"IK pose range: [{ik_pose.min():.3f}, {ik_pose.max():.3f}]")
    print(f"IK pose norm: {torch.norm(ik_pose):.3f}")

    # Check for NaN/Inf
    if torch.isnan(ik_pose).any():
        print("⚠️  WARNING: IK pose contains NaN!")
    if torch.isinf(ik_pose).any():
        print("⚠️  WARNING: IK pose contains Inf!")

    # Create SMPL model and run forward kinematics
    print("\nRunning SMPL forward kinematics...")
    smpl = smplx.create(config.SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=1)

    # Convert pose to (1, 72)
    pose_flat = ik_pose.reshape(1, -1)  # (1, 72)
    betas = torch.zeros(1, 10)  # Zero shape

    ik_joints = forward_kinematics_smpl(pose_flat, betas, smpl)  # (1, 45, 3)

    print(f"IK joints shape: {ik_joints.shape}")
    print(f"IK joints range: [{ik_joints.min():.3f}, {ik_joints.max():.3f}]")

    # Compare positions - both centered to pelvis for fair comparison
    gt_extended = torch.from_numpy(joints_24).float().unsqueeze(0)
    ik_joints_24 = ik_joints[:, :24, :]  # Take first 24 joints

    # Center both to their pelvis for comparison (IK output is already centered internally)
    gt_centered = gt_extended - gt_extended[:, 0:1, :]
    ik_centered = ik_joints_24 - ik_joints_24[:, 0:1, :]

    # Compute errors
    position_error = torch.norm(ik_centered - gt_centered, dim=-1)  # (1, 24)

    print("\nPer-joint position errors (cm):")
    joint_names = [
        "Pelvis",
        "L_Hip",
        "R_Hip",
        "Spine1",
        "L_Knee",
        "R_Knee",
        "Spine2",
        "L_Ankle",
        "R_Ankle",
        "Spine3",
        "L_Foot",
        "R_Foot",
        "Neck",
        "L_Collar",
        "R_Collar",
        "Head",
        "L_Shoulder",
        "R_Shoulder",
        "L_Elbow",
        "R_Elbow",
        "L_Wrist",
        "R_Wrist",
        "L_Hand",
        "R_Hand",
    ]

    for i in range(24):
        error_cm = position_error[0, i].item() * 100
        print(f"  {joint_names[i]:12s}: {error_cm:7.2f}cm")

    mean_error_cm = position_error[:, :22].mean().item() * 100  # Only first 22 (HumanML3D joints)
    print(f"\nMean error (first 22 joints): {mean_error_cm:.2f}cm")

    # Now run optimization with IK initialization
    print(f"\n{'=' * 60}")
    print(f"Running optimization with IK initialization ({num_opt_iters} iters)...")
    print(f"{'=' * 60}")

    from motion_toolbox.converters.joints2smpl.smplify_batched import BatchedJoints2SMPL

    # Use a single frame for visualization
    single_frame_joints = frame_joints[np.newaxis, :, :]  # (1, 22, 3)

    converter = BatchedJoints2SMPL(
        num_frames=1,
        device_id=-1,
        cuda=False,
        num_smplify_iters=num_opt_iters,
        batch_size=1,
        temporal_smooth=False,
        use_lbfgs=True,
        use_ik_init=True,
    )

    result = converter.joint2smpl(single_frame_joints)

    # Get optimized joint positions
    opt_pose = torch.from_numpy(result["pose"]).float()
    opt_betas = torch.from_numpy(result["betas"]).float()
    if opt_betas.ndim == 1:
        opt_betas = opt_betas.unsqueeze(0)
    opt_transl = torch.from_numpy(result["transl"]).float()

    with torch.no_grad():
        opt_output = smpl(global_orient=opt_pose[:, :3], body_pose=opt_pose[:, 3:], betas=opt_betas, transl=opt_transl)

    opt_joints_24 = opt_output.joints[:, :24, :]  # (1, 24, 3)

    # Compute optimized error (centered)
    opt_centered = opt_joints_24 - opt_joints_24[:, 0:1, :]
    opt_error = torch.norm(opt_centered - gt_centered, dim=-1)  # (1, 24)
    opt_mean_error_cm = opt_error[:, :22].mean().item() * 100

    print(f"\nOptimized result error (first 22 joints): {opt_mean_error_cm:.2f}cm")
    print(f"Improvement: {mean_error_cm:.2f}cm → {opt_mean_error_cm:.2f}cm")

    # Prepare data for visualization (3 skeletons)
    # IMPORTANT: Align all to same coordinate system!
    # GT comes in world space (pelvis at Y≈0.9m), IK and opt output are body-relative (pelvis at Y≈-0.2m)
    # We'll center all to pelvis, then position them at Y=1.0m for floor alignment

    gt_positions_raw = frame_joints  # (22, 3) - world space
    ik_positions_raw = ik_joints_24[0, :22, :].numpy()  # (22, 3) - body-relative
    opt_positions_raw = opt_joints_24[0, :22, :].numpy()  # (22, 3) - body-relative

    # Center all to their pelvis
    gt_positions = gt_positions_raw - gt_positions_raw[0:1, :]  # (22, 3) - centered
    ik_positions = ik_positions_raw - ik_positions_raw[0:1, :]  # (22, 3) - centered
    opt_positions = opt_positions_raw - opt_positions_raw[0:1, :]  # (22, 3) - centered

    # Position all at Y=1.0m (so feet are near floor at Y=0)
    floor_offset = np.array([0, 1.0, 0], dtype=np.float32)
    gt_positions = gt_positions + floor_offset
    ik_positions = ik_positions + floor_offset
    opt_positions = opt_positions + floor_offset

    # Create visualization data
    # Stack side by side with offset in X
    offset_x = 2.0  # Separate by 2 meters

    all_positions = np.zeros((3, 22, 3), dtype=np.float32)
    all_positions[0] = gt_positions  # GT on left
    all_positions[1] = ik_positions  # IK in middle
    all_positions[1, :, 0] += offset_x
    all_positions[2] = opt_positions  # Optimized on right
    all_positions[2, :, 0] += 2 * offset_x

    # Create HTML viewer with comparison
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>IK Solver Visualization</title>
    <style>
        body {{
            margin: 0;
            background: #1a1a1a;
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 20px;
            background: rgba(0,0,0,0.8);
            border-radius: 5px;
            max-width: 400px;
        }}
        #info h2 {{
            margin-top: 0;
            color: #4a90e2;
        }}
        .error {{
            color: #ff6b6b;
            font-weight: bold;
        }}
        .good {{
            color: #51cf66;
        }}
        canvas {{
            display: block;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="info">
        <h2>IK + Optimization Pipeline</h2>
        <p><strong>Frame:</strong> {frame_idx} | <strong>Opt Iters:</strong> {num_opt_iters}</p>
        
        <div style="margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 3px;">
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; margin-right: 8px; border-radius: 2px; vertical-align: middle; background: #4a90e2;"></span>
                <strong>Ground Truth</strong>
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; margin-right: 8px; border-radius: 2px; vertical-align: middle; background: #ff6b6b;"></span>
                <strong>IK Init</strong> ({mean_error_cm:.2f}cm error)
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; margin-right: 8px; border-radius: 2px; vertical-align: middle; background: #51cf66;"></span>
                <strong>Optimized</strong> ({opt_mean_error_cm:.2f}cm error)
            </div>
        </div>
        
        <p><strong>Pipeline:</strong><br>
        GT → IK ({mean_error_cm:.2f}cm) → Opt ({opt_mean_error_cm:.2f}cm)</p>
        
        <p><em>Rotate: Left mouse<br>
        Pan: Right mouse<br>
        Zoom: Scroll wheel</em></p>
    </div>
    
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);
        
        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(4, 1.5, 6);
        camera.lookAt(2, 1, 0);
        
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        scene.add(directionalLight);
        
        // Checkerboard floor
        const floorSize = 10;
        const floorDiv = 20;
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        const tileSize = canvas.width / floorDiv;
        for (let i = 0; i < floorDiv; i++) {{
            for (let j = 0; j < floorDiv; j++) {{
                ctx.fillStyle = (i + j) % 2 === 0 ? '#cccccc' : '#999999';
                ctx.fillRect(i * tileSize, j * tileSize, tileSize, tileSize);
            }}
        }}
        const floorTexture = new THREE.CanvasTexture(canvas);
        floorTexture.magFilter = THREE.NearestFilter;
        floorTexture.minFilter = THREE.NearestFilter;
        
        const floorGeometry = new THREE.PlaneGeometry(floorSize, floorSize);
        const floorMaterial = new THREE.MeshBasicMaterial({{map: floorTexture}});
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0;
        scene.add(floor);
        
        // Joint data
        const gtJoints = {gt_positions.tolist()};
        const ikJoints = {ik_positions.tolist()};
        const optJoints = {opt_positions.tolist()};
        
        // Skeleton connections (HumanML3D 22 joints)
        const skeleton = [
            [0, 1], [0, 2], [0, 3],  // Pelvis to hips and spine
            [1, 4], [4, 7], [7, 10],  // Left leg
            [2, 5], [5, 8], [8, 11],  // Right leg
            [3, 6], [6, 9], [9, 12], [12, 15],  // Spine to head
            [9, 13], [13, 16], [16, 18], [18, 20],  // Left arm
            [9, 14], [14, 17], [17, 19], [19, 21],  // Right arm
        ];
        
        // Draw skeleton
        function drawSkeleton(joints, color, offset) {{
            const material = new THREE.LineBasicMaterial({{ color: color, linewidth: 2 }});
            
            skeleton.forEach(([from, to]) => {{
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array([
                    joints[from][0] + offset, joints[from][1], joints[from][2],
                    joints[to][0] + offset, joints[to][1], joints[to][2]
                ]);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                const line = new THREE.Line(geometry, material);
                scene.add(line);
            }});
            
            // Draw joints as spheres
            const sphereGeometry = new THREE.SphereGeometry(0.02, 8, 8);
            const sphereMaterial = new THREE.MeshPhongMaterial({{ color: color }});
            joints.forEach(joint => {{
                const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
                sphere.position.set(joint[0] + offset, joint[1], joint[2]);
                scene.add(sphere);
            }});
        }}
        
        // Draw all three skeletons
        drawSkeleton(gtJoints, 0x4a90e2, 0);  // Blue = GT
        drawSkeleton(ikJoints, 0xff6b6b, {offset_x});  // Red = IK
        drawSkeleton(optJoints, 0x51cf66, {offset_x * 2});  // Green = Optimized
        
        // Camera controls (simple orbit)
        let isDragging = false;
        let previousMousePosition = {{ x: 0, y: 0 }};
        
        renderer.domElement.addEventListener('mousedown', (e) => {{
            isDragging = true;
        }});
        
        renderer.domElement.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                const deltaMove = {{
                    x: e.clientX - previousMousePosition.x,
                    y: e.clientY - previousMousePosition.y
                }};
                
                const deltaRotationQuaternion = new THREE.Quaternion()
                    .setFromEuler(new THREE.Euler(
                        (deltaMove.y * 0.01),
                        (deltaMove.x * 0.01),
                        0,
                        'XYZ'
                    ));
                
                camera.position.applyQuaternion(deltaRotationQuaternion);
                camera.lookAt(1, 1, 0);
            }}
            
            previousMousePosition = {{
                x: e.clientX,
                y: e.clientY
            }};
        }});
        
        renderer.domElement.addEventListener('mouseup', (e) => {{
            isDragging = false;
        }});
        
        // Zoom
        renderer.domElement.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const zoomSpeed = 0.1;
            const direction = new THREE.Vector3();
            camera.getWorldDirection(direction);
            camera.position.addScaledVector(direction, e.deltaY > 0 ? -zoomSpeed : zoomSpeed);
        }});
        
        // Render loop
        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""

    # Save HTML
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"\n✅ Visualization saved to: {output_file}")
    print("\nOpen in browser to see:")
    print("  - Left (Blue): Ground Truth")
    print(f"  - Middle (Red): IK Initialization ({mean_error_cm:.2f}cm error)")
    print(f"  - Right (Green): Optimized ({opt_mean_error_cm:.2f}cm error)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize IK solver output vs ground truth")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to HumanML3D joints file (.npy)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ik_debug_visualization.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Which frame to visualize (default: 0)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of optimization iterations to run (default: 50)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("IK + Optimization Visualization")
    print("=" * 60)

    visualize_ik_vs_gt(args.input_file, args.output, args.frame, args.iters)


if __name__ == "__main__":
    main()
