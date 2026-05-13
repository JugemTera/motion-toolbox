#!/usr/bin/env python3
"""Create triple visualization: Original Skeleton | SMPL Skeleton | SMPL Mesh

Uses saved SMPL parameters to generate all three views.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Create triple visualization")
    parser.add_argument("--sample", type=str, required=True, help="Sample ID")
    parser.add_argument("--data-dir", type=str, default="sample_data/humanml3d")
    parser.add_argument("--output-dir", type=str, default="outputs/humanml3d_smpl_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    print(f"Creating triple visualization for sample {args.sample}...")

    # Load original joints
    joints_path = data_dir / f"{args.sample}_joints.npy"
    joints = np.load(joints_path)
    print(f"Loaded joints: {joints.shape}")

    # Load SMPL parameters
    params_path = output_dir / f"{args.sample}_smpl_params.npz"
    params = np.load(params_path)
    print("Loaded SMPL params")

    # Generate SMPL mesh GLB
    from motion_toolbox.core import Motion, Skeleton
    from motion_toolbox.visualization import get_glb_exporter

    print("Generating SMPL mesh...")

    # Create Motion object from SMPL params
    global_orient = params["global_orient"]
    body_pose = params["body_pose"]
    transl = params["transl"]
    betas = params["betas"]

    # Combine to joint_rotations (24 joints × 3)
    # Need to pad body_pose from 69 to 72 (23 joints)
    if body_pose.shape[1] == 69:
        # smplx format - pad hands
        hand_pose = np.zeros((body_pose.shape[0], 6))
        full_body_pose = np.concatenate([body_pose[:, :63], hand_pose], axis=1)
    else:
        full_body_pose = body_pose

    joint_rotations = np.concatenate([global_orient, full_body_pose], axis=1)
    joint_rotations = joint_rotations.reshape(-1, 24, 3)

    # Create a minimal skeleton (SMPL will override)
    skeleton = Skeleton.from_smpl()

    motion = Motion(
        joint_rotations=joint_rotations,
        root_translation=transl,
        skeleton=skeleton,
        framerate=20,
        body_model_path=str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl/"),
    )

    # Export to GLB
    glb_path = output_dir / f"{args.sample}_smpl_mesh.glb"
    exporter = get_glb_exporter(motion.body_model_path)
    exporter.export(motion, str(glb_path))
    print(f"Exported SMPL mesh to: {glb_path}")

    # Compute SMPL joints for comparison
    import smplx

    smpl = smplx.create(
        motion.body_model_path,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=len(joints),
    )

    output = smpl(
        global_orient=torch.from_numpy(global_orient).float(),
        body_pose=torch.from_numpy(body_pose).float(),
        transl=torch.from_numpy(transl).float(),
        betas=torch.from_numpy(betas).float().unsqueeze(0).expand(len(joints), -1),
    )
    smpl_joints = output.joints.detach().numpy()[:, :22]

    # Create triple HTML visualization

    print("Creating triple HTML...")

    # Offset positions for side-by-side
    joints_left = joints.copy()
    smpl_joints_middle = smpl_joints + np.array([2.5, 0, 0])
    # Mesh will be at x=5.0 via GLB positioning

    html = _generate_triple_html(
        joints_left.tolist(),
        smpl_joints_middle.tolist(),
        str(glb_path.name),
        args.sample,
    )

    html_path = output_dir / f"{args.sample}_triple.html"
    html_path.write_text(html)
    print(f"✅ Created triple visualization: {html_path}")


def _generate_triple_html(pos_left, pos_middle, glb_filename, sample_id):
    """Generate HTML with three views."""
    import json

    from motion_toolbox.visualization.skeleton_comparison import CHAIN_COLORS, KINEMATIC_CHAIN

    chains_json = json.dumps(KINEMATIC_CHAIN)
    colors_json = json.dumps(CHAIN_COLORS)
    pos_left_json = json.dumps(pos_left)
    pos_middle_json = json.dumps(pos_middle)
    num_frames = len(pos_left)
    fps = 20

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Triple View - {sample_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            overflow: hidden;
        }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #info {{
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 30px;
            border-radius: 20px;
            text-align: center;
            z-index: 100;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }}
        #info h1 {{ 
            font-size: 1.5em; 
            margin: 0;
            color: rgba(255, 255, 255, 0.95);
        }}
        #labels {{
            position: absolute;
            top: 100px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 120px;
            z-index: 100;
        }}
        .label {{
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 20px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.9em;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            text-align: center;
            min-width: 140px;
        }}
        .label.left {{ color: #4ecdc4; }}
        .label.middle {{ color: #f9ca24; }}
        .label.right {{ color: #ff6b6b; }}
        #controls {{
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 25px;
            border-radius: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            z-index: 100;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }}
        button {{
            background: #2196F3;
            border: none;
            padding: 10px 24px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 0.95em;
            color: white;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        button:hover {{ 
            background: #1976D2;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4);
        }}
        button:active {{
            transform: translateY(0);
        }}
        #frameInfo {{
            color: rgba(255, 255, 255, 0.9);
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9em;
            background: rgba(255, 255, 255, 0.1);
        }}
        #frameInfo span {{
            color: #2196F3;
            font-weight: 700;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h1>Triple View - {sample_id}</h1>
        <div>Original Skeleton | SMPL Skeleton | SMPL Mesh</div>
    </div>
    <div id="controls">
        <button id="playBtn">Play</button>
        <button id="resetBtn">Reset</button>
        <div id="frameInfo">Frame: <span id="frameNum">0</span> / {num_frames - 1}</div>
    </div>
    <div id="labels">
        <div class="label left">Original (HumanML3D)</div>
        <div class="label middle">SMPL Skeleton</div>
        <div class="label right">SMPL Mesh</div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script>
        const posLeft = {pos_left_json};
        const posMiddle = {pos_middle_json};
        const chains = {chains_json};
        const chainColors = {colors_json};
        const fps = {fps};
        const numFrames = {num_frames};
        
        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf5f5f5);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(2.5, 2, 8);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(2.5, 1, 0);
        controls.update();
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);
        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
        dirLight1.position.set(5, 10, 5);
        scene.add(dirLight1);
        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight2.position.set(-5, 5, -5);
        scene.add(dirLight2);
        
        // Ground grid
        const gridHelper = new THREE.GridHelper(15, 30, 0xcccccc, 0xe0e0e0);
        scene.add(gridHelper);
        
        // Create skeletons
        function createSkeleton(color) {{
            const group = new THREE.Group();
            const jointGeom = new THREE.SphereGeometry(0.03);
            const jointMat = new THREE.MeshLambertMaterial({{ color: 0xffffff }});
            const joints = [];
            for (let i = 0; i < 22; i++) {{
                const joint = new THREE.Mesh(jointGeom, jointMat);
                joints.push(joint);
                group.add(joint);
            }}
            const bones = [];
            chains.forEach((chain, chainIdx) => {{
                const color = new THREE.Color(chainColors[chainIdx]);
                for (let i = 0; i < chain.length - 1; i++) {{
                    const geom = new THREE.BufferGeometry();
                    const positions = new Float32Array(6);
                    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    const mat = new THREE.LineBasicMaterial({{ color: color, linewidth: 3 }});
                    const line = new THREE.Line(geom, mat);
                    bones.push({{ line, from: chain[i], to: chain[i + 1] }});
                    group.add(line);
                }}
            }});
            return {{ group, joints, bones }};
        }}
        
        const skelLeft = createSkeleton(0x4ecdc4);
        const skelMiddle = createSkeleton(0xf9ca24);
        scene.add(skelLeft.group);
        scene.add(skelMiddle.group);
        
        // Load GLB mesh
        let glbMixer = null;
        let glbAction = null;
        let glbDuration = 0;
        const loader = new THREE.GLTFLoader();
        loader.load('{glb_filename}', (gltf) => {{
            const mesh = gltf.scene;
            mesh.position.set(5, 0, 0);
            scene.add(mesh);
            
            console.log('GLB loaded, animations:', gltf.animations.length);
            
            if (gltf.animations && gltf.animations.length > 0) {{
                glbMixer = new THREE.AnimationMixer(mesh);
                const clip = gltf.animations[0];
                glbDuration = clip.duration;
                glbAction = glbMixer.clipAction(clip);
                glbAction.play();
                glbAction.paused = true;  // Control manually for frame sync
                
                console.log('Animation duration:', glbDuration, 'seconds');
                console.log('Frames:', numFrames, 'FPS:', fps);
            }}
        }}, undefined, (error) => {{
            console.error('Error loading GLB:', error);
        }});
        
        // Update skeleton
        function updateSkeleton(skel, positions) {{
            skel.joints.forEach((joint, i) => {{
                joint.position.set(positions[i][0], positions[i][1], positions[i][2]);
            }});
            skel.bones.forEach((bone) => {{
                const posAttr = bone.line.geometry.attributes.position;
                posAttr.array[0] = positions[bone.from][0];
                posAttr.array[1] = positions[bone.from][1];
                posAttr.array[2] = positions[bone.from][2];
                posAttr.array[3] = positions[bone.to][0];
                posAttr.array[4] = positions[bone.to][1];
                posAttr.array[5] = positions[bone.to][2];
                posAttr.needsUpdate = true;
            }});
        }}
        
        // Animation
        let currentFrame = 0;
        let playing = false;
        let lastTime = 0;
        const frameInterval = 1000 / fps;
        
        const playBtn = document.getElementById('playBtn');
        const resetBtn = document.getElementById('resetBtn');
        const frameNum = document.getElementById('frameNum');
        
        playBtn.addEventListener('click', () => {{
            playing = !playing;
            playBtn.textContent = playing ? 'Pause' : 'Play';
        }});
        
        resetBtn.addEventListener('click', () => {{
            currentFrame = 0;
            frameNum.textContent = currentFrame;
            if (glbMixer && glbClip) {{
                glbMixer.setTime(0);
            }}
        }});
        
        function animate(time) {{
            requestAnimationFrame(animate);
            
            if (playing && time - lastTime > frameInterval) {{
                currentFrame = (currentFrame + 1) % numFrames;
                frameNum.textContent = currentFrame;
                lastTime = time;
            }}
            
            // Sync GLB to current frame by setting action time directly
            if (glbMixer && glbAction && glbDuration > 0) {{
                const frameTime = (currentFrame / numFrames) * glbDuration;
                glbAction.time = frameTime;
                glbMixer.update(0);  // Force update without advancing time
            }}
            
            updateSkeleton(skelLeft, posLeft[currentFrame]);
            updateSkeleton(skelMiddle, posMiddle[currentFrame]);
            
            renderer.render(scene, camera);
        }}
        
        updateSkeleton(skelLeft, posLeft[0]);
        updateSkeleton(skelMiddle, posMiddle[0]);
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        animate(0);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
