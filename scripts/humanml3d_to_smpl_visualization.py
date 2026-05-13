#!/usr/bin/env python3
"""Create HumanML3D visualization with SMPL mesh, checkerboard floor, and playback controls.

This script demonstrates the complete workflow:
1. Load HumanML3D joint positions
2. Convert to SMPL parameters using GPU-accelerated optimization
3. Generate SMPL mesh animation
4. Create GLB with embedded checkerboard floor
5. Generate interactive HTML viewer with play/pause controls

Usage:
    python create_humanml3d_visualization.py [input.npy] [output.html]

Example:
    python create_humanml3d_visualization.py \
        sample_data/humanml3d/000000_joints.npy \
        outputs/humanml3d_demo.html
"""

import base64
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

# Ensure we can import from motion_toolbox
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motion_toolbox.converters.joints2smpl.smplify_batched import BatchedJoints2SMPL
from motion_toolbox.visualization.gltf_utils_vertex_anim import build_gltf_vertex_animation


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path("sample_data/humanml3d/000000_joints.npy")

    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        output_file = Path("outputs") / f"{input_file.stem}_demo.html"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    glb_file = output_file.with_suffix(".glb")

    print("=" * 70)
    print("HumanML3D to SMPL Visualization Creator")
    print("=" * 70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    # Step 1: Load HumanML3D joints
    print("Step 1: Loading HumanML3D joint positions...")
    joints = np.load(input_file)  # Shape: (num_frames, 22, 3)
    num_frames = len(joints)
    print(f"  ✓ Loaded {num_frames} frames")
    print()

    # Step 2: Convert to SMPL parameters
    print("Step 2: Converting to SMPL parameters (GPU-accelerated)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    converter = BatchedJoints2SMPL(
        num_frames=num_frames,
        device_id=0 if torch.cuda.is_available() else -1,
        cuda=torch.cuda.is_available(),
        num_smplify_iters=20,  # Fast but good quality
        use_lbfgs=True,
    )

    result = converter.joint2smpl(joints)
    print("  ✓ Optimized SMPL parameters")
    print()

    # Step 3: Generate SMPL mesh
    print("Step 3: Generating SMPL mesh...")
    model_path = str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl")
    device_torch = torch.device(device)

    smpl = smplx.create(model_path, model_type="smpl", gender="neutral", ext="pkl", batch_size=num_frames).to(
        device_torch
    )

    # Result contains 'pose' (72D) which is global_orient (3D) + body_pose (69D)
    pose_params = torch.from_numpy(result["pose"]).float().to(device_torch)

    output = smpl(
        global_orient=pose_params[:, :3],  # First 3 dims
        body_pose=pose_params[:, 3:],  # Remaining 69 dims
        transl=torch.from_numpy(result["transl"]).float().to(device_torch),
        betas=torch.from_numpy(result["betas"]).float().to(device_torch),
    )

    vertices = output.vertices.detach().cpu().numpy()
    faces = smpl.faces

    # Align feet to floor (y=0)
    min_y = vertices[:, :, 1].min()
    vertices[:, :, 1] -= min_y

    print(f"  ✓ Generated mesh: {vertices.shape[1]} vertices, {len(faces)} faces")
    print()

    # Step 4: Create GLB with embedded checkerboard floor
    print("Step 4: Creating GLB with embedded floor...")
    gltf = build_gltf_vertex_animation(
        all_vertices=vertices,
        init_vertices=vertices[:1],
        faces=faces,
        fps=20,
        include_floor=True,  # Embed checkerboard floor in GLB
    )

    gltf.save(str(glb_file))
    print(f"  ✓ Saved GLB: {glb_file}")
    print()

    # Step 5: Create HTML viewer with playback controls
    print("Step 5: Creating interactive HTML viewer...")

    # Embed GLB in HTML as base64
    with open(glb_file, "rb") as f:
        glb_bytes = f.read()
    glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
    glb_src = f"data:model/gltf-binary;base64,{glb_base64}"

    # HTML template with play/pause controls
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HumanML3D Visualization - {input_file.name}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            overflow: hidden;
        }}

        #canvas-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}

        #controls {{
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.85);
            padding: 20px 30px;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            z-index: 100;
            min-width: 500px;
        }}

        #controls-title {{
            font-size: 11px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.7);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            text-align: center;
            margin-bottom: 5px;
        }}

        #timeline-row {{
            display: flex;
            gap: 15px;
            align-items: center;
            width: 100%;
        }}

        #timeline-slider {{
            flex: 1;
            -webkit-appearance: none;
            appearance: none;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            outline: none;
            cursor: pointer;
        }}

        #timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: #4a90e2;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 2px 6px rgba(74, 144, 226, 0.4);
        }}

        #timeline-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.6);
        }}

        #timeline-slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: #4a90e2;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 2px 6px rgba(74, 144, 226, 0.4);
        }}

        #time-display {{
            color: rgba(255, 255, 255, 0.95);
            font-size: 12px;
            font-weight: 600;
            min-width: 100px;
            text-align: right;
            font-family: 'Monaco', 'Courier New', monospace;
            display: flex;
            flex-direction: column;
            gap: 3px;
            line-height: 1.3;
        }}

        #time-display .time-seconds {{
            font-size: 13px;
            color: #4a90e2;
        }}

        #time-display .time-frames {{
            font-size: 10px;
            color: rgba(255, 255, 255, 0.6);
        }}

        #button-row {{
            display: flex;
            gap: 12px;
            justify-content: center;
        }}

        button {{
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
            min-width: 100px;
        }}

        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(74, 144, 226, 0.5);
            background: linear-gradient(135deg, #5a9ef2 0%, #4585cd 100%);
        }}

        button:active {{
            transform: translateY(0);
        }}

        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}

        #info-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 20px;
            border-radius: 12px;
            min-width: 220px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            z-index: 100;
        }}

        #info-panel h3 {{
            margin: 0 0 15px 0;
            font-size: 14px;
            font-weight: 600;
            color: #4a90e2;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }}

        .info-item {{
            margin: 10px 0;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            color: rgba(255, 255, 255, 0.95);
        }}

        .info-label {{
            font-weight: 600;
            color: rgba(255, 255, 255, 0.7);
            margin-right: 15px;
        }}

        .info-value {{
            font-family: 'Monaco', 'Courier New', monospace;
        }}

        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 40px 60px;
            border-radius: 15px;
            font-size: 18px;
            font-weight: 600;
            z-index: 1000;
            text-align: center;
        }}

        .spinner {{
            border: 4px solid rgba(255, 255, 255, 0.2);
            border-top: 4px solid #4a90e2;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 25px auto 0;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div id="loading">
        Loading animation...
        <div class="spinner"></div>
    </div>

    <div id="canvas-container"></div>

    <div id="info-panel">
        <h3>Animation Info</h3>
        <div class="info-item">
            <span class="info-label">Source:</span>
            <span class="info-value">{input_file.name}</span>
        </div>
        <div class="info-item">
            <span class="info-label">Frames:</span>
            <span class="info-value">{num_frames}</span>
        </div>
        <div class="info-item">
            <span class="info-label">Duration:</span>
            <span class="info-value">{num_frames / 20:.1f}s</span>
        </div>
        <div class="info-item">
            <span class="info-label">FPS:</span>
            <span class="info-value">20</span>
        </div>
        <div class="info-item">
            <span class="info-label">Vertices:</span>
            <span class="info-value">6,890</span>
        </div>
    </div>

    <div id="controls">
        <div id="controls-title">Playback Controls</div>
        <div id="timeline-row">
            <input type="range" id="timeline-slider" min="0" max="1000" value="0" step="1">
            <div id="time-display">
                <div class="time-seconds">0.00s / {num_frames / 20:.2f}s</div>
                <div class="time-frames">Frame 0 / {num_frames}</div>
            </div>
        </div>
        <div id="button-row">
            <button id="play-btn">▶ Play</button>
            <button id="pause-btn">⏸ Pause</button>
            <button id="restart-btn">⏮ Restart</button>
        </div>
    </div>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        // Setup scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);  // White background

        // Setup camera
        const camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        camera.position.set(3, 1.5, 3);
        camera.lookAt(0, 1, 0);

        // Setup renderer
        const renderer = new THREE.WebGLRenderer({{
            antialias: true,
            alpha: false,
        }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        document.getElementById('canvas-container').appendChild(renderer.domElement);

        // Setup controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 1, 0);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.camera.left = -5;
        directionalLight.shadow.camera.right = 5;
        directionalLight.shadow.camera.top = 5;
        directionalLight.shadow.camera.bottom = -5;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
        fillLight.position.set(-5, 8, -5);
        scene.add(fillLight);

        // Load GLB (floor is embedded with vertex colors)
        const loader = new GLTFLoader();
        let mixer = null;
        let action = null;
        let isPlaying = true;
        let animationDuration = 0;
        const animationFPS = 20;

        loader.load(
            '{glb_src}',
            (gltf) => {{
                const model = gltf.scene;

                // Apply materials: blue for character, vertex colors for floor
                model.traverse((child) => {{
                    if (child.isMesh) {{
                        if (child.geometry) {{
                            child.geometry.computeVertexNormals();
                        }}

                        // Check if this is the floor mesh
                        const isFloor = child.name === 'FloorNode' || 
                                       (child.geometry.attributes.color !== undefined);

                        if (isFloor) {{
                            // Floor: use vertex colors from GLB
                            const material = new THREE.MeshPhongMaterial({{
                                vertexColors: true,
                                shininess: 10,
                                side: THREE.DoubleSide,
                            }});
                            child.material = material;
                            child.receiveShadow = true;
                        }} else {{
                            // Character: use blue material
                            const material = new THREE.MeshPhongMaterial({{
                                color: 0x4a90e2,
                                shininess: 20,
                                specular: 0x222222,
                                flatShading: false,
                                side: THREE.DoubleSide,
                            }});
                            child.material = material;
                            child.castShadow = true;
                            child.receiveShadow = true;
                        }}
                    }}
                }});

                scene.add(model);

                // Setup animation
                if (gltf.animations && gltf.animations.length > 0) {{
                    mixer = new THREE.AnimationMixer(model);
                    action = mixer.clipAction(gltf.animations[0]);
                    action.play();
                    animationDuration = gltf.animations[0].duration;
                    updateTimeDisplay(0, animationDuration);
                }}

                document.getElementById('loading').style.display = 'none';
            }},
            undefined,
            (error) => {{
                console.error('Error loading GLB:', error);
                document.getElementById('loading').textContent = 'Error loading animation';
            }}
        );

        // Timeline controls
        const timelineSlider = document.getElementById('timeline-slider');
        const timeDisplay = document.getElementById('time-display');
        let isScrubbing = false;

        function updateTimeDisplay(currentTime, totalTime) {{
            const currentFrame = Math.round(currentTime * animationFPS);
            const totalFrames = {num_frames};

            const secondsDiv = timeDisplay.querySelector('.time-seconds');
            const framesDiv = timeDisplay.querySelector('.time-frames');

            if (secondsDiv && framesDiv) {{
                secondsDiv.textContent = `${{currentTime.toFixed(2)}}s / ${{totalTime.toFixed(2)}}s`;
                framesDiv.textContent = `Frame ${{currentFrame}} / ${{totalFrames}}`;
            }}
        }}

        timelineSlider.addEventListener('input', (e) => {{
            if (!action) return;
            isScrubbing = true;
            const sliderValue = parseInt(e.target.value);
            const targetTime = (sliderValue / 1000) * animationDuration;
            action.time = targetTime;
            mixer.update(0);
            updateTimeDisplay(targetTime, animationDuration);
            renderer.render(scene, camera);
        }});

        timelineSlider.addEventListener('change', () => {{
            isScrubbing = false;
            clock.getDelta();
        }});

        // Playback buttons
        const playBtn = document.getElementById('play-btn');
        const pauseBtn = document.getElementById('pause-btn');
        const restartBtn = document.getElementById('restart-btn');

        playBtn.addEventListener('click', () => {{
            isPlaying = true;
            if (mixer) mixer.timeScale = 1;
            clock.getDelta();
            playBtn.disabled = true;
            pauseBtn.disabled = false;
        }});

        pauseBtn.addEventListener('click', () => {{
            isPlaying = false;
            if (mixer) mixer.timeScale = 0;
            playBtn.disabled = false;
            pauseBtn.disabled = true;
        }});

        restartBtn.addEventListener('click', () => {{
            if (action) {{
                action.time = 0;
                mixer.timeScale = 1;
                mixer.update(0);
            }}
            timelineSlider.value = 0;
            updateTimeDisplay(0, animationDuration);
            clock.getDelta();
            isPlaying = true;
            playBtn.disabled = true;
            pauseBtn.disabled = false;
        }});

        // Initial button states
        playBtn.disabled = true;
        pauseBtn.disabled = false;

        // Window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Animation loop
        const clock = new THREE.Clock();

        function animate() {{
            requestAnimationFrame(animate);

            if (isPlaying && !isScrubbing && mixer) {{
                const delta = clock.getDelta();
                mixer.update(delta);

                // Update timeline
                if (action && animationDuration > 0) {{
                    const currentTime = action.time || 0;
                    if (currentTime >= animationDuration) {{
                        // Loop animation
                        action.time = 0;
                        mixer.update(0);
                        timelineSlider.value = 0;
                        updateTimeDisplay(0, animationDuration);
                    }} else {{
                        const sliderValue = (currentTime / animationDuration) * 1000;
                        timelineSlider.value = sliderValue;
                        updateTimeDisplay(currentTime, animationDuration);
                    }}
                }}
            }} else if (isScrubbing) {{
                clock.getDelta();
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        animate();
    </script>
</body>
</html>"""

    output_file.write_text(html_content)
    print(f"  ✓ Saved HTML: {output_file}")
    print()

    print("=" * 70)
    print("✅ Visualization created successfully!")
    print("=" * 70)
    print()
    print("📂 Output files:")
    print(f"   - {output_file}")
    print(f"   - {glb_file}")
    print()
    print("🌐 Open in browser:")
    print(f"   file://{output_file.absolute()}")
    print()
    print("Features:")
    print("  ✓ White background with proper lighting")
    print("  ✓ Blue SMPL character mesh")
    print("  ✓ Checkerboard floor (grey/cyan)")
    print("  ✓ Play/Pause/Restart controls")
    print("  ✓ Timeline scrubber")
    print("  ✓ Frame counter")
    print("  ✓ GPU-accelerated conversion")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
