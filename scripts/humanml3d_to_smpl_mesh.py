#!/usr/bin/env python3
"""Simple script to convert HumanML3D format to SMPL mesh visualization.

Usage:
    uv run python scripts/humanml3d_to_smpl_mesh.py input.npy output.html
    uv run python scripts/humanml3d_to_smpl_mesh.py input.npy output_dir/ --config balanced

Input formats supported:
    - .npy file with 263D HumanML3D features
    - .npy file with joint positions (N, 22, 3)
"""

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

from motion_toolbox.converters.humanml3d_to_smpl import HumanML3DToSMPLConverterFast
from motion_toolbox.visualization.gltf_utils_vertex_anim import build_gltf_vertex_animation


def main():
    """Convert HumanML3D format to SMPL mesh visualization with interactive viewer."""
    parser = argparse.ArgumentParser(
        description="Convert HumanML3D format to SMPL mesh visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default quality settings
  python humanml3d_to_smpl_mesh.py sample_data/humanml3d/000010.npy output.html

  # Use balanced (faster) configuration
  python humanml3d_to_smpl_mesh.py input.npy output.html --config balanced

  # Save to directory (creates .glb and .html)
  python humanml3d_to_smpl_mesh.py input.npy output_dir/ --save-params

  # Process joint positions directly
  python humanml3d_to_smpl_mesh.py joints.npy output.html --input-type joints
        """,
    )
    parser.add_argument("input", type=str, help="Input .npy file (HumanML3D 263D or joint positions)")
    parser.add_argument("output", type=str, help="Output .html file or directory")
    parser.add_argument(
        "--config",
        type=str,
        choices=["default", "balanced"],
        default="default",
        help="Configuration preset (default: quality with LBFGS 150)",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["humanml3d", "joints"],
        default="humanml3d",
        help="Input data type: humanml3d (263D) or joints (N,22,3)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use for computation (default: auto-detect cuda:0 or cpu)"
    )
    parser.add_argument(
        "--save-params", action="store_true", help="Save SMPL parameters (.npz) in addition to visualization"
    )
    parser.add_argument("--save-glb", action="store_true", help="Save GLB mesh file in addition to HTML")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for visualization (default: 20)")

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        return 1

    # Setup output paths
    output_path = Path(args.output)
    if output_path.suffix == "":
        # Output is a directory
        output_path.mkdir(parents=True, exist_ok=True)
        html_path = output_path / f"{input_path.stem}_mesh.html"
        glb_path = output_path / f"{input_path.stem}_mesh.glb"
        params_path = output_path / f"{input_path.stem}_smpl.npz"
    else:
        # Output is a file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html_path = output_path
        glb_path = output_path.with_suffix(".glb")
        params_path = output_path.with_suffix(".npz")

    # Auto-detect device if not specified
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("HumanML3D to SMPL Mesh Converter")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {html_path}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device} {'✓' if torch.cuda.is_available() else '(CPU - will be slower)'}")
    print()

    # Load input data
    print("Loading input data...")
    data = np.load(input_path)

    if args.input_type == "humanml3d":
        # 263D HumanML3D format
        if data.shape[-1] != 263:
            print(f"Warning: Expected 263D features, got shape {data.shape}")
            print("Trying to interpret as joint positions...")
            args.input_type = "joints"

    # Setup converter with config
    print(f"Initializing converter ({args.config} configuration)...")
    if args.config == "default":
        # Default: LBFGS 100, two-stage (fast & good quality)
        converter = HumanML3DToSMPLConverterFast(
            num_smplify_iters=100, use_lbfgs=True, single_stage=False, device=args.device
        )
        print("  - LBFGS 100 iterations")
        print("  - Two-stage optimization")
        print("  - Expected error: ~0.5cm")
    else:  # balanced
        # Balanced: LBFGS 150, two-stage (higher quality)
        converter = HumanML3DToSMPLConverterFast(
            num_smplify_iters=150, use_lbfgs=True, single_stage=False, device=args.device
        )
        print("  - LBFGS 150 iterations")
        print("  - Two-stage optimization")
        print("  - Expected error: ~0.43cm")

    print()

    # Convert to SMPL parameters
    print("Converting to SMPL parameters...")
    if args.input_type == "joints":
        result = converter.convert_with_positions(data, recompute_joints=True)
    else:
        result = converter.convert(data)

    print("✓ Conversion complete!")
    print(f"  - Frames: {len(result['global_orient'])}")
    print(f"  - Betas optimized: {result['betas'].std() > 0.01}")
    print()

    # Save SMPL parameters if requested
    if args.save_params:
        np.savez(
            params_path,
            global_orient=result["global_orient"],
            body_pose=result["body_pose"],
            transl=result["transl"],
            betas=result["betas"],
        )
        print(f"✓ Saved SMPL parameters: {params_path}")

    # Generate SMPL mesh
    print("Generating SMPL mesh...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl")

    smpl = smplx.create(
        model_path, model_type="smpl", gender="neutral", ext="pkl", batch_size=len(result["global_orient"])
    ).to(device)

    num_frames = len(result["global_orient"])
    output = smpl(
        global_orient=torch.from_numpy(result["global_orient"]).float().to(device),
        body_pose=torch.from_numpy(result["body_pose"]).float().to(device),
        transl=torch.from_numpy(result["transl"]).float().to(device),
        betas=torch.from_numpy(result["betas"]).unsqueeze(0).repeat(num_frames, 1).float().to(device),
    )

    vertices = output.vertices.detach().cpu().numpy()
    faces = smpl.faces

    print(f"✓ Generated mesh with {vertices.shape[1]} vertices")
    print()

    # Create GLB file
    print("Creating visualization...")
    gltf = build_gltf_vertex_animation(
        all_vertices=vertices,
        init_vertices=vertices[:1],
        faces=faces,
        fps=args.fps,
        include_floor=True,  # Floor embedded in GLB with vertex colors
    )

    # Always save GLB to file first
    gltf.save(str(glb_path))

    # Embed GLB in HTML or use as separate file
    if args.save_glb:
        # Use GLB as separate file
        glb_src = glb_path.name
        print(f"✓ Saved GLB file: {glb_path}")
    else:
        # Embed GLB in HTML as base64
        with open(glb_path, "rb") as f:
            glb_bytes = f.read()
        glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
        glb_src = f"data:model/gltf-binary;base64,{glb_base64}"
        # Delete the temporary GLB file since it's embedded
        glb_path.unlink()

    # Create HTML viewer with Three.js (black background, blue mesh, playback controls)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMPL Mesh - {input_path.name}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #000000;
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
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 25px;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            z-index: 100;
            min-width: 450px;
        }}

        #controls-title {{
            font-size: 10px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.7);
            text-transform: uppercase;
            letter-spacing: 1px;
            text-align: center;
            margin-bottom: 4px;
        }}

        #timeline-row {{
            display: flex;
            gap: 12px;
            align-items: center;
            width: 100%;
        }}

        #timeline-slider {{
            flex: 1;
            -webkit-appearance: none;
            appearance: none;
            height: 6px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 3px;
            outline: none;
            cursor: pointer;
        }}

        #timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            background: #2196F3;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        #timeline-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
            box-shadow: 0 0 10px rgba(33, 150, 243, 0.6);
        }}

        #timeline-slider::-moz-range-thumb {{
            width: 16px;
            height: 16px;
            background: #2196F3;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        #time-display {{
            color: rgba(255, 255, 255, 0.9);
            font-size: 11px;
            font-weight: 600;
            min-width: 90px;
            text-align: center;
            font-family: 'Courier New', monospace;
            display: flex;
            flex-direction: column;
            gap: 2px;
            line-height: 1.2;
        }}

        #time-display .time-seconds {{
            font-size: 12px;
        }}

        #time-display .time-frames {{
            font-size: 10px;
            color: rgba(255, 255, 255, 0.7);
        }}

        #button-row {{
            display: flex;
            gap: 10px;
            justify-content: center;
        }}

        button {{
            background: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 18px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        }}

        button:hover {{
            background: #1976D2;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
        }}

        button:active {{
            transform: translateY(0);
        }}

        #info-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 20px;
            border-radius: 12px;
            min-width: 200px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            z-index: 100;
        }}

        #info-panel h3 {{
            margin: 0 0 15px 0;
            font-size: 14px;
            font-weight: 600;
            color: #2196F3;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .info-item {{
            margin: 8px 0;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            color: rgba(255, 255, 255, 0.9);
        }}

        .info-label {{
            font-weight: 600;
            color: rgba(255, 255, 255, 0.7);
            margin-right: 10px;
        }}

        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 30px 50px;
            border-radius: 15px;
            font-size: 18px;
            font-weight: 600;
            z-index: 1000;
            text-align: center;
        }}

        .spinner {{
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto 0;
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
            <span class="info-label">Input:</span>
            <span>{input_path.name}</span>
        </div>
        <div class="info-item">
            <span class="info-label">Frames:</span>
            <span>{num_frames}</span>
        </div>
        <div class="info-item">
            <span class="info-label">Vertices:</span>
            <span>6890</span>
        </div>
        <div class="info-item">
            <span class="info-label">Config:</span>
            <span>{args.config}</span>
        </div>
        <div class="info-item">
            <span class="info-label">FPS:</span>
            <span>{args.fps}</span>
        </div>
    </div>

    <div id="controls">
        <div id="controls-title">Playback Controls</div>
        <div id="timeline-row">
            <input type="range" id="timeline-slider" min="0" max="1000" value="0" step="1">
            <div id="time-display">
                <div class="time-seconds">0.00s / 0.00s</div>
                <div class="time-frames">0f / {num_frames}f</div>
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
        camera.position.set(3, 2, 3);
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
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
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
        const animationFPS = {args.fps};

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

                        // Check if this is the floor mesh (by name or by checking for vertex colors)
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
                                color: 0x4a90e2,  // Blue color
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
                framesDiv.textContent = `${{currentFrame}}f / ${{totalFrames}}f`;
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
            renderer.render(scene, camera);
        }});

        // Playback buttons
        document.getElementById('play-btn').addEventListener('click', () => {{
            isPlaying = true;
            if (mixer) mixer.timeScale = 1;
            clock.getDelta();
        }});

        document.getElementById('pause-btn').addEventListener('click', () => {{
            isPlaying = false;
            if (mixer) mixer.timeScale = 0;
        }});

        document.getElementById('restart-btn').addEventListener('click', () => {{
            if (action) {{
                action.time = 0;
                mixer.timeScale = 1;
                mixer.update(0);
            }}
            timelineSlider.value = 0;
            updateTimeDisplay(0, animationDuration);
            clock.getDelta();
            renderer.render(scene, camera);
            isPlaying = true;
        }});

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

    html_path.write_text(html_content)
    print(f"✓ Saved HTML visualization: {html_path}")

    print()
    print("=" * 70)
    print("✅ Conversion complete!")
    print("=" * 70)
    print(f"Open in browser: file://{html_path.absolute()}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
