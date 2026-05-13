"""Side-by-side skeleton visualization for comparing original vs converted motions.

This visualizer shows two animated skeletons side by side:
- Left: Original HumanML3D skeleton (22 joints)
- Right: Converted SMPL skeleton (for comparison)

Uses Three.js for interactive 3D visualization.
"""

import json
from pathlib import Path

import numpy as np

# HumanML3D 22-joint kinematic chains for visualization
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],  # Right leg: pelvis -> r_hip -> r_knee -> r_ankle -> r_foot
    [0, 1, 4, 7, 10],  # Left leg: pelvis -> l_hip -> l_knee -> l_ankle -> l_foot
    [0, 3, 6, 9, 12, 15],  # Spine: pelvis -> spine1 -> spine2 -> spine3 -> neck -> head
    [9, 14, 17, 19, 21],  # Right arm: spine3 -> r_collar -> r_shoulder -> r_elbow -> r_wrist
    [9, 13, 16, 18, 20],  # Left arm: spine3 -> l_collar -> l_shoulder -> l_elbow -> l_wrist
]

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

CHAIN_COLORS = [
    "#ff6b6b",  # Right leg - red
    "#4ecdc4",  # Left leg - cyan
    "#45b7d1",  # Spine - blue
    "#f9ca24",  # Right arm - yellow
    "#6c5ce7",  # Left arm - purple
]


def create_skeleton_comparison_html(
    positions_left: np.ndarray,
    positions_right: np.ndarray,
    output_path: str,
    title: str = "Skeleton Comparison",
    left_label: str = "Original (HumanML3D)",
    right_label: str = "Converted (SMPL)",
    fps: int = 20,
) -> str:
    """Create side-by-side skeleton comparison visualization.

    Args:
        positions_left: Left skeleton positions, shape (T, 22, 3)
        positions_right: Right skeleton positions, shape (T, 22, 3)
        output_path: Path to save HTML file
        title: Title for the visualization
        left_label: Label for left skeleton
        right_label: Label for right skeleton
        fps: Frames per second

    Returns:
        Path to generated HTML file
    """
    # Ensure same number of frames
    min_frames = min(len(positions_left), len(positions_right))
    positions_left = positions_left[:min_frames]
    positions_right = positions_right[:min_frames]

    # Offset right skeleton to be side by side
    offset = np.array([2.0, 0.0, 0.0])  # 2 meters to the right
    positions_right_offset = positions_right + offset

    # Convert to lists for JSON
    pos_left_json = positions_left.tolist()
    pos_right_json = positions_right_offset.tolist()

    html_content = _generate_html(
        pos_left_json,
        pos_right_json,
        title,
        left_label,
        right_label,
        fps,
        min_frames,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    return str(output_path)


def create_single_skeleton_html(
    positions: np.ndarray,
    output_path: str,
    title: str = "Skeleton Visualization",
    label: str = "Skeleton",
    fps: int = 20,
) -> str:
    """Create single skeleton visualization.

    Args:
        positions: Skeleton positions, shape (T, 22, 3)
        output_path: Path to save HTML file
        title: Title for the visualization
        label: Label for the skeleton
        fps: Frames per second

    Returns:
        Path to generated HTML file
    """
    pos_json = positions.tolist()

    html_content = _generate_single_html(
        pos_json,
        title,
        label,
        fps,
        len(positions),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    return str(output_path)


def _generate_html(
    pos_left: list,
    pos_right: list,
    title: str,
    left_label: str,
    right_label: str,
    fps: int,
    num_frames: int,
) -> str:
    """Generate HTML content for side-by-side comparison."""

    chains_json = json.dumps(KINEMATIC_CHAIN)
    colors_json = json.dumps(CHAIN_COLORS)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        #container {{ 
            width: 100vw; 
            height: 100vh; 
            position: relative;
        }}
        #info {{
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            padding: 15px 30px;
            border-radius: 10px;
            text-align: center;
            z-index: 100;
        }}
        #info h1 {{ font-size: 1.5em; margin-bottom: 10px; }}
        #labels {{
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 100px;
            z-index: 100;
        }}
        .label {{
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
        }}
        .label.left {{ color: #4ecdc4; }}
        .label.right {{ color: #f9ca24; }}
        #controls {{
            position: absolute;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 20px;
            align-items: center;
            z-index: 100;
        }}
        button {{
            background: #4ecdc4;
            border: none;
            padding: 10px 25px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            color: #1a1a2e;
            font-weight: bold;
        }}
        button:hover {{ background: #45b7d1; }}
        #frameInfo {{
            background: rgba(0,0,0,0.7);
            padding: 8px 15px;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h1>{title}</h1>
        <div>Frames: {num_frames} | FPS: {fps}</div>
    </div>
    <div id="controls">
        <button id="playBtn">Play</button>
        <button id="resetBtn">Reset</button>
        <div id="frameInfo">Frame: <span id="frameNum">0</span> / {num_frames - 1}</div>
    </div>
    <div id="labels">
        <div class="label left">{left_label}</div>
        <div class="label right">{right_label}</div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Data
        const posLeft = {json.dumps(pos_left)};
        const posRight = {json.dumps(pos_right)};
        const chains = {chains_json};
        const chainColors = {colors_json};
        const fps = {fps};
        const numFrames = {num_frames};
        
        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(0, 2, 6);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(1, 1, 0);
        controls.update();
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);
        
        // Ground grid
        const gridHelper = new THREE.GridHelper(10, 20, 0x444444, 0x333333);
        scene.add(gridHelper);
        
        // Create skeleton geometry
        function createSkeleton(positions, colorMultiplier = 1.0) {{
            const group = new THREE.Group();
            
            // Joints
            const jointGeom = new THREE.SphereGeometry(0.03);
            const jointMat = new THREE.MeshLambertMaterial({{ color: 0xffffff }});
            const joints = [];
            for (let i = 0; i < 22; i++) {{
                const joint = new THREE.Mesh(jointGeom, jointMat);
                joints.push(joint);
                group.add(joint);
            }}
            
            // Bones
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
        
        const skelLeft = createSkeleton(posLeft[0]);
        const skelRight = createSkeleton(posRight[0]);
        scene.add(skelLeft.group);
        scene.add(skelRight.group);
        
        // Update skeleton positions
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
        }});
        
        function animate(time) {{
            requestAnimationFrame(animate);
            
            if (playing && time - lastTime > frameInterval) {{
                currentFrame = (currentFrame + 1) % numFrames;
                frameNum.textContent = currentFrame;
                lastTime = time;
            }}
            
            updateSkeleton(skelLeft, posLeft[currentFrame]);
            updateSkeleton(skelRight, posRight[currentFrame]);
            
            renderer.render(scene, camera);
        }}
        
        // Initialize
        updateSkeleton(skelLeft, posLeft[0]);
        updateSkeleton(skelRight, posRight[0]);
        
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


def _generate_single_html(
    positions: list,
    title: str,
    label: str,
    fps: int,
    num_frames: int,
) -> str:
    """Generate HTML content for single skeleton visualization."""

    chains_json = json.dumps(KINEMATIC_CHAIN)
    colors_json = json.dumps(CHAIN_COLORS)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        #container {{ 
            width: 100vw; 
            height: 100vh; 
            position: relative;
        }}
        #info {{
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            padding: 15px 30px;
            border-radius: 10px;
            text-align: center;
            z-index: 100;
        }}
        #info h1 {{ font-size: 1.5em; margin-bottom: 10px; }}
        #controls {{
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 20px;
            align-items: center;
            z-index: 100;
        }}
        button {{
            background: #4ecdc4;
            border: none;
            padding: 10px 25px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            color: #1a1a2e;
            font-weight: bold;
        }}
        button:hover {{ background: #45b7d1; }}
        #frameInfo {{
            background: rgba(0,0,0,0.7);
            padding: 8px 15px;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h1>{title}</h1>
        <div>{label} | Frames: {num_frames} | FPS: {fps}</div>
    </div>
    <div id="controls">
        <button id="playBtn">Play</button>
        <button id="resetBtn">Reset</button>
        <div id="frameInfo">Frame: <span id="frameNum">0</span> / {num_frames - 1}</div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Data
        const positions = {json.dumps(positions)};
        const chains = {chains_json};
        const chainColors = {colors_json};
        const fps = {fps};
        const numFrames = {num_frames};
        
        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(0, 2, 4);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.update();
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);
        
        // Ground grid
        const gridHelper = new THREE.GridHelper(6, 12, 0x444444, 0x333333);
        scene.add(gridHelper);
        
        // Create skeleton
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
                const pos = new Float32Array(6);
                geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
                const mat = new THREE.LineBasicMaterial({{ color: color, linewidth: 3 }});
                const line = new THREE.Line(geom, mat);
                bones.push({{ line, from: chain[i], to: chain[i + 1] }});
                group.add(line);
            }}
        }});
        scene.add(group);
        
        // Update function
        function updateSkeleton(frame) {{
            const pos = positions[frame];
            joints.forEach((joint, i) => {{
                joint.position.set(pos[i][0], pos[i][1], pos[i][2]);
            }});
            bones.forEach((bone) => {{
                const posAttr = bone.line.geometry.attributes.position;
                posAttr.array[0] = pos[bone.from][0];
                posAttr.array[1] = pos[bone.from][1];
                posAttr.array[2] = pos[bone.from][2];
                posAttr.array[3] = pos[bone.to][0];
                posAttr.array[4] = pos[bone.to][1];
                posAttr.array[5] = pos[bone.to][2];
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
        }});
        
        function animate(time) {{
            requestAnimationFrame(animate);
            
            if (playing && time - lastTime > frameInterval) {{
                currentFrame = (currentFrame + 1) % numFrames;
                frameNum.textContent = currentFrame;
                lastTime = time;
            }}
            
            updateSkeleton(currentFrame);
            renderer.render(scene, camera);
        }}
        
        updateSkeleton(0);
        
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
