"""Hand motion visualization backend.

Renders bimanual hand motion as a high-quality capsule-mesh animation.
Uses hand-scale geometry (4-6mm joints, 3-4mm bones) with skin-tone
materials. No MANO model files required for this renderer.

For full MANO surface mesh (778 vertices), use MANOMeshVisualizer which
requires MANO .pkl model files from https://mano.is.tue.mpg.de/.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .constants import (
    HAND_CHAIN_COLORS,
    KINEMATIC_CHAINS,
)
from .skeleton_visualizer import SkeletonVisualizer

logger = logging.getLogger(__name__)

HAND_KINEMATIC_CHAINS = KINEMATIC_CHAINS["hand"]

# Hand-scale geometry constants
_JOINT_RADIUS = 0.006  # 6mm -- roughly the size of a knuckle joint
_BONE_RADIUS = 0.004  # 4mm -- slender finger segments
# Left/right hand base skin colors (matched at wrist, differentiated by chain colors)
_LEFT_SKIN = "#e8c9a0"
_RIGHT_SKIN = "#d4a574"


class HandVisualizer(SkeletonVisualizer):
    """Visualizer for bimanual hand motion.

    Renders 32-joint (2×16 MANO) hand animations as capsule-mesh geometry:
    small spheres at knuckles, cylinders along finger bones. Hand-scale
    sizing so individual fingers are clearly visible.

    Examples:
        >>> from motion_toolbox.visualization.hand_visualizer import HandVisualizer
        >>> vis = HandVisualizer()
        >>>
        >>> # From 393D format_hand tensor
        >>> vis.visualize_format_hand(motion_393d, "hand_anim.html")
        >>>
        >>> # From joint positions (T, 32, 3)
        >>> positions = np.random.randn(100, 32, 3)
        >>> vis.visualize_positions(positions, "hand_skeleton.html", fps=30)
    """

    def __init__(
        self,
        kinematic_chains: list[list[int]] | None = None,
        chain_colors: list[str] | None = None,
        joint_radius: float = _JOINT_RADIUS,
        bone_radius: float = _BONE_RADIUS,
    ):
        super().__init__(
            kinematic_chains=kinematic_chains or HAND_KINEMATIC_CHAINS,
            chain_colors=chain_colors or HAND_CHAIN_COLORS,
            joint_radius=joint_radius,
            bone_radius=bone_radius,
            auto_flip=False,  # Hand joints have no body head/feet semantics
        )
        self.joint_radius = joint_radius
        self.bone_radius = bone_radius

    def _generate_single_skeleton_html(
        self,
        positions: list,
        title: str,
        label: str,
        fps: int,
        num_frames: int,
    ) -> str:
        """Override with hand-optimised capsule-mesh renderer.

        Uses hand-scale joint (6mm) and bone (4mm) geometry with
        skin-tone materials and physically-based lighting so individual
        fingers are clearly visible.
        """
        chains_json = json.dumps(self.kinematic_chains)
        colors_json = json.dumps(self.chain_colors)
        positions_json = json.dumps(positions)
        joint_r = self.joint_radius
        bone_r = self.bone_radius

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #111318; overflow: hidden; font-family: 'Inter', system-ui, sans-serif; color: #ccc; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #info {{
            position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
            background: rgba(0,0,0,0.55); padding: 6px 18px; border-radius: 20px;
            font-size: 13px; font-weight: 600; letter-spacing: 0.02em; pointer-events: none;
        }}
        #badge {{
            position: absolute; top: 16px; right: 16px;
            background: rgba(255,200,80,0.15); border: 1px solid rgba(255,200,80,0.35);
            padding: 5px 12px; border-radius: 8px; font-size: 11px; color: #f5c842;
            pointer-events: none;
        }}
        #controls {{
            position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
            display: flex; gap: 10px; align-items: center; z-index: 100;
            background: rgba(20,20,30,0.88); padding: 10px 18px; border-radius: 28px;
            backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08);
        }}
        .ctrl-btn {{
            width: 34px; height: 34px; border-radius: 50%;
            background: rgba(255,255,255,0.1); border: none;
            cursor: pointer; font-size: 14px; color: #fff;
            display: flex; align-items: center; justify-content: center;
            transition: background 0.15s;
        }}
        .ctrl-btn:hover {{ background: rgba(255,255,255,0.2); }}
        #timeline {{
            width: 180px; height: 4px; -webkit-appearance: none; appearance: none;
            background: rgba(255,255,255,0.15); border-radius: 2px; outline: none;
        }}
        #timeline::-webkit-slider-thumb {{
            -webkit-appearance: none; width: 13px; height: 13px;
            background: #fff; border-radius: 50%; cursor: pointer;
        }}
        #frameInfo {{ color: #fff; font-size: 12px; min-width: 70px; text-align: center; opacity: 0.8; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="info">{label}</div>
        <div id="badge">Capsule-mesh · MANO 32J</div>
        <div id="controls">
            <button class="ctrl-btn" id="playBtn">⏸</button>
            <button class="ctrl-btn" id="resetBtn">↺</button>
            <input type="range" id="timeline" min="0" max="{num_frames - 1}" value="0">
            <div id="frameInfo"><span id="timeInfo">0.00s</span><br><span style="opacity:0.5;font-size:10px">frame <span id="frameNum">0</span></span></div>
        </div>
    </div>
    <script type="importmap">
      {{ "imports": {{ "three": "https://cdn.jsdelivr.net/npm/three@0.161/build/three.module.js",
                       "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.161/examples/jsm/" }} }}
    </script>
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        const positions = {positions_json};
        const chains    = {chains_json};
        const chainColors = {colors_json};
        const fps       = {fps};
        const numFrames = {num_frames};
        const JOINT_R   = {joint_r};
        const BONE_R    = {bone_r};

        // ── Scene ────────────────────────────────────────────────────────────
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;
        document.getElementById('container').appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111318);
        scene.fog = new THREE.FogExp2(0x111318, 0.4);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 10);
        camera.position.set(0, 0.92, 0.45);
        camera.lookAt(0, 0.88, 0);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0.88, 0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 0.05;
        controls.maxDistance = 1.5;

        // ── Lighting ─────────────────────────────────────────────────────────
        const ambient = new THREE.AmbientLight(0xfff8f0, 0.6);
        scene.add(ambient);

        const key = new THREE.DirectionalLight(0xfff4e0, 1.8);
        key.position.set(0.3, 1.5, 0.8);
        key.castShadow = true;
        key.shadow.mapSize.set(1024, 1024);
        key.shadow.camera.near = 0.01;
        key.shadow.camera.far = 5;
        key.shadow.camera.left = key.shadow.camera.bottom = -0.5;
        key.shadow.camera.right = key.shadow.camera.top = 0.5;
        scene.add(key);

        const fill = new THREE.DirectionalLight(0xc8e0ff, 0.5);
        fill.position.set(-0.5, 0.8, -0.3);
        scene.add(fill);

        const rim = new THREE.DirectionalLight(0xffe0c0, 0.3);
        rim.position.set(0, -0.5, -0.8);
        scene.add(rim);

        // ── Subtle floor grid ─────────────────────────────────────────────────
        const grid = new THREE.GridHelper(1, 20, 0x333344, 0x222233);
        grid.position.y = 0.75;
        grid.material.opacity = 0.3;
        grid.material.transparent = true;
        scene.add(grid);

        // ── Shared geometries ─────────────────────────────────────────────────
        const jointGeom = new THREE.SphereGeometry(JOINT_R, 10, 8);
        // Unit cylinder (height=1, will be scaled per bone)
        const boneGeom  = new THREE.CylinderGeometry(BONE_R, BONE_R, 1, 8);

        // ── Build per-chain materials + meshes ────────────────────────────────
        const group = new THREE.Group();
        scene.add(group);

        // Joint meshes (sparse – only for joints that appear in chains)
        const jointMeshes = [];
        const jointIndices = new Set();
        chains.forEach(chain => chain.forEach(i => jointIndices.add(i)));
        const numJoints = positions[0].length;
        for (let i = 0; i < numJoints; i++) {{
            if (jointIndices.has(i)) {{
                const isRight = (i >= 16);
                const mat = new THREE.MeshStandardMaterial({{
                    color: isRight ? 0xd4a574 : 0xe8c9a0,
                    roughness: 0.6, metalness: 0.0,
                }});
                const m = new THREE.Mesh(jointGeom, mat);
                m.castShadow = true;
                jointMeshes[i] = m;
                group.add(m);
            }}
        }}

        // Bone meshes per chain segment
        const boneMeshes = [];
        chains.forEach((chain, ci) => {{
            const col = new THREE.Color(chainColors[ci]);
            // Lighten slightly for a skin-like appearance
            col.lerp(new THREE.Color(0xe8c9a0), 0.35);
            const mat = new THREE.MeshStandardMaterial({{
                color: col, roughness: 0.55, metalness: 0.0,
            }});
            for (let i = 0; i < chain.length - 1; i++) {{
                const m = new THREE.Mesh(boneGeom, mat);
                m.castShadow = true;
                boneMeshes.push({{ mesh: m, from: chain[i], to: chain[i+1] }});
                group.add(m);
            }}
        }});

        // ── Update skeleton to frame ──────────────────────────────────────────
        const _up = new THREE.Vector3(0, 1, 0);
        const _dir = new THREE.Vector3();
        const _q   = new THREE.Quaternion();

        function updateSkeleton(frame) {{
            const pos = positions[frame];
            for (let i = 0; i < jointMeshes.length; i++) {{
                if (jointMeshes[i] && i < pos.length) {{
                    jointMeshes[i].position.set(pos[i][0], pos[i][1], pos[i][2]);
                }}
            }}
            boneMeshes.forEach(bone => {{
                const a = pos[bone.from], b = pos[bone.to];
                bone.mesh.position.set((a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2);
                _dir.set(b[0]-a[0], b[1]-a[1], b[2]-a[2]);
                const len = _dir.length();
                bone.mesh.scale.set(1, len, 1);
                if (len > 0.0001) {{
                    _dir.normalize();
                    _q.setFromUnitVectors(_up, _dir);
                    bone.mesh.quaternion.copy(_q);
                }}
            }});
        }}

        // ── Animation loop ────────────────────────────────────────────────────
        let currentFrame = 0, playing = true, lastTime = 0;
        const frameInterval = 1000 / fps;

        const playBtn  = document.getElementById('playBtn');
        const resetBtn = document.getElementById('resetBtn');
        const timeline = document.getElementById('timeline');
        const frameNum = document.getElementById('frameNum');
        const timeInfo = document.getElementById('timeInfo');

        function updateDisplay(f) {{
            frameNum.textContent = f;
            timeline.value = f;
            timeInfo.textContent = (f / fps).toFixed(2) + 's';
        }}

        playBtn.addEventListener('click',  () => {{ playing = !playing; playBtn.textContent = playing ? '⏸' : '▶'; }});
        resetBtn.addEventListener('click', () => {{ currentFrame = 0; updateDisplay(0); playing = false; playBtn.textContent = '▶'; }});
        timeline.addEventListener('input', e => {{ currentFrame = +e.target.value; updateDisplay(currentFrame); playing = false; playBtn.textContent = '▶'; }});

        function animate(time) {{
            requestAnimationFrame(animate);
            if (playing && time - lastTime > frameInterval) {{
                currentFrame = (currentFrame + 1) % numFrames;
                updateDisplay(currentFrame);
                lastTime = time;
            }}
            updateSkeleton(currentFrame);
            controls.update();
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
</html>'''

    def visualize_format_hand(
        self,
        motion_393d: np.ndarray,
        output_path: str | Path,
        fps: int = 30,
        title: str = "Hand Motion",
        label: str = "Bimanual Hand",
        denorm_mean: np.ndarray | None = None,
        denorm_std: np.ndarray | None = None,
    ) -> Path:
        """Visualize a 393D format_hand motion tensor.

        Args:
            motion_393d: (T, 393) hand motion features
            output_path: Path to save HTML file
            fps: Frames per second
            title: Title for the visualization
            label: Label for the skeleton
            denorm_mean: Optional Mean.npy for denormalization
            denorm_std: Optional Std.npy for denormalization

        Returns:
            Path to the saved HTML file
        """
        if motion_393d.ndim != 2 or motion_393d.shape[1] != 393:
            raise ValueError(f"Expected (T, 393), got {motion_393d.shape}")

        if denorm_mean is not None and denorm_std is not None:
            motion_393d = motion_393d * denorm_std + denorm_mean

        from ..converters.hand_motion_converters import FormatHandToMotionConverter

        converter = FormatHandToMotionConverter()
        result = converter.convert(motion_393d)
        positions = result["positions"]

        return self.visualize_positions(
            positions,
            output_path,
            fps=fps,
            title=title,
            label=label,
        )

    def visualize_mano_params(
        self,
        mano_dict: dict,
        output_path: str | Path,
        body_model_path: str,
        fps: int = 30,
        title: str = "MANO Hand Motion",
    ) -> Path:
        """Visualize from raw MANO parameters (requires MANO model files).

        Args:
            mano_dict: Dict with left/right hand MANO params
            output_path: Path to save HTML file
            body_model_path: Path to body model directory
            fps: Frames per second
            title: Title for the visualization

        Returns:
            Path to saved HTML file
        """
        from ..converters.hand_motion_converters import MANOToHandRepConverter

        converter = MANOToHandRepConverter(body_model_path=body_model_path)
        motion_393d = converter.convert(mano_dict)
        return self.visualize_format_hand(motion_393d, output_path, fps=fps, title=title)


def visualize_hand_motion(
    data: np.ndarray | dict | str,
    output_path: str | Path,
    body_model_path: str | None = None,
    fps: int = 30,
    title: str = "Hand Motion",
    denorm_mean: np.ndarray | None = None,
    denorm_std: np.ndarray | None = None,
) -> Path:
    """Convenience function for hand motion visualization.

    Automatically detects the input format (393D tensor, MANO params dict,
    or joint positions array) and visualizes accordingly.

    Args:
        data: Hand motion data in any supported format
        output_path: Output HTML file path
        body_model_path: Required if data is MANO params dict
        fps: Frames per second
        title: Visualization title
        denorm_mean: Optional denormalization mean
        denorm_std: Optional denormalization std

    Returns:
        Path to the saved HTML file
    """
    vis = HandVisualizer()

    if isinstance(data, str):
        data = np.load(data)

    if isinstance(data, dict):
        if body_model_path is None:
            raise ValueError("body_model_path required for MANO params visualization")
        return vis.visualize_mano_params(data, output_path, body_model_path, fps, title)

    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] == 393:
            return vis.visualize_format_hand(
                data,
                output_path,
                fps,
                title,
                denorm_mean=denorm_mean,
                denorm_std=denorm_std,
            )
        if data.ndim == 3 and data.shape[1] == 32:
            return vis.visualize_positions(data, output_path, fps=fps, title=title)

    raise ValueError(f"Cannot visualize hand data with shape {getattr(data, 'shape', type(data))}")
