"""Mesh visualization backend for unified visualizer.

This module provides mesh-based visualization for all supported body models
(SMPL, R15, MHR), wrapping the existing exporters with a unified interface.
"""

import logging
from pathlib import Path

import numpy as np

from ..core.motion import Motion
from .constants import SMPL_VERTEX_INDICES

logger = logging.getLogger(__name__)


class MeshVisualizer:
    """Backend for mesh visualization across multiple body models.

    Routes to appropriate exporter based on body_model parameter and
    provides a unified interface for mesh visualization.

    Supports:
    - SMPL (via GLBExporter)
    - R15 (via R15GLBExporter)
    - MHR (via MHRBodyModel) - requires MHR setup

    Attributes:
        body_model: Body model type ('smpl', 'r15', 'mhr')
        body_model_path: Path to body model files (for SMPL)
        fps: Frames per second
        include_floor: Whether to include checkerboard floor

    Examples:
        >>> vis = MeshVisualizer(body_model='smpl', body_model_path='/path/to/models')
        >>> vis.visualize_motion(motion_obj, "output.glb")
    """

    def __init__(
        self,
        body_model: str = "smpl",
        body_model_path: str | None = None,
        fps: int = 30,
        include_floor: bool = True,
        **kwargs,
    ):
        """Initialize mesh visualizer.

        Args:
            body_model: Body model type ('smpl', 'r15', 'mhr')
            body_model_path: Path to body model files (required for SMPL)
            fps: Frames per second
            include_floor: Whether to include checkerboard floor
            **kwargs: Additional arguments for specific exporters
        """
        self.body_model = body_model.lower()
        self.body_model_path = body_model_path
        self.fps = fps
        self.include_floor = include_floor
        self.kwargs = kwargs

        # Validate body model
        supported_models = {"smpl", "r15", "mhr"}
        if self.body_model not in supported_models:
            raise ValueError(f"Unsupported body model: {body_model}. Supported models: {supported_models}")

        # Initialize exporter based on body model
        if self.body_model == "smpl":
            self._init_smpl_exporter()
        elif self.body_model == "r15":
            self._init_r15_exporter()
        elif self.body_model == "mhr":
            self._init_mhr_exporter()

    def _init_smpl_exporter(self):
        """Initialize SMPL vertex animation exporter.

        Uses morph target animation instead of skeletal animation
        to avoid LBS banding artifacts on the mesh.
        """
        if self.body_model_path is None:
            # Default path
            self.body_model_path = str(Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl")

        logger.info(f"Initializing SMPL vertex animation exporter with model path: {self.body_model_path}")

        # We'll use vertex animation, store the model path for later
        self.exporter = None  # Will use custom export in visualize_motion

    def _init_r15_exporter(self):
        """Initialize R15 GLBExporter."""
        from .r15_glb_exporter import R15GLBExporter

        logger.info("Initializing R15 exporter")

        self.exporter = R15GLBExporter(
            fps=self.fps,
            include_floor=self.include_floor,
            **self.kwargs,
        )

    def _init_mhr_exporter(self):
        """Initialize MHR body model.

        MHR requires:
        - The mhr package (installed via setup_mhr.sh)
        - MHR assets at ~/.motion_toolbox/datasets/mhr/assets
        - Appropriate LD_LIBRARY_PATH for native libraries
        """
        from pathlib import Path

        # Check for MHR assets
        if self.body_model_path is None:
            self.body_model_path = str(Path.home() / ".motion_toolbox" / "datasets" / "mhr" / "assets")

        if not Path(self.body_model_path).exists():
            raise RuntimeError(
                f"MHR assets not found at {self.body_model_path}. Run 'bash scripts/setup_mhr.sh' to set up MHR."
            )

        logger.info(f"Initializing MHR exporter with assets: {self.body_model_path}")

        # We'll use vertex animation like SMPL, store the path for later
        self.exporter = None  # Will use custom export in visualize_motion

    def visualize_motion(
        self,
        motion: Motion,
        output_path: str | Path,
        generate_html: bool = True,
        skip_rotation: bool = False,
    ) -> Path:
        """Visualize Motion object as animated mesh.

        Args:
            motion: Motion object
            output_path: Path to save GLB file (and HTML if generate_html=True)
            generate_html: Whether to generate HTML viewer alongside GLB
            skip_rotation: Whether to skip model rotation in viewer (for 205D converted motion)

        Returns:
            Path to generated file (HTML if generated, else GLB)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply coordinate normalization for SMPL (Y-down -> Y-up, Z-flip)
        motion = self._normalize_motion_coordinates(motion)

        # Convert Motion if needed for specific body model
        motion_converted = self._convert_motion_if_needed(motion)

        # Generate GLB file
        if output_path.suffix == ".html":
            # User wants HTML, generate GLB with .glb extension
            glb_path = output_path.with_suffix(".glb")
        else:
            glb_path = output_path

        logger.info(f"Exporting {motion_converted.num_frames} frames to {self.body_model} mesh: {glb_path}")

        if self.body_model == "smpl":
            # Use vertex animation for SMPL to avoid LBS banding artifacts
            self._export_smpl_vertex_animation(motion_converted, str(glb_path))
        elif self.body_model == "mhr":
            # Use MHRBodyModel's export_glb method (tested and working)
            from ..core.body_models.mhr import MHRBodyModel

            body_model = MHRBodyModel(model_path=self.body_model_path, lod=1)
            body_model.export_glb(motion_converted, str(glb_path), fps=self.fps, include_floor=self.include_floor)
        else:
            self.exporter.export(motion_converted, str(glb_path))

        # Generate HTML viewer if requested
        if generate_html:
            html_path = glb_path.with_suffix(".html")
            # Use the skip_rotation flag saved earlier
            self._generate_html_viewer(glb_path, html_path, motion, skip_rotation=skip_rotation)
            logger.info(f"HTML viewer generated: {html_path}")
            return html_path

        return glb_path

    def _normalize_motion_coordinates(self, motion: Motion) -> Motion:
        """Normalize motion coordinates for mesh visualization.

        The GLB viewer uses a unified Three.js scene with rotation applied
        to the model, so we don't need to flip coordinates here.
        We just ensure floor alignment.

        Args:
            motion: Input Motion object

        Returns:
            Motion with floor-aligned coordinates
        """
        # The unified HTML viewer rotates the model to face camera
        # and uses its own checkerboard floor, so minimal processing needed
        return motion

    def _convert_motion_if_needed(self, motion: Motion) -> Motion:
        """Convert Motion to appropriate format for body model.

        Args:
            motion: Input Motion object

        Returns:
            Motion object compatible with target body model
        """
        # Check if motion is already compatible
        if self.body_model == "smpl":
            # SMPL works with 24-joint motion or can adapt
            if motion.num_joints == 24:
                return motion
            # For other joint counts, assume it's compatible
            # (actual conversion would happen in body model)
            return motion

        elif self.body_model == "r15":
            # R15 needs 16-joint motion
            # SMPLToR15Converter handles this internally
            return motion

        elif self.body_model == "mhr":
            # MHR needs model_data with model_parameters
            # Check if motion already has the required data
            if hasattr(motion, "model_data") and motion.model_data is not None:
                if "model_parameters" in motion.model_data:
                    return motion
            # If not, we can't convert - MHR parameters are complex
            raise ValueError(
                "MHR visualization requires motion with model_data containing 'model_parameters'. "
                "Load MHR data from .pt files using MHRToMotionConverter."
            )

        return motion

    def _export_smpl_vertex_animation(self, motion: Motion, output_path: str):
        """Export SMPL motion using vertex animation (morph targets).

        This avoids LBS banding artifacts by computing all vertex positions
        upfront and using morph target animation instead of skeletal animation.
        """
        import smplx
        import torch

        from .gltf_utils_vertex_anim import build_gltf_vertex_animation

        device = torch.device("cpu")

        # Create SMPL model
        smpl = smplx.create(
            self.body_model_path,
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=motion.num_frames,
        ).to(device)

        # Get motion parameters
        global_orient = torch.from_numpy(motion.joint_rotations[:, 0:1, :]).float().to(device)
        body_pose = torch.from_numpy(motion.joint_rotations[:, 1:, :]).float().to(device)
        transl = torch.from_numpy(motion.root_translation).float().to(device)

        # Get betas from motion or use zeros
        if hasattr(motion, "betas") and motion.betas is not None:
            betas = motion.betas
            if betas.ndim == 1:
                betas = np.tile(betas, (motion.num_frames, 1))
            betas = torch.from_numpy(betas).float().to(device)
        else:
            betas = torch.zeros(motion.num_frames, 10).float().to(device)

        # Flatten rotations to axis-angle vectors (3,) -> (3,)
        global_orient = global_orient.reshape(motion.num_frames, 3)
        body_pose = body_pose.reshape(motion.num_frames, 69)

        # Run SMPL forward pass
        output = smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas,
        )

        vertices = output.vertices.detach().cpu().numpy()
        faces = smpl.faces

        logger.info(f"SMPL vertex animation: {vertices.shape[0]} frames, {vertices.shape[1]} vertices")

        # Check if mesh is upside down (head below feet) and flip if needed
        # Use defined vertex indices for head and feet from constants
        head_vert_y = float(vertices[0, SMPL_VERTEX_INDICES["head"], 1])
        left_foot_y = float(vertices[0, SMPL_VERTEX_INDICES["left_foot"], 1])
        right_foot_y = float(vertices[0, SMPL_VERTEX_INDICES["right_foot"], 1])
        feet_avg_y = (left_foot_y + right_foot_y) / 2
        is_head_above_feet = head_vert_y > feet_avg_y

        if not is_head_above_feet:
            # Mesh is inverted - apply 180° rotation around X axis
            # This flips Y and Z: (x, y, z) -> (x, -y, -z)
            vertices[:, :, 1] = -vertices[:, :, 1]
            vertices[:, :, 2] = -vertices[:, :, 2]
            logger.debug("Auto-flipped inverted SMPL mesh")

        # Apply floor correction to vertices (like StableMotion does)
        y_min = float(vertices[:, :, 1].min())
        if y_min < 0:
            vertices[:, :, 1] -= y_min

        # Build GLB with vertex animation
        gltf = build_gltf_vertex_animation(
            all_vertices=vertices,
            init_vertices=vertices[:1],
            faces=faces,
            fps=self.fps,
            include_floor=self.include_floor,
        )

        gltf.save(output_path)
        logger.info(f"Exported SMPL vertex animation GLB: {output_path}")

    def _generate_html_viewer(
        self,
        glb_path: Path,
        html_path: Path,
        motion: Motion,
        skip_rotation: bool = False,
    ):
        """Generate unified Three.js HTML viewer for GLB file.

        Uses the same scene setup as skeleton visualizer for consistency:
        - White background
        - Checkerboard floor
        - Same camera/lighting
        - Same Roblox-styled controls

        Body model-specific transformations:
        - SMPL: Rotate 180° around X (Y-down → Y-up)
        - R15: Already Y-up in GLB, just needs floor offset adjustment
        - MHR: TBD
        """
        import base64

        # Read and encode GLB as base64
        with open(glb_path, "rb") as f:
            glb_data = f.read()
        glb_base64 = base64.b64encode(glb_data).decode("utf-8")

        title = f"{self.body_model.upper()} Animation"
        num_frames = motion.num_frames
        fps = self.fps
        body_model = self.body_model

        # Generate unified Three.js HTML (matching skeleton visualizer)
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #ffffff;
            color: #333;
            overflow: hidden;
        }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #controls {{
            position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
            display: flex; gap: 12px; align-items: center; z-index: 100;
            background: rgba(40, 40, 40, 0.85); padding: 12px 20px; border-radius: 30px;
            backdrop-filter: blur(10px);
        }}
        .ctrl-btn {{
            width: 36px; height: 36px; border-radius: 50%;
            background: rgba(255, 255, 255, 0.15); border: none;
            cursor: pointer; font-size: 14px; color: #fff;
            display: flex; align-items: center; justify-content: center;
            transition: background 0.2s;
        }}
        .ctrl-btn:hover {{ background: rgba(255, 255, 255, 0.25); }}
        #timeline {{
            width: 200px; height: 4px;
            -webkit-appearance: none; appearance: none;
            background: rgba(255, 255, 255, 0.2); border-radius: 2px; outline: none;
        }}
        #timeline::-webkit-slider-thumb {{
            -webkit-appearance: none; appearance: none;
            width: 14px; height: 14px; background: #fff;
            border-radius: 50%; cursor: pointer;
        }}
        #timeline::-moz-range-thumb {{
            width: 14px; height: 14px; background: #fff;
            border-radius: 50%; cursor: pointer; border: none;
        }}
        #frameInfo {{
            color: #fff; font-size: 12px; font-weight: 500;
            min-width: 90px; text-align: center; opacity: 0.9;
            display: flex; flex-direction: column; line-height: 1.4;
        }}
        #timeInfo {{
            font-size: 13px; font-weight: 600;
        }}
        #frameDisplay {{
            font-size: 11px; opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="controls">
        <button class="ctrl-btn" id="playBtn">⏸</button>
        <button class="ctrl-btn" id="resetBtn">⏮</button>
        <input type="range" id="timeline" min="0" max="{num_frames - 1}" value="0">
        <div id="frameInfo"><div id="timeInfo">0.00s</div><div id="frameDisplay"><span id="frameNum">0</span> / {num_frames - 1}</div></div>
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
        
        const fps = {fps};
        const numFrames = {num_frames};
        const bodyModel = '{body_model}';
        const skipRotation = {str(skip_rotation).lower()};
        
        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);
        
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(3, 2, 3);
        camera.up.set(0, 1, 0);
        camera.lookAt(0, 1, 0);
        
        const renderer = new THREE.WebGLRenderer({{ 
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance',
        }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;
        container.appendChild(renderer.domElement);
        
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 1, 0);
        controls.update();
        
        // Lighting (matching working SMPL script)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.1;
        directionalLight.shadow.camera.far = 50;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
        directionalLight.shadow.bias = -0.0001;
        scene.add(directionalLight);
        
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
        fillLight.position.set(-5, 8, -5);
        scene.add(fillLight);
        
        // Checkerboard floor
        const floorSize = 10;
        const floorGeometry = new THREE.PlaneGeometry(floorSize, floorSize);
        floorGeometry.rotateX(-Math.PI / 2);
        
        const floorCanvas = document.createElement('canvas');
        floorCanvas.width = 512;
        floorCanvas.height = 512;
        const ctx = floorCanvas.getContext('2d');
        const checks = 20;
        const checkSize = floorCanvas.width / checks;
        for (let i = 0; i < checks; i++) {{
            for (let j = 0; j < checks; j++) {{
                ctx.fillStyle = (i + j) % 2 === 0 ? '#cccccc' : '#eeeeee';
                ctx.fillRect(i * checkSize, j * checkSize, checkSize, checkSize);
            }}
        }}
        const floorTexture = new THREE.CanvasTexture(floorCanvas);
        floorTexture.colorSpace = THREE.SRGBColorSpace;
        floorTexture.anisotropy = renderer.capabilities.getMaxAnisotropy();
        const floorMaterial = new THREE.MeshBasicMaterial({{ map: floorTexture }});
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.receiveShadow = true;
        scene.add(floor);
        
        // Load GLB
        const loader = new GLTFLoader();
        const glbData = atob('{glb_base64}');
        const glbArray = new Uint8Array(glbData.length);
        for (let i = 0; i < glbData.length; i++) {{
            glbArray[i] = glbData.charCodeAt(i);
        }}
        
        let mixer = null;
        let action = null;
        const clock = new THREE.Clock();
        let duration = 0;
        
        loader.parse(glbArray.buffer, '', (gltf) => {{
            const model = gltf.scene;
            
            // Body model-specific coordinate transformations
            // SMPL meshes are now exported with correct Y-up orientation
            // No rotation needed - GLB is directly visualizable
            if (bodyModel === 'smpl') {{
                // No rotation needed - mesh vertices are already correct
                // Just ensure facing camera if needed (optional Y rotation)
                // model.rotation.y = Math.PI;  // Uncomment if you want to face camera
            }} else if (bodyModel === 'r15') {{
                // R15 needs to be flipped around X axis
                model.rotation.x = Math.PI;
                
                // After X rotation, Y is inverted. The R15 exporter already applied
                // a floor offset of ~1.168m, but it's now inverted.
                // We need to shift down to compensate (the model is floating high up)
                // Wait for model to be processed, then align
                setTimeout(() => {{
                    model.updateMatrixWorld(true);
                    const box = new THREE.Box3().setFromObject(model);
                    // Shift so min.y = 0 (feet on floor)
                    model.position.y = -box.min.y;
                }}, 100);
            }} else if (bodyModel === 'mhr') {{
                // MHR is already in Y-up coordinate system
                // Floor alignment is handled in the export (MHRBodyModel.export_glb)
                // No additional transformations needed
            }}
            
            // Apply materials and hide attachment point meshes
            model.traverse((child) => {{
                // Hide R15 attachment points and bone tips (debug geometry)
                // Check node name and parent name for _Att, _end patterns
                const nodeName = child.name || '';
                const parentName = child.parent ? (child.parent.name || '') : '';
                const isAttachment = nodeName.endsWith('_Att') || parentName.endsWith('_Att') ||
                                    nodeName.endsWith('_end') || parentName.endsWith('_end') ||
                                    nodeName.endsWith('_end_end') || parentName.endsWith('_end_end');
                
                if (isAttachment) {{
                    child.visible = false;
                    return;
                }}
                
                if (child.isMesh) {{
                    // Compute vertex normals for smooth shading
                    if (child.geometry) {{
                        child.geometry.computeVertexNormals();
                    }}
                    
                    // Check if floor mesh (has vertex colors)
                    const isFloor = child.name === 'FloorNode' || 
                                   (child.geometry && child.geometry.attributes.color !== undefined);
                    
                    if (isFloor) {{
                        // Floor: use vertex colors (matching original script)
                        child.material = new THREE.MeshPhongMaterial({{
                            vertexColors: true,
                            shininess: 10,
                            side: THREE.DoubleSide,
                        }});
                        child.receiveShadow = true;
                    }} else {{
                        // Character: blue MeshPhongMaterial (matching original script)
                        child.material = new THREE.MeshPhongMaterial({{
                            color: 0x4a90e2,
                            shininess: 20,
                            specular: 0x222222,
                            flatShading: false,
                            side: THREE.DoubleSide,
                        }});
                        child.castShadow = true;
                        // Don't receive shadows on character to avoid self-shadow artifacts
                        child.receiveShadow = false;
                    }}
                }}
            }});
            
            scene.add(model);

            // Auto-fit camera to the loaded model bounding box (Y-up isometric view)
            const box = new THREE.Box3().setFromObject(model);
            const bCenter = new THREE.Vector3();
            box.getCenter(bCenter);
            const bSize = new THREE.Vector3();
            box.getSize(bSize);
            const maxDim = Math.max(bSize.x, bSize.y, bSize.z);
            const fitDist = maxDim * 1.8 / Math.tan(camera.fov * Math.PI / 360);
            const ISO_AZ = Math.PI / 4;   // 45°
            const ISO_EL = Math.PI / 6;   // 30°
            camera.position.set(
                bCenter.x + fitDist * Math.cos(ISO_EL) * Math.sin(ISO_AZ),
                bCenter.y + fitDist * Math.sin(ISO_EL),
                bCenter.z + fitDist * Math.cos(ISO_EL) * Math.cos(ISO_AZ)
            );
            camera.up.set(0, 1, 0);
            controls.target.copy(bCenter);
            controls.minDistance = fitDist * 0.1;
            controls.maxDistance = fitDist * 8;
            controls.update();
            
            // Setup animation
            if (gltf.animations && gltf.animations.length > 0) {{
                mixer = new THREE.AnimationMixer(model);
                action = mixer.clipAction(gltf.animations[0]);
                duration = gltf.animations[0].duration;
                action.play();
            }}
        }});
        
        // Animation controls
        let currentFrame = 0;
        let playing = true;
        
        const playBtn = document.getElementById('playBtn');
        const resetBtn = document.getElementById('resetBtn');
        const frameNum = document.getElementById('frameNum');
        const timeInfo = document.getElementById('timeInfo');
        const timeline = document.getElementById('timeline');
        
        function updateDisplay(frame) {{
            frameNum.textContent = frame;
            timeline.value = frame;
            timeInfo.textContent = (frame / fps).toFixed(2) + 's';
        }}
        
        playBtn.addEventListener('click', () => {{
            playing = !playing;
            playBtn.textContent = playing ? '⏸' : '▶';
            if (action) {{
                action.paused = !playing;
            }}
        }});
        
        resetBtn.addEventListener('click', () => {{
            currentFrame = 0;
            updateDisplay(currentFrame);
            if (mixer) {{
                mixer.setTime(0);
            }}
        }});
        
        timeline.addEventListener('input', (e) => {{
            currentFrame = parseInt(e.target.value);
            updateDisplay(currentFrame);
            if (mixer && duration > 0) {{
                mixer.setTime((currentFrame / numFrames) * duration);
            }}
            playing = false;
            playBtn.textContent = '▶';
            if (action) {{
                action.paused = true;
            }}
        }});
        
        function animate() {{
            requestAnimationFrame(animate);
            
            controls.update();  // For damping
            
            const delta = clock.getDelta();
            if (mixer && playing) {{
                mixer.update(delta);
                // Update frame display based on animation time
                if (duration > 0) {{
                    const time = mixer.time % duration;
                    currentFrame = Math.floor((time / duration) * numFrames);
                    updateDisplay(currentFrame);
                }}
            }}
            
            renderer.render(scene, camera);
        }}
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        animate();
    </script>
</body>
</html>'''

        html_path.write_text(html_content)
        logger.info(f"Unified HTML viewer created: {html_path}")


def visualize_mesh(
    motion: Motion,
    output_path: str | Path,
    body_model: str = "smpl",
    **kwargs,
) -> Path:
    """Convenience function to visualize motion as mesh.

    Args:
        motion: Motion object
        output_path: Path to save output
        body_model: Body model to use ('smpl', 'r15', 'mhr')
        **kwargs: Additional arguments for MeshVisualizer

    Returns:
        Path to generated file

    Examples:
        >>> visualize_mesh(motion, "output.html", body_model='smpl')
    """
    vis = MeshVisualizer(body_model=body_model, **kwargs)
    return vis.visualize_motion(motion, output_path)
