"""MANO mesh visualizer for bimanual hand motion.

Runs MANO forward kinematics (LBS) on each frame to get the full 778-vertex
hand mesh and renders both hands as an animated Three.js WebGL scene.

No dependency on smplx at runtime -- uses the converted *_numpy.pkl files
that contain the MANO model data in plain numpy format.

Usage:
    from motion_toolbox.visualization.mano_mesh_visualizer import MANOMeshVisualizer
    vis = MANOMeshVisualizer("data/body_models/mano")
    vis.visualize_positions(joint_positions_32x3, "hand_mesh.html")
    vis.visualize_mano_params(mano_dict, "hand_mesh.html")
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# MANO joint topology (16 joints per hand, 0=wrist)
MANO_KINTREE = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]


def _load_mano_numpy(mano_dir: str, is_right: bool) -> dict:
    """Load MANO model data from pre-converted numpy pkl."""
    fname = "MANO_RIGHT_numpy.pkl" if is_right else "MANO_LEFT_numpy.pkl"
    path = Path(mano_dir) / fname
    if not path.exists():
        # Try to convert on the fly from original pkl
        orig = path.parent / fname.replace("_numpy.pkl", ".pkl")
        if not orig.exists():
            raise FileNotFoundError(
                f"MANO model not found at {path} or {orig}. "
                "Download from https://mano.is.tue.mpg.de/ and place in the mano/ directory."
            )
        logger.info(f"Converting {orig.name} to numpy format...")
        _convert_mano_pkl(str(orig), str(path))
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _convert_mano_pkl(src: str, dst: str) -> None:
    """Convert chumpy-based MANO pkl to plain numpy pkl."""
    import numpy as np

    _aliases = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "unicode": np.str_,
    }
    for name, val in _aliases.items():
        if not hasattr(np, name):
            setattr(np, name, val)

    import pickle

    # patch numpy in chumpy's import environment
    with open(src, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    out = {}
    for k, v in raw.items():
        try:
            out[k] = np.array(v)
        except Exception:
            out[k] = v
    with open(dst, "wb") as f:
        pickle.dump(out, f)


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """Axis-angle (N,3) -> rotation matrices (N,3,3)."""
    angles = np.linalg.norm(r, axis=1, keepdims=True)  # (N,1)
    axes = r / np.where(angles > 1e-8, angles, 1.0)  # (N,3)
    c, s = np.cos(angles), np.sin(angles)  # (N,1)
    t = 1.0 - c
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    # Build rotation matrices row by row
    R = np.stack(
        [
            c[:, 0] + t[:, 0] * x * x,
            t[:, 0] * x * y - s[:, 0] * z,
            t[:, 0] * x * z + s[:, 0] * y,
            t[:, 0] * x * y + s[:, 0] * z,
            c[:, 0] + t[:, 0] * y * y,
            t[:, 0] * y * z - s[:, 0] * x,
            t[:, 0] * x * z - s[:, 0] * y,
            t[:, 0] * y * z + s[:, 0] * x,
            c[:, 0] + t[:, 0] * z * z,
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    return R


def mano_fk(
    model: dict,
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    transl: np.ndarray | None = None,
    betas: np.ndarray | None = None,
) -> np.ndarray:
    """Forward kinematics: MANO params -> posed vertices (T, 778, 3).

    Args:
        model:        MANO model dict (numpy format)
        global_orient: (T, 3) wrist global orientation (axis-angle)
        hand_pose:    (T, 45) finger pose (axis-angle, 15 joints × 3)
        transl:       (T, 3) wrist translation; zeros if None
        betas:        (10,) or (T,10) shape parameters; zeros if None

    Returns:
        posed_verts: (T, 778, 3) world-space mesh vertices
    """
    T = global_orient.shape[0]
    v_template = model["v_template"].astype(np.float32)  # (778, 3)
    shapedirs = model["shapedirs"].astype(np.float32)  # (778, 3, 10)
    posedirs = model["posedirs"].astype(np.float32)  # (778*3, 135)
    _jr = model["J_regressor"]  # (16, 778)
    # Stored as 0-dim object array wrapping a scipy sparse matrix
    if isinstance(_jr, np.ndarray) and _jr.dtype == object:
        _jr = _jr.item()  # unwrap
    if hasattr(_jr, "toarray"):
        _jr = _jr.toarray()
    J_regressor = np.array(_jr, dtype=np.float32)  # (16, 778)
    weights = model["weights"].astype(np.float32)  # (778, 16)
    kintree = model["kintree_table"].astype(int)  # (2, 16)
    parents = kintree[0]  # parent index per joint

    # 1. Shape blend
    if betas is None:
        betas = np.zeros((T, 10), dtype=np.float32)
    elif betas.ndim == 1:
        betas = np.tile(betas[None], (T, 1)).astype(np.float32)
    # shapedirs: (778, 3, 10), betas: (T, 10)
    # blend[t, v, j] = sum_k shapedirs[v, j, k] * betas[t, k]
    blend = np.einsum("ti,vji->tvj", betas, shapedirs)  # (T, 778, 3)
    v_shaped = v_template[None] + blend  # (T, 778, 3)

    # 2. Template joint positions from J_regressor
    J_template = J_regressor @ v_template  # (16, 3)

    # 3. Build per-frame local rotation matrices: (T, 16, 3, 3)
    all_aa = np.concatenate([global_orient, hand_pose], axis=1)  # (T, 48)
    all_aa = all_aa.reshape(T * 16, 3)
    Rs = _rodrigues(all_aa).reshape(T, 16, 3, 3)

    # 4. Pose blend shapes: posedirs is (778, 3, 135)
    Id = np.eye(3, dtype=np.float32)
    pose_feature = (Rs[:, 1:] - Id[None]).reshape(T, -1)  # (T, 135)
    # posedirs: (778, 3, 135) -> reshape to (778*3, 135)
    pd = posedirs.reshape(-1, posedirs.shape[-1])  # (778*3, 135)
    pose_blend = (pose_feature @ pd.T).reshape(T, 778, 3)
    v_posed = v_shaped + pose_blend  # (T, 778, 3)

    # 5. Global joint transforms (forward kinematics) -- fully batched
    #    Build (T, 16, 4, 4) global transforms
    G = np.zeros((T, 16, 4, 4), dtype=np.float32)
    G[:, :, 3, 3] = 1.0

    for j in range(16):
        p = int(parents[j])
        # Local bone vector (relative to parent, in template pose)
        if p >= 16:  # root (kintree stores 4294967295 for root parent)
            t_j = J_template[j]
        else:
            t_j = J_template[j] - J_template[p]

        # Local transform: (T, 4, 4)
        T_local = np.zeros((T, 4, 4), dtype=np.float32)
        T_local[:, :3, :3] = Rs[:, j]  # rotation
        T_local[:, :3, 3] = t_j  # translation (broadcast over T)
        T_local[:, 3, 3] = 1.0

        if p >= 16:
            G[:, j] = T_local
        else:
            G[:, j] = G[:, p] @ T_local

    # Subtract rest-pose joint contribution to get skinning transforms
    G_hat = G.copy()
    for j in range(16):
        rest_j = J_template[j]  # (3,)
        # G_hat[:, j, :3, 3] -= G[:, j, :3, :3] @ rest_j
        G_hat[:, j, :3, 3] -= np.einsum("tij,j->ti", G[:, j, :3, :3], rest_j)

    # 6. Linear blend skinning
    # v_posed: (T, 778, 3) -> homogeneous (T, 778, 4)
    v_hom = np.concatenate([v_posed, np.ones((T, 778, 1), dtype=np.float32)], axis=-1)
    # weights: (778, 16), G_hat: (T, 16, 4, 4)
    # weighted transform per vertex: (T, 778, 4, 4)
    W = np.einsum("vj,tjkl->tvkl", weights, G_hat)  # (T, 778, 4, 4)
    # apply: (T, 778, 4)
    v_world = np.einsum("tvkl,tvl->tvk", W, v_hom)[:, :, :3]  # (T, 778, 3)

    # 7. Translation
    if transl is not None:
        v_world += transl[:, None, :]  # broadcast (T,1,3)

    return v_world.astype(np.float32)


class MANOMeshVisualizer:
    """Render bimanual hand motion as animated MANO surface meshes.

    Uses LBS forward kinematics to compute the 778-vertex MANO hand mesh
    for every frame, then embeds the vertex animation in a Three.js scene.

    Args:
        mano_dir: Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl
                  (or their _numpy.pkl counterparts)

    Examples:
        >>> vis = MANOMeshVisualizer("data/body_models/mano")
        >>> vis.visualize_mano_params(mano_dict, "hands.html")
    """

    # Skin-tone material colors
    LEFT_COLOR = "#e8c4a0"
    RIGHT_COLOR = "#d4a07a"

    def __init__(self, mano_dir: str):
        self.mano_dir = mano_dir
        self._left_model = None
        self._right_model = None
        self._left_faces = None
        self._right_faces = None

    def _ensure_models(self):
        if self._left_model is None:
            logger.info("Loading MANO models...")
            self._left_model = _load_mano_numpy(self.mano_dir, is_right=False)
            self._right_model = _load_mano_numpy(self.mano_dir, is_right=True)
            self._left_faces = self._left_model["f"].astype(int).tolist()
            self._right_faces = self._right_model["f"].astype(int).tolist()
            logger.info("MANO models loaded (778 verts, 1538 faces each)")

    def visualize_mano_params(
        self,
        mano_dict: dict,
        output_path: str | Path,
        fps: int = 30,
        title: str = "MANO Hand Mesh",
    ) -> Path:
        """Render from raw MANO parameter dict.

        Expected keys: left_hand_pose (T,45), left_global_orient (T,3),
                       left_transl (T,3), right_* equivalents.
        """
        self._ensure_models()

        T = mano_dict["left_hand_pose"].shape[0]
        betas_l = mano_dict.get("betas_left", np.zeros((10,)))
        betas_r = mano_dict.get("betas_right", np.zeros((10,)))

        logger.info(f"Running MANO FK for {T} frames (left hand)...")
        verts_left = mano_fk(
            self._left_model,
            mano_dict["left_global_orient"].astype(np.float32),
            mano_dict["left_hand_pose"].astype(np.float32),
            mano_dict.get("left_transl", np.zeros((T, 3))).astype(np.float32),
            betas_l,
        )

        logger.info(f"Running MANO FK for {T} frames (right hand)...")
        verts_right = mano_fk(
            self._right_model,
            mano_dict["right_global_orient"].astype(np.float32),
            mano_dict["right_hand_pose"].astype(np.float32),
            mano_dict.get("right_transl", np.zeros((T, 3))).astype(np.float32),
            betas_r,
        )

        return self._generate_html(
            verts_left,
            verts_right,
            self._left_faces,
            self._right_faces,
            output_path,
            fps,
            title,
        )

    def visualize_positions(
        self,
        positions: np.ndarray,
        output_path: str | Path,
        fps: int = 30,
        title: str = "MANO Hand Skeleton",
        label: str = "Bimanual Hand",
    ) -> Path:
        """Render hand motion from joint positions (T, 32, 3) as a skeleton.

        When only joint positions are available (no MANO parameters), the full
        LBS mesh cannot be reconstructed. This method falls back to the
        capsule-skeleton renderer from HandVisualizer, which does not require
        MANO model files.

        For the full 778-vertex mesh, use ``visualize_mano_params`` or
        ``visualize_format_hand`` which require the MANO pkl model files.

        Args:
            positions: (T, 32, 3) world-space joint positions
            output_path: Path to save the HTML file
            fps: Frames per second
            title: Title shown in the visualization
            label: Label shown in the visualization overlay

        Returns:
            Path to the saved HTML file
        """
        from .hand_visualizer import HandVisualizer

        return HandVisualizer().visualize_positions(
            positions,
            output_path,
            fps=fps,
            title=title,
            label=label,
        )

    def visualize_format_hand(
        self,
        motion_393d: np.ndarray,
        output_path: str | Path,
        body_model_path: str | None = None,
        fps: int = 30,
        title: str = "MANO Hand Mesh",
        denorm_mean: np.ndarray | None = None,
        denorm_std: np.ndarray | None = None,
    ) -> Path:
        """Render from 393D format_hand tensor.

        Decodes 393D to MANO params, then runs FK.
        Requires body_model_path with MANO pkl files.
        """
        if denorm_mean is not None and denorm_std is not None:
            motion_393d = motion_393d * denorm_std + denorm_mean

        from ..converters.hand_motion_converters import HandRepToMANOConverter

        mano_path = body_model_path or self.mano_dir
        converter = HandRepToMANOConverter(body_model_path=mano_path)
        mano_dict = converter.convert(motion_393d)
        return self.visualize_mano_params(mano_dict, output_path, fps, title)

    def _generate_html(
        self,
        verts_left: np.ndarray,  # (T, 778, 3)
        verts_right: np.ndarray,  # (T, 778, 3)
        faces_left: list,  # (1538, 3) ints
        faces_right: list,  # (1538, 3) ints
        output_path: str | Path,
        fps: int,
        title: str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        T = verts_left.shape[0]
        logger.info(f"Generating MANO mesh HTML ({T} frames, 2×778 verts)...")

        # Compute scene geometry for auto-fit camera and floor placement
        all_verts = np.vstack([verts_left.reshape(-1, 3), verts_right.reshape(-1, 3)])
        v_min = all_verts.min(axis=0)
        v_max = all_verts.max(axis=0)
        center = ((v_min + v_max) / 2).tolist()  # [cx, cy, cz]
        bbox_diag = float(np.linalg.norm(v_max - v_min))
        cam_dist = max(bbox_diag * 2.2, 0.4)
        y_floor = float(v_min[1]) - 0.005  # slightly below fingertips
        floor_sz = max(bbox_diag * 4.0, 0.5)

        # Round to 4dp to keep JSON compact
        vl = np.round(verts_left, 4).tolist()
        vr = np.round(verts_right, 4).tolist()
        fl = faces_left

        # Offset right hand faces by 778 (they share a single vertex buffer)
        fr_offset = [[i + 778, j + 778, k + 778] for i, j, k in faces_right]

        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ background:#ffffff; overflow:hidden; font-family:system-ui,sans-serif; }}
        #info {{
            position:absolute; top:14px; left:50%; transform:translateX(-50%);
            background:rgba(0,0,0,0.55); color:#fff; padding:5px 16px;
            border-radius:18px; font-size:13px; font-weight:600; pointer-events:none;
        }}
        #badge {{
            position:absolute; top:14px; right:14px;
            background:rgba(0,0,0,0.08); border:1px solid rgba(0,0,0,0.12);
            color:#444; padding:4px 11px; border-radius:7px; font-size:11px; pointer-events:none;
        }}
        #controls {{
            position:absolute; bottom:18px; left:50%; transform:translateX(-50%);
            display:flex; gap:10px; align-items:center;
            background:rgba(30,30,30,0.85); padding:9px 16px; border-radius:26px;
            backdrop-filter:blur(12px);
        }}
        .btn {{
            width:32px; height:32px; border-radius:50%; background:rgba(255,255,255,0.15);
            border:none; cursor:pointer; font-size:13px; color:#fff;
            display:flex; align-items:center; justify-content:center; transition:background 0.15s;
        }}
        .btn:hover {{ background:rgba(255,255,255,0.28); }}
        #tl {{
            width:170px; height:4px; -webkit-appearance:none; appearance:none;
            background:rgba(255,255,255,0.2); border-radius:2px; outline:none;
        }}
        #tl::-webkit-slider-thumb {{
            -webkit-appearance:none; width:12px; height:12px;
            background:#fff; border-radius:50%; cursor:pointer;
        }}
        #fi {{ color:#bbb; font-size:11px; min-width:60px; text-align:center; }}
    </style>
</head>
<body>
    <div id="info">{title}</div>
    <div id="badge">MANO mesh · 2×778 verts</div>
    <div id="controls">
        <button class="btn" id="playBtn">⏸</button>
        <button class="btn" id="resetBtn">↺</button>
        <input type="range" id="tl" min="0" max="{T - 1}" value="0">
        <div id="fi"><span id="ti">0.00s</span><br><span style="opacity:0.5;font-size:10px">f<span id="fn">0</span></span></div>
    </div>
    <script type="importmap">
      {{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.161/build/three.module.js",
                   "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.161/examples/jsm/"}}}}
    </script>
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        const vertsLeft  = {json.dumps(vl)};
        const vertsRight = {json.dumps(vr)};
        const facesLeft  = {json.dumps(fl)};
        const facesRight = {json.dumps(fr_offset)};
        const fps        = {fps};
        const T          = {T};
        const CENTER     = new THREE.Vector3({center[0]}, {center[1]}, {center[2]});
        const CAM_DIST   = {cam_dist:.4f};
        const Y_FLOOR    = {y_floor:.4f};
        const FLOOR_SZ   = {floor_sz:.4f};

        // ── Renderer ──────────────────────────────────────────────────────────
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 0.9;
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        document.body.appendChild(renderer.domElement);

        // ── Scene ─────────────────────────────────────────────────────────────
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);
        scene.up = new THREE.Vector3(0, 1, 0);

        // ── Camera — isometric-style, Y-up ───────────────────────────────────
        const cam = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.001, 20);
        // 45° azimuth, ~30° elevation from the centre of both hands
        const ISO_AZ = Math.PI / 4;          // 45°
        const ISO_EL = Math.PI / 6;          // 30°
        cam.position.set(
            CENTER.x + CAM_DIST * Math.cos(ISO_EL) * Math.sin(ISO_AZ),
            CENTER.y + CAM_DIST * Math.sin(ISO_EL),
            CENTER.z + CAM_DIST * Math.cos(ISO_EL) * Math.cos(ISO_AZ)
        );
        cam.up.set(0, 1, 0);

        const controls = new OrbitControls(cam, renderer.domElement);
        controls.target.copy(CENTER);
        controls.enableDamping = true;
        controls.dampingFactor = 0.07;
        controls.minDistance = CAM_DIST * 0.1;
        controls.maxDistance = CAM_DIST * 6.0;
        controls.update();

        // ── Lighting (tuned for white background) ────────────────────────────
        scene.add(new THREE.AmbientLight(0xffffff, 0.65));

        const key = new THREE.DirectionalLight(0xfff5e8, 1.8);
        key.position.set(CENTER.x + 0.5, CENTER.y + 1.2, CENTER.z + 0.8);
        key.castShadow = true;
        key.shadow.mapSize.set(2048, 2048);
        key.shadow.camera.near = 0.01;
        key.shadow.camera.far  = 10;
        key.shadow.camera.left = key.shadow.camera.bottom = -1.5;
        key.shadow.camera.right = key.shadow.camera.top   =  1.5;
        key.shadow.bias = -0.0005;
        scene.add(key);

        const fill = new THREE.DirectionalLight(0xd0e8ff, 0.6);
        fill.position.set(CENTER.x - 0.6, CENTER.y + 0.4, CENTER.z - 0.5);
        scene.add(fill);

        const rim = new THREE.DirectionalLight(0xffe0c0, 0.3);
        rim.position.set(CENTER.x, CENTER.y - 0.3, CENTER.z - 0.8);
        scene.add(rim);

        // ── Checkerboard floor ────────────────────────────────────────────────
        const floorCanvas = document.createElement('canvas');
        floorCanvas.width = floorCanvas.height = 512;
        const fctx = floorCanvas.getContext('2d');
        const CHECKS = 24;
        const CS = floorCanvas.width / CHECKS;
        for (let i = 0; i < CHECKS; i++) {{
            for (let j = 0; j < CHECKS; j++) {{
                fctx.fillStyle = (i + j) % 2 === 0 ? '#d8d8d8' : '#f0f0f0';
                fctx.fillRect(i * CS, j * CS, CS, CS);
            }}
        }}
        const floorTex = new THREE.CanvasTexture(floorCanvas);
        floorTex.colorSpace = THREE.SRGBColorSpace;
        floorTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
        const floorGeo = new THREE.PlaneGeometry(FLOOR_SZ, FLOOR_SZ);
        floorGeo.rotateX(-Math.PI / 2);
        const floor = new THREE.Mesh(floorGeo,
            new THREE.MeshStandardMaterial({{ map: floorTex, roughness: 0.9, metalness: 0.0 }}));
        floor.position.set(CENTER.x, Y_FLOOR, CENTER.z);
        floor.receiveShadow = true;
        scene.add(floor);

        // ── Build combined mesh geometry ──────────────────────────────────────
        // Left + right share one BufferGeometry. Right face indices pre-offset by 778 in Python.
        const NV = 778;
        const allFaces = [...facesLeft, ...facesRight];
        const indexArr = new Uint32Array(allFaces.length * 3);
        allFaces.forEach((tri, i) => {{
            indexArr[i*3]   = tri[0];
            indexArr[i*3+1] = tri[1];
            indexArr[i*3+2] = tri[2];
        }});

        const posAttr = new THREE.BufferAttribute(new Float32Array(NV * 2 * 3), 3);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', posAttr);
        geom.setIndex(new THREE.BufferAttribute(indexArr, 1));

        const matL = new THREE.MeshStandardMaterial({{
            color: 0xe8c4a0, roughness: 0.5, metalness: 0.0, side: THREE.FrontSide,
        }});
        const matR = new THREE.MeshStandardMaterial({{
            color: 0xd4a07a, roughness: 0.5, metalness: 0.0, side: THREE.FrontSide,
        }});
        geom.addGroup(0, facesLeft.length * 3, 0);
        geom.addGroup(facesLeft.length * 3, facesRight.length * 3, 1);

        const mesh = new THREE.Mesh(geom, [matL, matR]);
        mesh.castShadow = true;
        mesh.receiveShadow = false;
        scene.add(mesh);

        // ── Frame update ──────────────────────────────────────────────────────
        function updateVertices(frame) {{
            const vl = vertsLeft[frame];
            const vr = vertsRight[frame];
            for (let i = 0; i < NV; i++) {{
                posAttr.setXYZ(i,      vl[i][0], vl[i][1], vl[i][2]);
                posAttr.setXYZ(i + NV, vr[i][0], vr[i][1], vr[i][2]);
            }}
            posAttr.needsUpdate = true;
            geom.computeVertexNormals();
        }}

        // ── Animation controls ────────────────────────────────────────────────
        let frame = 0, playing = true, lastT = 0;
        const fi = 1000 / fps;
        const playBtn  = document.getElementById('playBtn');
        const resetBtn = document.getElementById('resetBtn');
        const tl = document.getElementById('tl');
        const fn = document.getElementById('fn');
        const ti = document.getElementById('ti');

        function setFrame(f) {{
            frame = f; fn.textContent = f; tl.value = f;
            ti.textContent = (f / fps).toFixed(2) + 's';
            updateVertices(f);
        }}

        playBtn.addEventListener('click', () => {{
            playing = !playing;
            playBtn.textContent = playing ? '⏸' : '▶';
        }});
        resetBtn.addEventListener('click', () => {{ playing = false; playBtn.textContent = '▶'; setFrame(0); }});
        tl.addEventListener('input', e => {{ playing = false; playBtn.textContent = '▶'; setFrame(+e.target.value); }});

        setFrame(0);

        function animate(t) {{
            requestAnimationFrame(animate);
            if (playing && t - lastT > fi) {{
                frame = (frame + 1) % T;
                setFrame(frame);
                lastT = t;
            }}
            controls.update();
            renderer.render(scene, cam);
        }}

        window.addEventListener('resize', () => {{
            cam.aspect = innerWidth / innerHeight;
            cam.updateProjectionMatrix();
            renderer.setSize(innerWidth, innerHeight);
        }});

        animate(0);
    </script>
</body>
</html>'''

        output_path.write_text(html)
        logger.info(f"MANO mesh HTML saved: {output_path} ({output_path.stat().st_size:,} bytes)")
        return output_path
