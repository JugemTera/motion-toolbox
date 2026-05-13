"""Hand motion representation converters for bimanual MANO-based formats.

Implements converters between MANO hand parameters and a 393-dimensional
bimanual hand motion representation, analogous to the 272-dim body format.

The format_hand (393D) layout:
  [0:3]     Root velocity 3D (midpoint of wrists in canonical frame)
  [3:9]     Root angular velocity 6D (orientation delta of hand system)
  [9:105]   Joint positions: 32 joints x 3 = 96 (relative to root, normalized)
  [105:201] Joint velocities: 32 joints x 3 = 96
  [201:393] Joint rotations: 32 joints x 6D = 192

Joints are ordered: left hand (16 MANO joints) then right hand (16 MANO joints).
Per-hand order: wrist, index1-3, middle1-3, ring1-3, pinky1-3, thumb1-3.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from ..core.mano_constants import (
    DEGENERATE_FRAME_THRESHOLD,
    HAND_FORMAT_DIM,
    MANO_JOINT_NAMES_BIMANUAL,
    MANO_REST_WRIST_LEFT,
    MANO_REST_WRIST_RIGHT,
    NUM_JOINTS_BIMANUAL,
    NUM_JOINTS_PER_HAND,
)
from ..utils.humanml_utils import (
    cont6d_to_matrix_np,
    matrix_to_cont6d_np,
)
from ..utils.quaternion import (
    axis_angle_to_quaternion,
    rotation_matrix_to_axis_angle,
)
from .base_converter import BaseMotionConverter

logger = logging.getLogger(__name__)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to rotation matrices. Input: (..., 4) wxyz."""
    from ..utils.humanml_utils import quaternion_to_matrix_np

    return quaternion_to_matrix_np(q)


def _build_reference_frame(left_wrist_pos: np.ndarray, right_wrist_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a canonical reference frame from two wrist positions.

    The frame origin is the midpoint of the wrists. The X axis points from
    left to right wrist. Y is world up. Z is the cross product (forward).

    Degenerate case: when both wrists are nearly vertically stacked,
    cross(x_axis, world_up) approaches zero. We detect this via the
    z_norm magnitude and copy the previous valid frame rather than
    producing a near-random axis, preventing silent data corruption.

    Args:
        left_wrist_pos: (T, 3) left wrist world positions
        right_wrist_pos: (T, 3) right wrist world positions

    Returns:
        origins: (T, 3) frame origins
        rotations: (T, 3, 3) rotation matrices (columns = axes in world frame)
    """
    T = left_wrist_pos.shape[0]
    origins = (left_wrist_pos + right_wrist_pos) / 2.0

    x_axis = right_wrist_pos - left_wrist_pos
    x_norm = np.linalg.norm(x_axis, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-8, None)
    x_axis = x_axis / x_norm

    y_up = np.zeros((T, 3), dtype=np.float32)
    y_up[:, 1] = 1.0

    z_axis = np.cross(x_axis, y_up)
    z_norm = np.linalg.norm(z_axis, axis=-1, keepdims=True)  # (T, 1)

    degenerate = z_norm[:, 0] < DEGENERATE_FRAME_THRESHOLD  # (T,)
    if degenerate.any():
        n_degen = int(degenerate.sum())
        logger.warning(
            "%d/%d frames have nearly vertical wrist-to-wrist axis "
            "(|z_norm| < %.2f). Copying nearest valid frame to avoid "
            "silent data corruption.",
            n_degen,
            T,
            DEGENERATE_FRAME_THRESHOLD,
        )

    z_axis_safe = z_axis / np.clip(z_norm, 1e-8, None)
    y_axis_safe = np.cross(z_axis_safe, x_axis)

    rotations = np.stack([x_axis, y_axis_safe, z_axis_safe], axis=-1)  # (T, 3, 3)

    # Fix degenerate frames by copying the nearest non-degenerate neighbor.
    if degenerate.any():
        valid = ~degenerate
        if not valid.any():
            # Entire sequence is degenerate — use identity frame.
            rotations[:] = np.eye(3, dtype=np.float32)
        else:
            valid_indices = np.where(valid)[0]
            for t in np.where(degenerate)[0]:
                nearest = valid_indices[np.argmin(np.abs(valid_indices - t))]
                rotations[t] = rotations[nearest]

    return origins, rotations


def _load_mano_model(model_path: str, is_right: bool) -> dict:
    """Load MANO model from numpy pkl (no chumpy/smplx dependency)."""
    import pickle
    from pathlib import Path as _Path

    side = "RIGHT" if is_right else "LEFT"
    numpy_pkl = _Path(model_path) / "mano" / f"MANO_{side}_numpy.pkl"
    orig_pkl = _Path(model_path) / "mano" / f"MANO_{side}.pkl"

    if numpy_pkl.exists():
        with open(numpy_pkl, "rb") as f:
            return pickle.load(f, encoding="latin1")

    # Convert on the fly: patch numpy aliases for chumpy compatibility
    _aliases = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
    }
    for name, val in _aliases.items():
        if not hasattr(np, name):
            setattr(np, name, val)

    with open(orig_pkl, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    out = {}
    for k, v in raw.items():
        try:
            out[k] = np.array(v)
        except Exception:
            out[k] = v
    with open(numpy_pkl, "wb") as f:
        pickle.dump(out, f)
    return out


def _run_mano_fk(
    hand_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray | None,
    is_right: bool,
    model_path: str,
) -> np.ndarray:
    """Run MANO forward kinematics to get joint positions (no smplx/chumpy).

    Args:
        hand_pose: (T, 45) axis-angle for 15 finger joints
        global_orient: (T, 3) wrist global orientation axis-angle
        transl: (T, 3) wrist translation
        betas: (T, 10) or (10,) shape params, or None
        is_right: True for right hand
        model_path: path to body model directory (containing mano/ subdirectory)

    Returns:
        Joint positions (T, 16, 3)
    """
    from scipy.spatial.transform import Rotation as _Rot

    model = _load_mano_model(model_path, is_right)

    T = hand_pose.shape[0]
    v_template = np.array(model["v_template"], dtype=np.float32)  # (778, 3)
    shapedirs = np.array(model.get("shapedirs", np.zeros((778, 3, 10))), dtype=np.float32)
    parents_m = model["kintree_table"][0].astype(int)  # (16,)

    # J_regressor may be sparse
    jr = model["J_regressor"]
    if jr.ndim == 0:
        jr = jr.item()
    if hasattr(jr, "toarray"):
        jr = jr.toarray()
    J_reg = np.array(jr, dtype=np.float32)  # (16, 778)

    # Shape blend
    if betas is not None:
        b = np.array(betas, dtype=np.float32)
        if b.ndim == 1:
            b = np.tile(b[None], (T, 1))
        nb = min(b.shape[1], shapedirs.shape[2])
        blend = np.einsum("ti,vji->tvj", b[:T, :nb], shapedirs[:, :, :nb])
        v_shaped = v_template[None] + blend  # (T, 778, 3)
    else:
        v_shaped = np.tile(v_template[None], (T, 1, 1))

    # T-pose joints from shaped vertices
    J_tpose = np.einsum("jv,tvd->tjd", J_reg, v_shaped)  # (T, 16, 3)

    # Build per-frame local rotation matrices
    full_aa = np.concatenate([global_orient, hand_pose], axis=1)  # (T, 48)
    local_R = _Rot.from_rotvec(full_aa.reshape(-1, 3)).as_matrix().reshape(T, 16, 3, 3)

    # FK: accumulate global rotations and joint positions
    global_R = np.zeros((T, 16, 3, 3))
    joint_world = np.zeros((T, 16, 3))
    global_R[:, 0] = local_R[:, 0]
    joint_world[:, 0] = np.einsum("tij,tj->ti", global_R[:, 0], J_tpose[:, 0])

    for j in range(1, 16):
        p = int(parents_m[j])
        if p >= 16:  # root (kintree stores large sentinel for root parent)
            global_R[:, j] = local_R[:, j]
            joint_world[:, j] = np.einsum("tij,tj->ti", global_R[:, j], J_tpose[:, j])
        else:
            global_R[:, j] = global_R[:, p] @ local_R[:, j]
            offset = J_tpose[:, j] - J_tpose[:, p]
            joint_world[:, j] = joint_world[:, p] + np.einsum("tij,tj->ti", global_R[:, p], offset)

    # Add world translation
    joint_world += transl[:, None, :]
    return joint_world.astype(np.float32)


class MANOToHandRepConverter(BaseMotionConverter):
    """Convert bimanual MANO parameters to 393-dim hand representation.

    The converter runs MANO FK for each hand, then canonicalizes joint
    positions/velocities/rotations relative to a reference frame defined
    by the wrist midpoint and wrist-to-wrist direction.

    Input dict keys:
        left_hand_pose: (T, 45) axis-angle for left hand joints
        left_global_orient: (T, 3) left wrist orientation
        left_transl: (T, 3) left wrist translation
        right_hand_pose: (T, 45) right hand joints
        right_global_orient: (T, 3) right wrist orientation
        right_transl: (T, 3) right wrist translation
        betas_left: optional (T, 10) or (10,)
        betas_right: optional (T, 10) or (10,)
    """

    input_format = "mano_bimanual"
    output_format = "format_hand"

    def __init__(self, body_model_path: str | None = None):
        super().__init__()
        self.body_model_path = body_model_path
        if not body_model_path:
            raise ValueError("body_model_path is required for MANO to hand rep conversion")

    def convert(self, data: str | dict[str, Any]) -> np.ndarray:
        """Convert bimanual MANO parameters to 393-dim hand representation.

        Args:
            data: dict with MANO parameters or path to .npz file

        Returns:
            (T, 393) hand motion features
        """
        if isinstance(data, str):
            if not data.endswith(".npz"):
                raise ValueError("MANO data file must be .npz format")
            mano_data = dict(np.load(data, allow_pickle=True))
        elif isinstance(data, dict):
            mano_data = data
        else:
            raise ValueError(f"Expected str or dict, got {type(data)}")

        required = [
            "left_hand_pose",
            "left_global_orient",
            "left_transl",
            "right_hand_pose",
            "right_global_orient",
            "right_transl",
        ]
        missing = [k for k in required if k not in mano_data]
        if missing:
            raise ValueError(f"Missing keys: {missing}")

        def _to_np(x):
            if hasattr(x, "numpy"):  # torch Tensor
                return x.numpy()
            return np.array(x, dtype=np.float32)

        left_hand_pose = _to_np(mano_data["left_hand_pose"])
        left_orient = _to_np(mano_data["left_global_orient"])
        left_transl = _to_np(mano_data["left_transl"])
        right_hand_pose = _to_np(mano_data["right_hand_pose"])
        right_orient = _to_np(mano_data["right_global_orient"])
        right_transl = _to_np(mano_data["right_transl"])

        T = left_hand_pose.shape[0]

        # --- FK: get joint positions ---
        left_positions = _run_mano_fk(
            left_hand_pose,
            left_orient,
            left_transl,
            mano_data.get("betas_left"),
            is_right=False,
            model_path=self.body_model_path,
        )
        right_positions = _run_mano_fk(
            right_hand_pose,
            right_orient,
            right_transl,
            mano_data.get("betas_right"),
            is_right=True,
            model_path=self.body_model_path,
        )

        # Stack: (T, 32, 3)
        all_positions = np.concatenate([left_positions, right_positions], axis=1)

        # --- Build rotation matrices from axis-angle ---
        left_rot_aa = np.zeros((T, NUM_JOINTS_PER_HAND, 3), dtype=np.float32)
        left_rot_aa[:, 0] = left_orient
        left_rot_aa[:, 1:] = left_hand_pose.reshape(T, 15, 3)

        right_rot_aa = np.zeros((T, NUM_JOINTS_PER_HAND, 3), dtype=np.float32)
        right_rot_aa[:, 0] = right_orient
        right_rot_aa[:, 1:] = right_hand_pose.reshape(T, 15, 3)

        all_rot_aa = np.concatenate([left_rot_aa, right_rot_aa], axis=1)  # (T, 32, 3)
        all_rot_quat = axis_angle_to_quaternion(all_rot_aa.reshape(-1, 3)).reshape(T, NUM_JOINTS_BIMANUAL, 4)
        all_rot_mat = _quaternion_to_matrix(all_rot_quat)  # (T, 32, 3, 3)

        # --- Reference frame from wrists ---
        left_wrist_pos = all_positions[:, 0, :]
        right_wrist_pos = all_positions[:, NUM_JOINTS_PER_HAND, :]
        ref_origins, ref_rotations = _build_reference_frame(left_wrist_pos, right_wrist_pos)

        # --- Canonicalize positions ---
        centered = all_positions - ref_origins[:, None, :]
        canon_positions = np.zeros_like(centered)
        for t in range(T):
            R_inv = ref_rotations[t].T  # (3, 3)
            canon_positions[t] = (R_inv @ centered[t].T).T

        # --- Velocities ---
        canon_velocities = np.zeros_like(canon_positions)
        if T > 1:
            canon_velocities[1:] = canon_positions[1:] - canon_positions[:-1]

        # --- Root velocity (3D, in canonical frame) ---
        root_vel_world = np.zeros((T, 3), dtype=np.float32)
        if T > 1:
            root_vel_world[1:] = ref_origins[1:] - ref_origins[:-1]
        root_vel_canon = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            root_vel_canon[t] = ref_rotations[t].T @ root_vel_world[t]

        # --- Root angular velocity (6D) ---
        root_angular_6d = np.zeros((T, 6), dtype=np.float32)
        root_angular_6d[0, 0] = 1.0  # identity col1 x
        root_angular_6d[0, 4] = 1.0  # identity col2 y
        if T > 1:
            for t in range(1, T):
                delta = ref_rotations[t - 1].T @ ref_rotations[t]
                root_angular_6d[t] = matrix_to_cont6d_np(delta[None])[0]

        # --- Canonicalize joint rotations ---
        canon_rot_mat = all_rot_mat.copy()
        for t in range(T):
            R_inv = ref_rotations[t].T
            for j in [0, NUM_JOINTS_PER_HAND]:  # wrist roots
                canon_rot_mat[t, j] = R_inv @ all_rot_mat[t, j]
        canon_rot_6d = matrix_to_cont6d_np(canon_rot_mat)  # (T, 32, 6)

        # --- Pack into 393-dim vector ---
        out = np.zeros((T, HAND_FORMAT_DIM), dtype=np.float32)
        idx = 0
        out[:, idx : idx + 3] = root_vel_canon
        idx += 3
        out[:, idx : idx + 6] = root_angular_6d
        idx += 6
        out[:, idx : idx + NUM_JOINTS_BIMANUAL * 3] = canon_positions.reshape(T, -1)
        idx += NUM_JOINTS_BIMANUAL * 3
        out[:, idx : idx + NUM_JOINTS_BIMANUAL * 3] = canon_velocities.reshape(T, -1)
        idx += NUM_JOINTS_BIMANUAL * 3
        out[:, idx : idx + NUM_JOINTS_BIMANUAL * 6] = canon_rot_6d.reshape(T, -1)
        assert idx + NUM_JOINTS_BIMANUAL * 6 == HAND_FORMAT_DIM

        return out


class HandRepToMANOConverter(BaseMotionConverter):
    """Convert 393-dim hand representation back to MANO parameters.

    Recovers global wrist positions, orientations, and per-joint rotations
    by inverting the canonicalization process.
    """

    input_format = "format_hand"
    output_format = "mano_bimanual"

    def __init__(self, body_model_path: str | None = None):
        super().__init__()
        if body_model_path is not None:
            self._j_rest_left = self._compute_j_rest(body_model_path, is_right=False)
            self._j_rest_right = self._compute_j_rest(body_model_path, is_right=True)
        else:
            self._j_rest_left = MANO_REST_WRIST_LEFT.copy()
            self._j_rest_right = MANO_REST_WRIST_RIGHT.copy()

    @staticmethod
    def _compute_j_rest(model_path: str, is_right: bool) -> np.ndarray:
        model = _load_mano_model(model_path, is_right)
        jr = model["J_regressor"]
        if jr.ndim == 0:
            jr = jr.item()
        if hasattr(jr, "toarray"):
            jr = jr.toarray()
        vt = np.array(model["v_template"], dtype=np.float32)
        return (np.array(jr, dtype=np.float32) @ vt)[0]

    def convert(self, data: str | np.ndarray) -> dict[str, np.ndarray]:
        """Convert 393-dim hand representation to MANO parameters.

        Args:
            data: (T, 393) hand features or path to .npy file

        Returns:
            Dict with keys: left_global_orient, left_hand_pose, left_transl,
                           right_global_orient, right_hand_pose, right_transl
                           (all axis-angle, translations in world frame)
        """
        if isinstance(data, str):
            motion = np.load(data)
        elif isinstance(data, np.ndarray):
            motion = data
        else:
            raise ValueError(f"Expected str or ndarray, got {type(data)}")

        if motion.ndim != 2 or motion.shape[1] != HAND_FORMAT_DIM:
            raise ValueError(f"Expected shape (T, {HAND_FORMAT_DIM}), got {motion.shape}")

        T = motion.shape[0]

        # --- Unpack ---
        idx = 0
        root_vel_canon = motion[:, idx : idx + 3]
        idx += 3
        root_angular_6d = motion[:, idx : idx + 6]
        idx += 6
        canon_positions = motion[:, idx : idx + NUM_JOINTS_BIMANUAL * 3].reshape(T, NUM_JOINTS_BIMANUAL, 3)
        idx += NUM_JOINTS_BIMANUAL * 3
        # skip velocities (derived quantity)
        idx += NUM_JOINTS_BIMANUAL * 3
        canon_rot_6d = motion[:, idx : idx + NUM_JOINTS_BIMANUAL * 6].reshape(T, NUM_JOINTS_BIMANUAL, 6)

        # --- Reconstruct reference frame orientations ---
        canon_rot_mat = cont6d_to_matrix_np(canon_rot_6d)

        ref_rotations = np.zeros((T, 3, 3), dtype=np.float32)
        ref_rotations[0] = np.eye(3)
        if T > 1:
            angular_deltas = cont6d_to_matrix_np(root_angular_6d)
            for t in range(1, T):
                ref_rotations[t] = ref_rotations[t - 1] @ angular_deltas[t]

        # --- Reconstruct root positions ---
        ref_origins = np.zeros((T, 3), dtype=np.float32)
        for t in range(1, T):
            ref_origins[t] = ref_origins[t - 1] + ref_rotations[t] @ root_vel_canon[t]

        # --- Recover world positions ---
        world_positions = np.zeros((T, NUM_JOINTS_BIMANUAL, 3), dtype=np.float32)
        for t in range(T):
            world_positions[t] = (ref_rotations[t] @ canon_positions[t].T).T + ref_origins[t]

        # --- Recover world rotations ---
        world_rot_mat = canon_rot_mat.copy()
        for t in range(T):
            for j in [0, NUM_JOINTS_PER_HAND]:  # wrist roots
                world_rot_mat[t, j] = ref_rotations[t] @ canon_rot_mat[t, j]

        # --- Convert to axis-angle ---
        world_rot_aa = np.zeros((T, NUM_JOINTS_BIMANUAL, 3), dtype=np.float32)
        for t in range(T):
            for j in range(NUM_JOINTS_BIMANUAL):
                world_rot_aa[t, j] = rotation_matrix_to_axis_angle(world_rot_mat[t, j])

        left_orient = world_rot_aa[:, 0, :]
        left_hand_pose = world_rot_aa[:, 1:NUM_JOINTS_PER_HAND, :].reshape(T, -1)

        right_orient = world_rot_aa[:, NUM_JOINTS_PER_HAND, :]
        right_hand_pose = world_rot_aa[:, NUM_JOINTS_PER_HAND + 1 :, :].reshape(T, -1)

        # MANO FK: wrist_pos = R_global @ J_rest[0] + transl
        # => transl = wrist_pos - R_global @ J_rest[0]
        left_wrist_offset = np.einsum("tij,j->ti", world_rot_mat[:, 0], self._j_rest_left)
        left_transl = world_positions[:, 0, :] - left_wrist_offset

        right_wrist_offset = np.einsum("tij,j->ti", world_rot_mat[:, NUM_JOINTS_PER_HAND], self._j_rest_right)
        right_transl = world_positions[:, NUM_JOINTS_PER_HAND, :] - right_wrist_offset

        return {
            "left_global_orient": left_orient,
            "left_hand_pose": left_hand_pose,
            "left_transl": left_transl,
            "right_global_orient": right_orient,
            "right_hand_pose": right_hand_pose,
            "right_transl": right_transl,
        }


class FormatHandToMotionConverter(BaseMotionConverter):
    """Convert 393-dim hand representation to joint positions for visualization.

    Decodes the canonical representation back to world-frame joint positions
    without requiring MANO model files (pure geometric reconstruction).
    """

    input_format = "format_hand"
    output_format = "hand_positions"

    def __init__(self):
        super().__init__()

    def convert(self, data: str | np.ndarray) -> dict[str, np.ndarray]:
        """Convert 393-dim hand features to world joint positions.

        Returns:
            Dict with 'positions' (T, 32, 3) and 'joint_names' tuple.
        """
        if isinstance(data, str):
            motion = np.load(data)
        elif isinstance(data, np.ndarray):
            motion = data
        else:
            raise ValueError(f"Expected str or ndarray, got {type(data)}")

        if motion.ndim != 2 or motion.shape[1] != HAND_FORMAT_DIM:
            raise ValueError(f"Expected shape (T, {HAND_FORMAT_DIM}), got {motion.shape}")

        T = motion.shape[0]
        root_vel_canon = motion[:, 0:3]
        root_angular_6d = motion[:, 3:9]
        canon_positions = motion[:, 9 : 9 + NUM_JOINTS_BIMANUAL * 3].reshape(T, NUM_JOINTS_BIMANUAL, 3)

        angular_deltas = cont6d_to_matrix_np(root_angular_6d)
        ref_rotations = np.zeros((T, 3, 3), dtype=np.float32)
        ref_rotations[0] = np.eye(3)
        for t in range(1, T):
            ref_rotations[t] = ref_rotations[t - 1] @ angular_deltas[t]

        ref_origins = np.zeros((T, 3), dtype=np.float32)
        for t in range(1, T):
            ref_origins[t] = ref_origins[t - 1] + ref_rotations[t] @ root_vel_canon[t]

        world_positions = np.zeros((T, NUM_JOINTS_BIMANUAL, 3), dtype=np.float32)
        for t in range(T):
            world_positions[t] = (ref_rotations[t] @ canon_positions[t].T).T + ref_origins[t]

        return {
            "positions": world_positions,
            "joint_names": MANO_JOINT_NAMES_BIMANUAL,
            "left_indices": list(range(NUM_JOINTS_PER_HAND)),
            "right_indices": list(range(NUM_JOINTS_PER_HAND, NUM_JOINTS_BIMANUAL)),
        }
