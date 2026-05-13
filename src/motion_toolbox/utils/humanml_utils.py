"""Utilities for HumanML3D-style feature extraction.

Contains minimal numpy-based quaternion helpers, conversion to continuous 6D,
and a compact HumanML skeleton class for IK needed by the SMPL->HumanML3D
feature pipeline. This module avoids external heavy deps and stays self-contained.
"""

from __future__ import annotations

import numpy as np


def qnormalize_np(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) to unit length.

    Args:
        q: Quaternion(s), shape (..., 4) in (w, x, y, z) format

    Returns:
        Normalized quaternion(s) with unit length, shape (..., 4)
    """
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return q / norm


def qinv_np(q: np.ndarray) -> np.ndarray:
    """Compute the inverse (conjugate) of unit quaternion(s).

    Args:
        q: Quaternion(s), shape (..., 4) in (w, x, y, z) format

    Returns:
        Inverted quaternion(s), shape (..., 4)
    """
    qn = qnormalize_np(q)
    mask = np.ones_like(qn)
    mask[..., 1:] *= -1
    return qn * mask


def qmul_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Multiply two quaternions using Hamilton product.

    Args:
        q: First quaternion(s), shape (..., 4) in (w, x, y, z) format
        r: Second quaternion(s), shape (..., 4) in (w, x, y, z) format

    Returns:
        Product quaternion(s) q * r, shape (..., 4)
    """
    qa = q.reshape(-1, 4)
    ra = r.reshape(-1, 4)
    w = ra[:, 0] * qa[:, 0] - ra[:, 1] * qa[:, 1] - ra[:, 2] * qa[:, 2] - ra[:, 3] * qa[:, 3]
    x = ra[:, 0] * qa[:, 1] + ra[:, 1] * qa[:, 0] - ra[:, 2] * qa[:, 3] + ra[:, 3] * qa[:, 2]
    y = ra[:, 0] * qa[:, 2] + ra[:, 1] * qa[:, 3] + ra[:, 2] * qa[:, 0] - ra[:, 3] * qa[:, 1]
    z = ra[:, 0] * qa[:, 3] - ra[:, 1] * qa[:, 2] + ra[:, 2] * qa[:, 1] + ra[:, 3] * qa[:, 0]
    out = np.stack((w, x, y, z), axis=1)
    return out.reshape(q.shape)


def qrot_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector(s) by quaternion(s).

    Args:
        q: Quaternion(s), shape (..., 4) in (w, x, y, z) format
        v: 3D vector(s) to rotate, shape (..., 3)

    Returns:
        Rotated vector(s), shape (..., 3)
    """
    qn = qnormalize_np(q)
    v_as_q = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=v.dtype), v], axis=-1)
    return qmul_np(qmul_np(qn, v_as_q), qinv_np(qn))[..., 1:]


def qbetween_np(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Compute quaternion that rotates from v0 to v1.

    Args:
        v0: Source vector(s), shape (..., 3)
        v1: Target vector(s), shape (..., 3)

    Returns:
        Quaternion(s) representing rotation from v0 to v1, shape (..., 4)
    """
    v = np.cross(v0, v1)
    w = np.sqrt((v0**2).sum(axis=-1, keepdims=True) * (v1**2).sum(axis=-1, keepdims=True)) + (v0 * v1).sum(
        axis=-1, keepdims=True
    )
    q = np.concatenate([w, v], axis=-1)
    return qnormalize_np(q)


def qfix_np(q: np.ndarray) -> np.ndarray:
    """Fix quaternion sequence to ensure temporal continuity.

    Flips quaternions when necessary to avoid discontinuities caused by
    quaternion double-cover (q and -q represent the same rotation).

    Args:
        q: Quaternion sequence, shape (T, J, 4)

    Returns:
        Fixed quaternion sequence with continuous trajectories, shape (T, J, 4)
    """
    result = q.copy()
    dots = (q[1:] * q[:-1]).sum(axis=2)
    mask = dots < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def quaternion_to_matrix_np(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to 3x3 rotation matrices.

    Args:
        quaternions: Quaternion(s), shape (..., 4) in (w, x, y, z) format

    Returns:
        Rotation matrices, shape (..., 3, 3)
    """
    r, i, j, k = np.split(quaternions, 4, axis=-1)
    two_s = 2.0 / np.maximum((quaternions * quaternions).sum(axis=-1, keepdims=True), 1e-8)
    m00 = 1 - two_s * (j * j + k * k)
    m01 = two_s * (i * j - k * r)
    m02 = two_s * (i * k + j * r)
    m10 = two_s * (i * j + k * r)
    m11 = 1 - two_s * (i * i + k * k)
    m12 = two_s * (j * k - i * r)
    m20 = two_s * (i * k - j * r)
    m21 = two_s * (j * k + i * r)
    m22 = 1 - two_s * (i * i + j * j)
    return np.concatenate(
        [
            np.concatenate([m00, m01, m02], axis=-1)[..., None, :],
            np.concatenate([m10, m11, m12], axis=-1)[..., None, :],
            np.concatenate([m20, m21, m22], axis=-1)[..., None, :],
        ],
        axis=-2,
    )


def quaternion_to_cont6d_np(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to continuous 6D rotation representation.

    Args:
        quaternions: Quaternion(s), shape (..., 4) in (w, x, y, z) format

    Returns:
        Continuous 6D representation (first two columns of rotation matrix), shape (..., 6)
    """
    mat = quaternion_to_matrix_np(quaternions)
    return np.concatenate([mat[..., 0], mat[..., 1]], axis=-1)


class HMMLSkeleton:
    """Minimal HumanML skeleton for IK/FK with unit offsets and kinematic chains.

    Provides inverse and forward kinematics for the HumanML3D 22-joint skeleton.
    """

    def __init__(self, raw_offsets: np.ndarray, kinematic_chain: list[list[int]]):
        """Initialize skeleton with bone offsets and kinematic chains.

        Args:
            raw_offsets: Static bone offsets in T-pose, shape (J, 3)
            kinematic_chain: List of joint chains defining parent-child relationships
        """
        self._raw_offset_np = raw_offsets.astype(np.float32)
        self._kinematic_chain = kinematic_chain
        self._parents = [0] * raw_offsets.shape[0]
        self._parents[0] = -1
        for chain in self._kinematic_chain:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def set_offset(self, new_offsets: np.ndarray) -> None:
        """Update bone offsets (for skeleton retargeting).

        Args:
            new_offsets: New bone offsets, shape (J, 3)
        """
        self._raw_offset_np = new_offsets.astype(np.float32)

    def inverse_kinematics_np(self, joints: np.ndarray, face_joint_idx: list[int]) -> np.ndarray:
        """Compute local quaternion rotations from joint positions.

        Args:
            joints: Joint positions, shape (T, J, 3)
            face_joint_idx: Indices [r_hip, l_hip, sdr_r, sdr_l] for facing direction

        Returns:
            Local quaternion rotations per joint, shape (T, J, 4)
        """
        r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.maximum(np.linalg.norm(across, axis=-1, keepdims=True), 1e-8)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward = forward / np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8)
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)
        quat_params = np.zeros(joints.shape[:-1] + (4,), dtype=np.float32)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        for chain in self._kinematic_chain:
            R = root_quat
            for j in range(len(chain) - 1):
                u = self._raw_offset_np[chain[j + 1]][np.newaxis, ...].repeat(len(joints), axis=0)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
                rot_u_v = qbetween_np(u, v)
                R_loc = qmul_np(qinv_np(R), rot_u_v)
                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)
        return quat_params

    def forward_kinematics_np(self, quat_params: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
        """Compute joint positions from local quaternion rotations.

        Args:
            quat_params: Local quaternion rotations per joint, shape (T, J, 4)
            root_pos: Root position trajectory, shape (T, 3)

        Returns:
            Joint positions, shape (T, J, 3)
        """
        T, J, _ = quat_params.shape
        positions = np.zeros((T, J, 3), dtype=np.float32)
        positions[:, 0, :] = root_pos.astype(np.float32)
        # world quats per joint
        world_q = np.zeros((T, J, 4), dtype=np.float32)
        world_q[:, 0, :] = quat_params[:, 0, :]
        # Traverse chains
        for chain in self._kinematic_chain:
            for j in range(1, len(chain)):
                c = chain[j]
                p = chain[j - 1]
                # world quaternion
                world_q[:, c, :] = qmul_np(world_q[:, p, :], quat_params[:, c, :])
                # rotate local offset by parent world rotation
                off = np.broadcast_to(self._raw_offset_np[c], (T, 3))
                delta = qrot_np(world_q[:, p, :], off)
                positions[:, c, :] = positions[:, p, :] + delta
        return positions


# === HumanML3D 22-joint template (unit offsets) ===
T2M_KINEMATIC_CHAIN: list[list[int]] = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

T2M_RAW_OFFSETS: np.ndarray = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ],
    dtype=np.float32,
)


def compute_parents_from_chain(kinematic_chain: list[list[int]], num_joints: int) -> list[int]:
    """Compute parent indices for each joint from kinematic chains.

    Args:
        kinematic_chain: List of joint chains defining parent-child relationships
        num_joints: Total number of joints

    Returns:
        Parent index for each joint (-1 for root)
    """
    parents = [-1] * num_joints
    parents[0] = -1
    for chain in kinematic_chain:
        for i in range(1, len(chain)):
            parents[chain[i]] = chain[i - 1]
    return parents


# HumanML3D face joint indices (r_hip, l_hip, sdr_r, sdr_l)
FACE_JOINT_IDX_T2M: list[int] = [2, 1, 17, 16]


def uniform_skeleton_np(
    positions: np.ndarray, target_offsets: np.ndarray, kinematic_chain: list[list[int]] = T2M_KINEMATIC_CHAIN
) -> np.ndarray:
    """Normalize skeleton to uniform bone lengths via IK->FK retargeting.

    Converts joint positions to use target bone offsets while preserving motion.
    Scales root trajectory by leg-length ratio to maintain global scale.

    Args:
        positions: Joint positions, shape (T, J, 3)
        target_offsets: Target bone offsets in T-pose, shape (J, 3)
        kinematic_chain: Kinematic chain definition

    Returns:
        Retargeted joint positions with uniform bone lengths, shape (T, J, 3)
    """
    T, J, _ = positions.shape

    # Compute leg-length ratio (lower legs indices 5,8)
    def _leg_len(off: np.ndarray) -> float:
        return float(np.abs(off[5]).max() + np.abs(off[8]).max())

    # Source offsets from first frame
    parents = compute_parents_from_chain(kinematic_chain, J)
    src_offsets = np.zeros((J, 3), dtype=np.float32)
    for j in range(J):
        p = parents[j]
        if p >= 0:
            src_offsets[j] = positions[0, j] - positions[0, p]
    scale_rt = 1.0
    src_ll = _leg_len(src_offsets)
    tgt_ll = _leg_len(target_offsets)
    if src_ll > 1e-6 and tgt_ll > 0:
        scale_rt = tgt_ll / src_ll

    # IK to quaternions, FK with target offsets
    skel = HMMLSkeleton(T2M_RAW_OFFSETS.copy(), kinematic_chain)
    quat_params = skel.inverse_kinematics_np(positions.copy(), FACE_JOINT_IDX_T2M)
    skel.set_offset(target_offsets.astype(np.float32))
    root_pos = positions[:, 0, :].copy() * scale_rt
    new_joints = skel.forward_kinematics_np(quat_params, root_pos)
    return new_joints


def expmap_to_quaternion_np(e: np.ndarray) -> np.ndarray:
    """Convert axis-angle (exponential map) to quaternion.

    Args:
        e: Axis-angle rotation vector(s), shape (..., 3)

    Returns:
        Quaternion(s) in (w, x, y, z) format, shape (..., 4)
    """
    orig_shape = e.shape
    e = e.reshape(-1, 3)
    theta = np.linalg.norm(e, axis=1, keepdims=True)
    w = np.cos(0.5 * theta)
    # sinc(x) = sin(pi x) / (pi x); here use np.sinc with theta/pi
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    q = np.concatenate([w, xyz], axis=1)
    return q.reshape(orig_shape[:-1] + (4,))


def axis_angle_to_matrix_np(axis_angles: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotations to rotation matrices.

    Args:
        axis_angles: (..., 3)

    Returns:
        (..., 3, 3) rotation matrices
    """
    q = expmap_to_quaternion_np(axis_angles)
    return quaternion_to_matrix_np(q)


def cont6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """Convert 6D continuous rotation representation to rotation matrix.

    Uses Gram-Schmidt orthogonalization.

    Args:
        d6: (..., 6) continuous 6D rotation representation

    Returns:
        (..., 3, 3) rotation matrices
    """
    # Split into first two columns
    a1, a2 = d6[..., :3], d6[..., 3:]

    # Gram-Schmidt orthogonalization
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2, axis=-1)

    # Stack columns to form rotation matrix
    return np.stack([b1, b2, b3], axis=-2)


def matrix_to_cont6d_np(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D continuous rotation representation.

    Args:
        matrix: (..., 3, 3) rotation matrices

    Returns:
        (..., 6) continuous 6D rotation representation
    """
    # Take first two columns and flatten
    d6 = matrix[..., :2, :].copy()
    return d6.reshape(matrix.shape[:-2] + (6,))


def rot_yaw_np(yaw: np.ndarray) -> np.ndarray:
    """Create Y-axis rotation matrices from yaw angles.

    Args:
        yaw: (...,) yaw angles in radians

    Returns:
        (..., 3, 3) rotation matrices
    """
    orig_shape = yaw.shape
    yaw = yaw.reshape(-1)
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    # Y-axis rotation: [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    mats = np.zeros((len(yaw), 3, 3), dtype=np.float32)
    mats[:, 0, 0] = cs
    mats[:, 0, 2] = sn
    mats[:, 1, 1] = 1.0
    mats[:, 2, 0] = -sn
    mats[:, 2, 2] = cs
    return mats.reshape(orig_shape + (3, 3))


def forward_kinematics_axis_angle_np(
    joint_axis_angles: np.ndarray,
    root_translation: np.ndarray,
    raw_offsets: np.ndarray = T2M_RAW_OFFSETS,
    kinematic_chain: list[list[int]] = T2M_KINEMATIC_CHAIN,
) -> np.ndarray:
    """Compute joint positions from axis-angle rotations via kinematic FK.

    Args:
        joint_axis_angles: (T, J, 3) axis-angle rotations
        root_translation: (T, 3) root positions
        raw_offsets: (J, 3) static offsets in T-pose
        kinematic_chain: list of chains describing parent-child order

    Returns:
        positions: (T, J, 3)
    """
    T, J, _ = joint_axis_angles.shape
    parents = compute_parents_from_chain(kinematic_chain, J)
    rot_mats = axis_angle_to_matrix_np(joint_axis_angles)  # (T, J, 3, 3)

    positions = np.zeros((T, J, 3), dtype=np.float32)
    world_rot = np.zeros((T, J, 3, 3), dtype=np.float32)

    # Root
    positions[:, 0, :] = root_translation
    world_rot[:, 0, :, :] = rot_mats[:, 0, :, :]

    # Others
    for j in range(1, J):
        p = parents[j]
        parent_rot = world_rot[:, p, :, :]
        # pos[j] = pos[parent] + parent_rot @ offset[j]
        off = raw_offsets[j].reshape(1, 3, 1)  # (1,3,1)
        delta = np.matmul(parent_rot, off).reshape(T, 3)
        positions[:, j, :] = positions[:, p, :] + delta
        # rot[j] in world = parent_rot @ local_rot
        world_rot[:, j, :, :] = np.matmul(parent_rot, rot_mats[:, j, :, :])

    return positions
