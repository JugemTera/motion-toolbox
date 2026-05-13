"""SMPL to HumanML3D converter.

Converts SMPL parameters to HumanML3D 263D representation.
Following the process_file() logic from HumanML3D repo.
"""

from pathlib import Path

import numpy as np
import torch

from .base_converter import BaseMotionConverter

# HumanML3D skeleton configuration
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],  # Right leg
    [0, 1, 4, 7, 10],  # Left leg
    [0, 3, 6, 9, 12, 15],  # Spine
    [9, 14, 17, 19, 21],  # Right arm
    [9, 13, 16, 18, 20],  # Left arm
]

T2M_RAW_OFFSETS = np.array(
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

# Face joint indices: r_hip, l_hip, sdr_r, sdr_l
FACE_JOINT_IDX = [2, 1, 17, 16]

# Foot indices for contact detection
FID_L = [7, 10]  # Left ankle, left foot
FID_R = [8, 11]  # Right ankle, right foot


def qmul_np(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def qinv_np(q):
    """Quaternion inverse."""
    q_inv = q.copy()
    q_inv[..., 1:] = -q_inv[..., 1:]
    return q_inv


def qrot_np(q, v):
    """Rotate vector by quaternion."""
    # Normalize quaternion
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)

    q_w = q[..., 0:1]
    q_xyz = q[..., 1:]

    t = 2 * np.cross(q_xyz, v, axis=-1)
    result = v + q_w * t + np.cross(q_xyz, t, axis=-1)

    return result


def qbetween_np(v1, v2):
    """Quaternion that rotates v1 to v2."""
    v1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)

    cross = np.cross(v1, v2, axis=-1)
    dot = (v1 * v2).sum(axis=-1, keepdims=True)

    # q = [w, x, y, z]
    w = np.sqrt((1 + dot) * 0.5)
    xyz = cross / (2 * w + 1e-8)

    return np.concatenate([w, xyz], axis=-1)


def quaternion_to_cont6d_np(quaternions):
    """Convert quaternions to 6D continuous representation."""
    q = quaternions
    # Normalize
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Convert to rotation matrix
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - zw)
    m02 = 2 * (xz + yw)
    m10 = 2 * (xy + zw)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - xw)

    # Return first two columns
    return np.stack([m00, m10, m02, m01, m11, m12], axis=-1).reshape(q.shape[:-1] + (6,))


def inverse_kinematics_np(
    positions: np.ndarray,
    raw_offsets: np.ndarray,
    kinematic_chain: list,
    face_joint_idx: list,
) -> np.ndarray:
    """Compute local quaternion rotations from joint positions.

    Args:
        positions: Joint positions, shape (T, J, 3)
        raw_offsets: Raw bone offsets, shape (J, 3)
        kinematic_chain: List of kinematic chains
        face_joint_idx: Face joint indices [r_hip, l_hip, sdr_r, sdr_l]

    Returns:
        Local quaternion rotations, shape (T, J, 4)
    """
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx

    # Compute facing direction
    across1 = positions[:, r_hip] - positions[:, l_hip]
    across2 = positions[:, sdr_r] - positions[:, sdr_l]
    across = across1 + across2
    across = across / (np.linalg.norm(across, axis=-1, keepdims=True) + 1e-8)

    # Forward direction (cross product with up)
    forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward = forward / (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8)

    # Root rotation to face Z+
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_quat = qbetween_np(forward, target)

    # Initialize quaternion params
    quat_params = np.zeros(positions.shape[:-1] + (4,))
    root_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
    quat_params[:, 0] = root_quat

    # Inverse kinematics along chains
    for chain in kinematic_chain:
        R = root_quat
        for j in range(len(chain) - 1):
            u = raw_offsets[chain[j + 1]][np.newaxis, ...].repeat(len(positions), axis=0)
            v = positions[:, chain[j + 1]] - positions[:, chain[j]]
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

            rot_u_v = qbetween_np(u, v)
            R_loc = qmul_np(qinv_np(R), rot_u_v)
            quat_params[:, chain[j + 1], :] = R_loc
            R = qmul_np(R, R_loc)

    return quat_params


def foot_detect(positions: np.ndarray, thres: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """Detect foot contacts based on velocity threshold.

    Args:
        positions: Joint positions, shape (T, J, 3)
        thres: Velocity threshold

    Returns:
        Tuple of (left_foot_contacts, right_foot_contacts), each shape (T-1, 2)
    """
    velfactor = np.array([thres, thres])

    # Left foot
    feet_l = []
    for fid in FID_L:
        vel = positions[1:, fid] - positions[:-1, fid]
        vel_sq = (vel**2).sum(axis=-1)
        feet_l.append((vel_sq < velfactor[len(feet_l)]).astype(np.float32))
    feet_l = np.stack(feet_l, axis=-1)  # (T-1, 2)

    # Right foot
    feet_r = []
    for fid in FID_R:
        vel = positions[1:, fid] - positions[:-1, fid]
        vel_sq = (vel**2).sum(axis=-1)
        feet_r.append((vel_sq < velfactor[len(feet_r)]).astype(np.float32))
    feet_r = np.stack(feet_r, axis=-1)  # (T-1, 2)

    return feet_l, feet_r


class SMPLToHumanML3DConverter(BaseMotionConverter):
    """Convert SMPL parameters to HumanML3D 263D representation.

    Following the process_file() logic from HumanML3D repo.
    """

    input_format = "smpl_params"
    output_format = "humanml3d"

    def __init__(
        self,
        smpl_model_path: str | None = None,
        normalize: bool = False,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        """Initialize converter.

        Args:
            smpl_model_path: Path to SMPL model files
            normalize: Whether to normalize output
            mean: Mean for normalization
            std: Std for normalization
        """
        super().__init__()
        if smpl_model_path is None:
            smpl_model_path = str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl/")
        self.smpl_model_path = smpl_model_path
        self.normalize = normalize
        self.mean = mean
        self.std = std

        self._smpl_model = None

    def _get_smpl_model(self):
        """Get or create SMPL model."""
        if self._smpl_model is None:
            import smplx

            self._smpl_model = smplx.create(
                self.smpl_model_path,
                model_type="smpl",
                gender="neutral",
                ext="pkl",
                batch_size=1,
            )
        return self._smpl_model

    def convert(
        self,
        smpl_params: dict,
        feet_thres: float = 0.002,
    ) -> np.ndarray:
        """Convert SMPL parameters to HumanML3D 263D representation.

        Args:
            smpl_params: Dictionary with SMPL parameters:
                - global_orient: (T, 3) root rotation
                - body_pose: (T, 63) body pose
                - transl: (T, 3) translation
                - betas: (10,) shape parameters (optional)
            feet_thres: Threshold for foot contact detection

        Returns:
            HumanML3D 263D representation, shape (T-1, 263)
        """
        # Get joint positions from SMPL
        positions = self._smpl_forward(smpl_params)

        return self.convert_from_positions(positions, feet_thres)

    def convert_from_positions(
        self,
        positions: np.ndarray,
        feet_thres: float = 0.002,
    ) -> np.ndarray:
        """Convert joint positions to HumanML3D 263D representation.

        Args:
            positions: Joint positions, shape (T, 22, 3)
            feet_thres: Threshold for foot contact detection

        Returns:
            HumanML3D 263D representation, shape (T-1, 263)
        """
        positions = positions.copy()

        # Put on floor
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height

        # XZ at origin (first frame)
        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz

        # All initially face Z+
        r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_IDX
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / (np.linalg.norm(across) + 1e-8)

        forward_init = np.cross(np.array([0, 1, 0]), across)
        forward_init = forward_init / (np.linalg.norm(forward_init) + 1e-8)

        target = np.array([0, 0, 1])
        root_quat_init = qbetween_np(forward_init[np.newaxis], target[np.newaxis])
        root_quat_init = np.ones((positions.shape[0], 1, 4)) * root_quat_init

        # Save global positions before rotation
        global_positions = positions.copy()

        # Rotate to face Z+
        positions = qrot_np(root_quat_init.repeat(positions.shape[1], axis=1), positions)

        # Foot contacts
        feet_l, feet_r = foot_detect(positions, feet_thres)

        # Compute quaternions and 6D rotations
        quat_params = inverse_kinematics_np(positions, T2M_RAW_OFFSETS, T2M_KINEMATIC_CHAIN, FACE_JOINT_IDX)
        cont_6d_params = quaternion_to_cont6d_np(quat_params)

        # Root rotation
        r_rot = quat_params[:, 0].copy()

        # Root linear velocity (in local frame)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)

        # Root angular velocity
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

        # Get RIC (rotation-invariant coordinates)
        def get_rifke(pos, r_rot):
            pos = pos.copy()
            pos[..., 0] -= pos[:, 0:1, 0]
            pos[..., 2] -= pos[:, 0:1, 2]
            pos = qrot_np(r_rot[:, None].repeat(pos.shape[1], axis=1), pos)
            return pos

        ric_positions = get_rifke(positions, r_rot)

        # Root height
        root_y = positions[:, 0, 1:2]

        # Root rotation velocity (yaw only)
        r_velocity_y = np.arcsin(r_velocity[:, 2:3])

        # Linear velocity (XZ)
        l_velocity = velocity[:, [0, 2]]

        # Construct root data
        root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)

        # Rotation data (exclude root)
        rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

        # RIC data (exclude root)
        ric_data = ric_positions[:, 1:].reshape(len(ric_positions), -1)

        # Joint velocity
        local_vel = qrot_np(
            r_rot[:-1, None].repeat(global_positions.shape[1], axis=1), global_positions[1:] - global_positions[:-1]
        )
        local_vel = local_vel.reshape(len(local_vel), -1)

        # Concatenate all features
        data = np.concatenate(
            [
                root_data,  # (T-1, 4)
                ric_data[:-1],  # (T-1, 63)
                rot_data[:-1],  # (T-1, 126)
                local_vel,  # (T-1, 66)
                feet_l,
                feet_r,  # (T-1, 2) each
            ],
            axis=-1,
        )

        # Normalize if requested
        if self.normalize and self.mean is not None and self.std is not None:
            data = (data - self.mean) / self.std

        return data

    def _smpl_forward(self, smpl_params: dict) -> np.ndarray:
        """Run SMPL forward pass to get joint positions.

        Args:
            smpl_params: SMPL parameters

        Returns:
            Joint positions, shape (T, 22, 3)
        """

        smpl_model = self._get_smpl_model()

        global_orient = torch.from_numpy(smpl_params["global_orient"]).float()
        body_pose = torch.from_numpy(smpl_params["body_pose"]).float()
        transl = torch.from_numpy(smpl_params["transl"]).float()

        if "betas" in smpl_params and smpl_params["betas"] is not None:
            betas = torch.from_numpy(smpl_params["betas"]).float()
            if betas.ndim == 1:
                betas = betas.unsqueeze(0).repeat(len(global_orient), 1)
        else:
            betas = None

        # Run SMPL in batches
        batch_size = 64
        all_joints = []

        for i in range(0, len(global_orient), batch_size):
            end_i = min(i + batch_size, len(global_orient))

            output = smpl_model(
                global_orient=global_orient[i:end_i],
                body_pose=body_pose[i:end_i],
                transl=transl[i:end_i],
                betas=betas[i:end_i] if betas is not None else None,
            )

            # Get first 22 joints
            joints = output.joints[:, :22].detach().numpy()
            all_joints.append(joints)

        return np.concatenate(all_joints, axis=0)
