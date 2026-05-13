"""Fast analytic IK initialization for SMPL pose parameters.

This module provides a fast, heuristic inverse kinematics solver that converts
3D joint positions to approximate SMPL pose parameters using geometric calculations.
The output serves as a good initialization for optimization-based refinement.
"""

import torch


def qbetween_torch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute quaternion that rotates vector u to vector v.

    Args:
        u: Source vectors, shape (..., 3)
        v: Target vectors, shape (..., 3)

    Returns:
        Quaternions (w, x, y, z), shape (..., 4)
    """
    # Normalize inputs
    u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-8)
    v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

    # Compute axis (cross product) and angle (dot product)
    axis = torch.cross(u, v, dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    # Angle
    dot = torch.sum(u * v, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    # Handle parallel/anti-parallel cases
    parallel = axis_norm < 1e-6

    # For parallel vectors (including same direction)
    w = torch.where(parallel.squeeze(-1), torch.ones_like(dot.squeeze(-1)), torch.sqrt((1 + dot.squeeze(-1)) / 2))

    # Axis for rotation
    axis = axis / (axis_norm + 1e-8)

    # Half angle
    half_angle = torch.acos(torch.clamp(dot, -1.0, 1.0)) / 2
    sin_half = torch.sin(half_angle)

    # Quaternion components
    quat = torch.cat(
        [
            torch.cos(half_angle),  # w
            axis * sin_half,
        ],
        dim=-1,
    )

    # Handle parallel case (return identity quaternion)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device, dtype=u.dtype)
    quat = torch.where(parallel, identity, quat)

    return quat


def qinv_torch(q: torch.Tensor) -> torch.Tensor:
    """Quaternion inverse (conjugate for unit quaternions).

    Args:
        q: Quaternions (w, x, y, z), shape (..., 4)

    Returns:
        Inverse quaternions, shape (..., 4)
    """
    return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)


def qmul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.

    Args:
        q1: First quaternions (w, x, y, z), shape (..., 4)
        q2: Second quaternions (w, x, y, z), shape (..., 4)

    Returns:
        Product quaternions, shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quat_to_axis_angle_torch(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to axis-angle representation.

    Args:
        quat: Quaternions (w, x, y, z), shape (..., 4)

    Returns:
        Axis-angle rotations, shape (..., 3)
    """
    # Normalize quaternion
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

    # Extract components
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]

    # Angle
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

    # Axis
    sin_half = torch.sqrt(1 - w * w)
    axis = xyz / (sin_half + 1e-8)

    # Axis-angle
    axis_angle = axis * angle

    # Handle near-zero rotation
    near_zero = (angle.abs() < 1e-6).squeeze(-1)
    axis_angle = torch.where(near_zero.unsqueeze(-1), torch.zeros_like(axis_angle), axis_angle)

    return axis_angle


# SMPL kinematic chain definition
# Based on SMPL joint hierarchy (24 joints)
# SMPL joint order: 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1, 4:L_Knee, 5:R_Knee, 6:Spine2,
#                   7:L_Ankle, 8:R_Ankle, 9:Spine3, 10:L_Foot, 11:R_Foot, 12:Neck,
#                   13:L_Collar, 14:R_Collar, 15:Head, 16:L_Shoulder, 17:R_Shoulder,
#                   18:L_Elbow, 19:R_Elbow, 20:L_Wrist, 21:R_Wrist, 22:L_Hand, 23:R_Hand
SMPL_KINEMATIC_CHAIN = [
    [0, 3, 6, 9, 12, 15],  # Spine chain: Pelvis -> Spine1 -> Spine2 -> Spine3 -> Neck -> Head
    [0, 1, 4, 7, 10],  # Left leg: Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Foot
    [0, 2, 5, 8, 11],  # Right leg: Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Foot
    [9, 13, 16, 18, 20, 22],  # Left arm: Spine3 -> L_Collar -> L_Shoulder -> L_Elbow -> L_Wrist -> L_Hand
    [9, 14, 17, 19, 21, 23],  # Right arm: Spine3 -> R_Collar -> R_Shoulder -> R_Elbow -> R_Wrist -> R_Hand
]

# Face joint indices for root orientation (from pelvis orientation)
SMPL_FACE_JOINTS = [1, 2, 13, 14]  # L_Hip, R_Hip, L_Collar, R_Collar


def fast_smpl_ik(joint_positions: torch.Tensor, smpl_offsets: torch.Tensor) -> torch.Tensor:
    """Fast analytic inverse kinematics for SMPL.

    Converts 3D joint positions to SMPL pose parameters using geometric calculations
    along kinematic chains. This provides a good initialization for optimization.

    NOTE: This function expects joints in WORLD SPACE (e.g., HumanML3D format).
    It will automatically center and normalize them to match SMPL coordinate system.

    Args:
        joint_positions: Joint positions, shape (B, 24, 3) or (24, 3) in WORLD SPACE
        smpl_offsets: SMPL bone offsets in rest pose, shape (24, 3)

    Returns:
        SMPL pose parameters in axis-angle format, shape (B, 24, 3) or (24, 3)
    """
    device = joint_positions.device
    dtype = joint_positions.dtype

    # Handle single frame
    single_frame = joint_positions.ndim == 2
    if single_frame:
        joint_positions = joint_positions.unsqueeze(0)

    batch_size = joint_positions.shape[0]
    num_joints = 24

    # CRITICAL FIX: Center joints to pelvis (convert from world space to body-relative)
    # This makes the joints compatible with SMPL's T-pose coordinate system
    pelvis_pos = joint_positions[:, 0:1, :].clone()  # (B, 1, 3)
    joint_positions = joint_positions - pelvis_pos  # Center to pelvis

    # Optional: Scale to match SMPL proportions
    # Compute current height (distance from pelvis to head or max Y)
    # For now, skip scaling and just use centered positions

    # Step 1: Compute root orientation from pelvis
    # Use hip joints to determine facing direction
    l_hip_idx, r_hip_idx, l_collar_idx, r_collar_idx = SMPL_FACE_JOINTS

    # Compute facing direction from hips and shoulders
    across1 = joint_positions[:, r_hip_idx] - joint_positions[:, l_hip_idx]
    across2 = joint_positions[:, r_collar_idx] - joint_positions[:, l_collar_idx]
    across = across1 + across2
    across = across / (torch.norm(across, dim=-1, keepdim=True) + 1e-8)

    # Forward direction (cross with up axis)
    up = torch.tensor([0, 1, 0], device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
    forward = torch.cross(up, across, dim=-1)
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-8)

    # Target forward direction (Z+)
    target = torch.tensor([0, 0, 1], device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
    root_quat = qbetween_torch(forward, target)

    # Initialize quaternion parameters
    quat_params = torch.zeros(batch_size, num_joints, 4, device=device, dtype=dtype)
    quat_params[:, :, 0] = 1.0  # Identity quaternions
    quat_params[:, 0] = root_quat  # Root orientation

    # Step 2: Inverse kinematics along chains
    for chain in SMPL_KINEMATIC_CHAIN:
        R = root_quat.clone()

        for j in range(len(chain) - 1):
            parent_idx = chain[j]
            child_idx = chain[j + 1]

            # Bone direction in rest pose (normalized)
            u = smpl_offsets[child_idx].unsqueeze(0).expand(batch_size, -1)
            u_norm = torch.norm(u, dim=-1, keepdim=True) + 1e-8
            u = u / u_norm

            # Bone direction in current pose (normalized)
            v = joint_positions[:, child_idx] - joint_positions[:, parent_idx]
            v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
            v = v / v_norm

            # Debug: Check if bone lengths differ significantly
            # Scale mismatch might indicate coordinate system issues
            length_ratio = v_norm / (u_norm + 1e-8)
            # If ratio is very different from 1.0, scaling might be needed

            # Rotation from rest to current
            rot_u_v = qbetween_torch(u, v)

            # Local rotation (remove parent's global rotation)
            R_loc = qmul_torch(qinv_torch(R), rot_u_v)
            quat_params[:, child_idx] = R_loc

            # Update global rotation for next iteration
            R = qmul_torch(R, R_loc)

    # Step 3: Convert quaternions to axis-angle
    axis_angle = quat_to_axis_angle_torch(quat_params)

    if single_frame:
        axis_angle = axis_angle.squeeze(0)

    return axis_angle


def get_smpl_rest_offsets(smpl_model_dir=None, device="cpu") -> torch.Tensor:
    """Get SMPL bone offsets in rest pose.

    Args:
        smpl_model_dir: Path to SMPL model directory. If None, uses config.SMPL_MODEL_DIR
        device: Device to create tensor on

    Returns:
        Bone offsets, shape (24, 3)
    """
    import smplx
    import torch

    from . import src

    if smpl_model_dir is None:
        smpl_model_dir = src.config.SMPL_MODEL_DIR

    try:
        # Create a single-batch SMPL model for offset extraction
        smpl_single = smplx.create(
            smpl_model_dir,
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=1,
        ).to(device)

        with torch.no_grad():
            # Get rest pose joints (zero pose, zero shape)
            zero_pose = torch.zeros(1, 69, device=device, dtype=torch.float32)
            zero_betas = torch.zeros(1, 10, device=device, dtype=torch.float32)
            zero_orient = torch.zeros(1, 3, device=device, dtype=torch.float32)

            output = smpl_single(
                global_orient=zero_orient,
                body_pose=zero_pose,
                betas=zero_betas,
            )

            joints = output.joints[0]  # Shape: (45, 3) - SMPL has 45 joints total

            # SMPL parent indices for first 24 joints (body skeleton)
            parent_indices = [
                -1,  # 0: Pelvis (root, no parent)
                0,  # 1: L_Hip
                0,  # 2: R_Hip
                0,  # 3: Spine1
                1,  # 4: L_Knee
                2,  # 5: R_Knee
                3,  # 6: Spine2
                4,  # 7: L_Ankle
                5,  # 8: R_Ankle
                6,  # 9: Spine3
                7,  # 10: L_Foot
                8,  # 11: R_Foot
                9,  # 12: Neck
                9,  # 13: L_Collar
                9,  # 14: R_Collar
                12,  # 15: Head
                13,  # 16: L_Shoulder
                14,  # 17: R_Shoulder
                16,  # 18: L_Elbow
                17,  # 19: R_Elbow
                18,  # 20: L_Wrist
                19,  # 21: R_Wrist
                20,  # 22: L_Hand
                21,  # 23: R_Hand
            ]

            offsets = torch.zeros(24, 3, device=device, dtype=torch.float32)
            for i in range(24):
                if parent_indices[i] >= 0:
                    # Compute offset as: child_pos - parent_pos
                    offsets[i] = joints[i] - joints[parent_indices[i]]

            return offsets

    except Exception as e:
        print(f"Warning: Could not extract SMPL offsets ({e}), using fallback approximations")
        # Fallback: Use approximate offsets
        offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 0: Pelvis (root)
                [-0.0632, -0.0749, 0.0],  # 1: L_Hip
                [0.0632, -0.0749, 0.0],  # 2: R_Hip
                [0.0, 0.0691, 0.0],  # 3: Spine1
                [0.0, -0.4089, 0.0],  # 4: L_Knee
                [0.0, -0.4089, 0.0],  # 5: R_Knee
                [0.0, 0.0819, 0.0],  # 6: Spine2
                [0.0, -0.4395, 0.0],  # 7: L_Ankle
                [0.0, -0.4395, 0.0],  # 8: R_Ankle
                [0.0, 0.1255, 0.0],  # 9: Spine3
                [0.0, -0.0871, 0.0],  # 10: L_Foot
                [0.0, -0.0871, 0.0],  # 11: R_Foot
                [0.0, 0.2155, 0.0],  # 12: Neck
                [-0.1106, 0.0, 0.0],  # 13: L_Collar
                [0.1106, 0.0, 0.0],  # 14: R_Collar
                [0.0, 0.2572, 0.0],  # 15: Head
                [-0.1650, 0.0, 0.0],  # 16: L_Shoulder
                [0.1650, 0.0, 0.0],  # 17: R_Shoulder
                [-0.2590, 0.0, 0.0],  # 18: L_Elbow
                [0.2590, 0.0, 0.0],  # 19: R_Elbow
                [-0.2455, 0.0, 0.0],  # 20: L_Wrist
                [0.2455, 0.0, 0.0],  # 21: R_Wrist
                [-0.0837, 0.0, 0.0],  # 22: L_Hand
                [0.0837, 0.0, 0.0],  # 23: R_Hand
            ],
            device=device,
            dtype=torch.float32,
        )

        return offsets
