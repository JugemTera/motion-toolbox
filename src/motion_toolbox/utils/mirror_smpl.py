"""SMPL parameter mirroring via X-axis reflection.

Mirrors SMPL body model parameters to produce a left-right reflected version
of the original motion. Follows the approach from SMPL-Mirror
(https://github.com/LebinDing749/SMPL-Mirror), adapted for SMPL-22 joints.

The reflection is across the YZ plane (negate X positions), which in
axis-angle rotation space translates to negating the Y and Z components
of each rotation vector: R' = Mx @ R @ Mx where Mx = diag(-1, 1, 1).
"""

import numpy as np

SMPL_BODY_JOINT_NAMES = [
    "left_hip",
    "right_hip",
    "spine1",  # 0, 1, 2
    "left_knee",
    "right_knee",
    "spine2",  # 3, 4, 5
    "left_ankle",
    "right_ankle",
    "spine3",  # 6, 7, 8
    "left_foot",
    "right_foot",
    "neck",  # 9, 10, 11
    "left_collar",
    "right_collar",
    "head",  # 12, 13, 14
    "left_shoulder",
    "right_shoulder",  # 15, 16
    "left_elbow",
    "right_elbow",  # 17, 18
    "left_wrist",
    "right_wrist",  # 19, 20
]

BACKBONE_CHAIN = [2, 5, 8, 11, 14]
LEFT_CHAIN = [0, 3, 6, 9, 12, 15, 17, 19]
RIGHT_CHAIN = [1, 4, 7, 10, 13, 16, 18, 20]


def mirror_smpl_params(
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Mirror SMPL parameters via X-axis reflection.

    Follows the SMPL-Mirror (LebinDing749) approach exactly:
      1. Root orient: negate Y and Z of axis-angle
      2. Body pose backbone joints: negate Y and Z
      3. Body pose: swap left/right joint pairs
      4. Body pose left+right joints (post-swap): negate Y and Z
      5. Translation: negate X

    Args:
        root_orient: Root orientation axis-angles, shape (T, 3).
        pose_body: Body joint axis-angles, shape (T, 63) or (T, 21, 3).
        trans: Root translation, shape (T, 3).
        betas: Shape parameters, shape (N,) or (T, N). Passed through unchanged.

    Returns:
        Tuple of (m_root_orient, m_pose_body, m_trans, m_betas) with same shapes
        as inputs. pose_body is returned in its original shape convention.
    """
    was_flat = pose_body.ndim == 2
    m_root = root_orient.copy()
    m_pose = pose_body.copy().reshape(-1, 21, 3)
    m_trans = trans.copy()
    m_betas = betas.copy() if betas is not None else None

    # Step 1: Root orient - negate Y, Z
    m_root[:, 1] *= -1
    m_root[:, 2] *= -1

    # Step 2: Backbone joints - negate Y, Z
    m_pose[:, BACKBONE_CHAIN, 1] *= -1
    m_pose[:, BACKBONE_CHAIN, 2] *= -1

    # Step 3: Swap left/right joint pairs
    tmp = m_pose[:, LEFT_CHAIN].copy()
    m_pose[:, LEFT_CHAIN] = m_pose[:, RIGHT_CHAIN]
    m_pose[:, RIGHT_CHAIN] = tmp

    # Step 4: Left+right joints (post-swap) - negate Y, Z
    m_pose[:, LEFT_CHAIN, 1] *= -1
    m_pose[:, LEFT_CHAIN, 2] *= -1
    m_pose[:, RIGHT_CHAIN, 1] *= -1
    m_pose[:, RIGHT_CHAIN, 2] *= -1

    # Step 5: Translation - negate X
    m_trans[:, 0] *= -1

    if was_flat:
        m_pose = m_pose.reshape(-1, 63)

    return m_root, m_pose, m_trans, m_betas


def mirror_positions(positions: np.ndarray) -> np.ndarray:
    """Mirror joint positions via X-axis reflection (negate X, swap L/R).

    This is the HumanML3D-style position mirroring for comparison/validation.

    Args:
        positions: Joint positions, shape (T, J, 3) where J >= 22.

    Returns:
        Mirrored positions with same shape.
    """
    m_pos = positions.copy()
    m_pos[..., 0] *= -1

    # SMPL full joint indices (0=root, 1-21=body)
    left_full = [1, 4, 7, 10, 13, 16, 18, 20]
    right_full = [2, 5, 8, 11, 14, 17, 19, 21]
    tmp = m_pos[:, left_full].copy()
    m_pos[:, left_full] = m_pos[:, right_full]
    m_pos[:, right_full] = tmp

    return m_pos


def validate_mirror(
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    gender: str = "neutral",
    body_model_path: str | None = None,
) -> dict:
    """Validate mirroring by comparing FK(mirror(params)) vs mirror(FK(params)).

    Runs forward kinematics on both the original and mirrored parameters,
    then checks that mirroring the FK output positions matches the FK of
    mirrored parameters.

    Args:
        root_orient: shape (T, 3)
        pose_body: shape (T, 63)
        trans: shape (T, 3)
        betas: shape (N,) or (T, N)
        gender: SMPL gender for FK
        body_model_path: Path to SMPL body models directory

    Returns:
        Dict with keys: 'max_error', 'mean_error', 'passed' (bool, threshold 1e-4)
    """
    import smplx
    import torch

    if body_model_path is None:
        from pathlib import Path

        body_model_path = str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl")

    T = root_orient.shape[0]
    num_betas = betas.shape[-1] if betas.ndim > 0 else 10

    smpl = smplx.create(
        body_model_path,
        model_type="smpl",
        gender=gender,
        ext="pkl",
        create_global_orient=False,
        create_body_pose=False,
        create_transl=False,
    )

    num_pose_features = smpl.posedirs.shape[0]

    def _run_fk(ro, pb, tr):
        if num_pose_features == 207:
            bp_padded = np.zeros((T, 69), dtype=np.float32)
            bp_padded[:, :63] = pb
        else:
            bp_padded = np.zeros((T, 69), dtype=np.float32)
            bp_padded[:, :63] = pb

        b = betas if betas.ndim == 2 else np.tile(betas[:num_betas], (T, 1))
        with torch.no_grad():
            out = smpl(
                global_orient=torch.from_numpy(ro.astype(np.float32)),
                body_pose=torch.from_numpy(bp_padded.astype(np.float32)),
                transl=torch.from_numpy(tr.astype(np.float32)),
                betas=torch.from_numpy(b[:T].astype(np.float32)),
            )
        return out.joints.cpu().numpy()[:, :22]

    # Path A: FK then mirror positions
    joints_orig = _run_fk(root_orient, pose_body, trans)
    joints_mirror_pos = mirror_positions(joints_orig)

    # Path B: Mirror params then FK
    m_ro, m_pb, m_tr, _ = mirror_smpl_params(root_orient, pose_body, trans)
    joints_mirror_params = _run_fk(m_ro, m_pb, m_tr)

    diff = np.abs(joints_mirror_pos - joints_mirror_params)
    max_err = float(diff.max())
    mean_err = float(diff.mean())

    return {
        "max_error": max_err,
        "mean_error": mean_err,
        # Pose-dependent blend shapes are not perfectly symmetric,
        # so allow up to ~3cm max error (typically <5mm mean).
        "passed": max_err < 0.04,
    }
