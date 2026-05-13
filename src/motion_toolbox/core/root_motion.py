"""Root motion removal utilities.

This module provides functions to remove horizontal and/or vertical root motion
from Motion objects. All functions return NEW Motion objects, preserving the original.
"""

from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .motion import Motion
from .skeleton import Skeleton


def _get_foot_joint_indices(skeleton: Skeleton) -> list[int]:
    """Find foot joint indices using flexible name matching.

    Handles different naming conventions:
    - SMPL: "left_foot", "right_foot", "left_ankle", "right_ankle"
    - R15: "LeftFoot", "RightFoot"

    Args:
        skeleton: Skeleton to search for foot joints

    Returns:
        List of foot joint indices (typically 2)

    Raises:
        ValueError: If no foot joints can be found
    """
    foot_indices = []
    foot_patterns = ["foot", "ankle"]

    for idx, joint_name in enumerate(skeleton.joint_names):
        name_lower = joint_name.lower()
        for pattern in foot_patterns:
            if pattern in name_lower:
                foot_indices.append(idx)
                break

    if len(foot_indices) == 0:
        raise ValueError(
            f"Could not find foot joints in skeleton. "
            f"Joint names: {skeleton.joint_names}. "
            f"Expected joints containing 'foot' or 'ankle'."
        )

    return foot_indices


def _detect_skeleton_type(skeleton: Skeleton) -> str:
    """Detect skeleton type from joint names.

    Args:
        skeleton: Skeleton to analyze

    Returns:
        "smpl", "r15", or "unknown"
    """
    joint_names_lower = [name.lower() for name in skeleton.joint_names]

    # R15 has specific joint names like "LowerTorso", "UpperTorso"
    if any("lowertorso" in name or "uppertorso" in name for name in joint_names_lower):
        return "r15"

    # SMPL has joints like "pelvis", "spine1", "spine2", "spine3"
    if any("pelvis" in name or "spine1" in name for name in joint_names_lower):
        return "smpl"

    return "unknown"


def _get_default_mesh_offset(skeleton: Skeleton) -> float:
    """Get default mesh-to-joint offset based on skeleton type.

    The foot joint (ankle) is not at the bottom of the foot mesh.
    - SMPL: Mesh extends ~0.02-0.03m below the ankle joint
    - R15: Mesh may be positioned differently, needs negative offset

    Args:
        skeleton: Skeleton to get offset for

    Returns:
        Offset in meters (positive = move character up)
    """
    skeleton_type = _detect_skeleton_type(skeleton)

    if skeleton_type == "smpl":
        # SMPL foot mesh extends below ankle joint
        return 0.01  # ~1cm up
    elif skeleton_type == "r15":
        # R15 needs downward adjustment
        return -0.05  # ~5cm down
    else:
        return 0.0  # No adjustment for unknown skeletons


def _get_per_frame_floor_offsets(
    positions: np.ndarray,
    foot_indices: list[int],
    up_idx: int,
) -> np.ndarray:
    """Compute per-frame floor offsets based on lowest foot position.

    Args:
        positions: Joint positions, shape (num_frames, num_joints, 3)
        foot_indices: Indices of foot joints
        up_idx: Index of the up axis (1 for Y-up, 2 for Z-up)

    Returns:
        Per-frame offsets, shape (num_frames,)
    """
    # Get foot heights for each frame
    foot_heights = positions[:, foot_indices, up_idx]  # (num_frames, num_feet)

    # Get minimum foot height per frame
    min_foot_heights = foot_heights.min(axis=1)  # (num_frames,)

    return min_foot_heights


def _smooth_signal(
    signal: np.ndarray,
    method: Literal["moving_avg", "gaussian"] = "moving_avg",
    window: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply smoothing to a 1D signal.

    Args:
        signal: 1D array to smooth
        method: Smoothing method - "moving_avg" or "gaussian"
        window: Window size for moving average (default: 5)
        sigma: Sigma for Gaussian filter (default: 1.0)

    Returns:
        Smoothed signal
    """
    if method == "moving_avg":
        return uniform_filter1d(signal, size=window, mode="nearest")
    elif method == "gaussian":
        return gaussian_filter1d(signal, sigma=sigma, mode="nearest")
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Use 'moving_avg' or 'gaussian'.")


def remove_horizontal_motion(motion: Motion) -> Motion:
    """Remove horizontal (X-Z) root motion, keeping character in place.

    The character will animate at the origin (X=0, Z=0) while vertical (Y)
    motion is preserved.

    Args:
        motion: Input motion

    Returns:
        New Motion object with horizontal motion removed
    """
    # Get the up axis index
    up_idx = 1 if motion.up_axis.lower() == "y" else 2

    # Create new root translation with zeroed horizontal components
    new_root_translation = motion.root_translation.copy()

    if up_idx == 1:  # Y-up: zero X and Z
        new_root_translation[:, 0] = 0.0  # X
        new_root_translation[:, 2] = 0.0  # Z
    else:  # Z-up: zero X and Y
        new_root_translation[:, 0] = 0.0  # X
        new_root_translation[:, 1] = 0.0  # Y

    return Motion(
        joint_rotations=motion.joint_rotations.copy(),
        root_translation=new_root_translation,
        skeleton=motion.skeleton,
        framerate=motion.framerate,
        up_axis=motion.up_axis,
        identity=motion.identity,
        forward_kinematics=motion.forward_kinematics,
        body_model_path=motion.body_model_path,
        floor_alignment="none",  # Don't re-apply floor alignment
    )


def remove_vertical_motion(
    motion: Motion,
    per_frame: bool = False,
    smoothing: Literal["moving_avg", "gaussian"] = "moving_avg",
    window: int = 5,
    sigma: float = 1.0,
    mesh_offset: float | None = None,
) -> Motion:
    """Remove vertical root motion, keeping character at floor level.

    This removes ALL vertical movement from the root translation, resulting in
    a character that stays at a constant height with feet touching the floor.

    Two modes for floor alignment:
    - Global (per_frame=False): Find the lowest foot position across ALL frames
      and set constant Y to keep that foot at floor level.
    - Per-frame (per_frame=True): Align each frame's lowest foot to the floor,
      then smooth to prevent jitter. Useful when character has intentional
      vertical motion that should be grounded frame-by-frame.

    Args:
        motion: Input motion
        per_frame: If True, align each frame to floor then smooth.
                   If False, use global minimum for constant height (default).
        smoothing: Smoothing method for per-frame mode - "moving_avg" (default) or "gaussian"
        window: Window size for moving average smoothing (default: 5)
        sigma: Sigma for Gaussian smoothing (default: 1.0)
        mesh_offset: Offset to account for mesh extending beyond joints (in meters).
                     Positive = move character up, Negative = move down.
                     If None, auto-detects based on skeleton type:
                     - SMPL: +0.025m (mesh below ankle)
                     - R15: -0.02m (needs downward adjustment)

    Returns:
        New Motion object with vertical motion removed (floor-aligned)
    """
    up_idx = 1 if motion.up_axis.lower() == "y" else 2

    # Get mesh offset (auto-detect if not provided)
    if mesh_offset is None:
        mesh_offset = _get_default_mesh_offset(motion.skeleton)

    # Get foot joint indices using flexible matching
    foot_indices = _get_foot_joint_indices(motion.skeleton)

    # Get current positions (uses FK)
    positions = motion.positions

    # Get foot heights for each frame
    foot_heights = positions[:, foot_indices, up_idx]  # (num_frames, num_feet)

    new_root_translation = motion.root_translation.copy()

    if not per_frame:
        # Global floor alignment: find what constant Y value keeps lowest foot at 0
        # For each frame, the foot height = root_Y + offset_from_root_to_foot
        # We want min(foot_height) = 0 across all frames
        # So we set root_Y to a constant that achieves this

        # Find the frame with the lowest foot
        min_foot_heights_per_frame = foot_heights.min(axis=1)  # (num_frames,)
        global_min_frame = np.argmin(min_foot_heights_per_frame)
        global_min_height = min_foot_heights_per_frame[global_min_frame]

        # The offset from root to foot at that frame
        root_to_foot_offset = global_min_height - motion.root_translation[global_min_frame, up_idx]

        # Set constant Y that keeps lowest foot at 0, plus mesh offset
        # foot_height = root_Y + root_to_foot_offset = 0
        # root_Y = -root_to_foot_offset + mesh_offset
        constant_y = -root_to_foot_offset + mesh_offset
        new_root_translation[:, up_idx] = constant_y
    else:
        # Per-frame floor alignment: each frame's lowest foot touches floor
        # foot_height[t] = root_Y[t] + offset_from_root_to_foot[t] = 0
        # root_Y[t] = -offset_from_root_to_foot[t] + mesh_offset

        min_foot_heights_per_frame = foot_heights.min(axis=1)  # (num_frames,)
        root_to_foot_offsets = min_foot_heights_per_frame - motion.root_translation[:, up_idx]

        # New root Y values that would put feet at floor
        target_y = -root_to_foot_offsets + mesh_offset

        # Smooth to prevent vertical jitter
        smoothed_y = _smooth_signal(target_y, method=smoothing, window=window, sigma=sigma)
        new_root_translation[:, up_idx] = smoothed_y

    return Motion(
        joint_rotations=motion.joint_rotations.copy(),
        root_translation=new_root_translation,
        skeleton=motion.skeleton,
        framerate=motion.framerate,
        up_axis=motion.up_axis,
        identity=motion.identity,
        forward_kinematics=motion.forward_kinematics,
        body_model_path=motion.body_model_path,
        floor_alignment="none",  # Don't re-apply floor alignment
    )


def remove_root_motion(
    motion: Motion,
    horizontal: bool = True,
    vertical: bool = True,
    per_frame: bool = False,
    smoothing: Literal["moving_avg", "gaussian"] = "moving_avg",
    window: int = 5,
    sigma: float = 1.0,
    mesh_offset: float | None = None,
) -> Motion:
    """Remove root motion (horizontal and/or vertical).

    This is a convenience function that combines remove_horizontal_motion
    and remove_vertical_motion.

    Args:
        motion: Input motion
        horizontal: If True, remove horizontal (X-Z) motion (default: True)
        vertical: If True, remove vertical motion / floor align (default: True)
        per_frame: If True and vertical=True, align each frame to floor then smooth
        smoothing: Smoothing method for per-frame mode - "moving_avg" (default) or "gaussian"
        window: Window size for moving average smoothing (default: 5)
        sigma: Sigma for Gaussian smoothing (default: 1.0)
        mesh_offset: Offset to account for mesh extending beyond joints.
                     If None, auto-detects based on skeleton type.

    Returns:
        New Motion object with root motion removed
    """
    result = motion

    if vertical:
        result = remove_vertical_motion(
            result,
            per_frame=per_frame,
            smoothing=smoothing,
            window=window,
            sigma=sigma,
            mesh_offset=mesh_offset,
        )

    if horizontal:
        result = remove_horizontal_motion(result)

    return result
