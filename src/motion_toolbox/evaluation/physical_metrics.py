"""Physical integrity metrics for motion evaluation.

This module provides functions to assess the physical plausibility of motion data,
including foot skating detection, motion smoothness analysis, and collision detection.
"""

from dataclasses import dataclass

import numpy as np

from ..core.motion import Motion


@dataclass
class MetricResult:
    """Container for metric computation results.

    Args:
        per_frame: Metric values for each frame, shape (num_frames,)
        per_joint: Metric values for each joint averaged across frames, shape (num_joints,)
        frame_average: Average metric value per frame, shape (num_frames,)
        sequence_average: Single average value for entire sequence

    Examples:
        >>> result = MetricResult(
        ...     per_frame=np.array([0.1, 0.2, 0.15]),
        ...     per_joint=np.array([0.12, 0.18]),
        ...     frame_average=np.array([0.15, 0.19, 0.135]),
        ...     sequence_average=0.16
        ... )
    """

    per_frame: np.ndarray
    per_joint: np.ndarray
    frame_average: np.ndarray
    sequence_average: float


def calculate_foot_skating(
    motion: Motion,
    foot_joint_names: list[str] | None = None,
    velocity_threshold: float = 0.02,
    contact_height_threshold: float = 0.05,
    displacement_threshold: float = 0.01,  # 1cm like StableMotion
    use_displacement_contact: bool = False,  # Option to use StableMotion-style contact detection
) -> MetricResult:
    """Calculate foot skating metric to detect sliding artifacts.

    Foot skating occurs when feet slide along the ground instead of lifting
    during locomotion. This metric detects sliding by measuring horizontal
    velocity when feet should be in contact with ground.

    Args:
        motion: Motion object to analyze
        foot_joint_names: Names of foot joints to analyze. If None, attempts
                         to find common foot joint names ['left_foot', 'right_foot']
        velocity_threshold: Maximum allowed horizontal velocity (m/s) when foot
                          is considered in contact with ground
        contact_height_threshold: Maximum height (m) above ground to consider
                                foot in contact
        displacement_threshold: Maximum displacement (m) per frame for StableMotion-style
                              contact detection (default: 0.01m = 1cm)
        use_displacement_contact: If True, use displacement-based contact detection
                                like StableMotion. If False, use height-based detection

    Returns:
        MetricResult with skating measurements:
        - per_frame: Skating violations per frame (sum across foot joints)
        - per_joint: Average skating violations per foot joint
        - frame_average: Average skating across feet per frame
        - sequence_average: Overall skating metric for sequence

    Examples:
        >>> import numpy as np
        >>> from motion_toolbox import Motion, Skeleton
        >>>
        >>> # Create simple motion with potential skating
        >>> joint_names = ('root', 'left_foot', 'right_foot')
        >>> skeleton = Skeleton(joint_names, (None, 0, 0),
        ...                    np.array([[0,0,0], [0,-1,0], [0,-1,0]]))
        >>>
        >>> rotations = np.array([[[1,0,0,0], [1,0,0,0], [1,0,0,0]]] * 10)
        >>> translation = np.linspace([0,0,0], [1,0,0], 10)  # Moving forward
        >>> motion = Motion(rotations, translation, skeleton)
        >>>
        >>> # Use height-based contact detection (default)
        >>> skating_result = calculate_foot_skating(motion)
        >>> print(f"Average skating: {skating_result.sequence_average:.3f}")
        >>>
        >>> # Use StableMotion-style displacement-based contact detection
        >>> skating_stable = calculate_foot_skating(motion, use_displacement_contact=True)
        >>> print(f"StableMotion-style skating: {skating_stable.sequence_average:.3f}")
    """
    # Motion class already stores axis-angles and provides position calculation

    # Get joint positions through forward kinematics
    positions = motion.positions  # Shape: (frames, joints, 3)
    num_frames, num_joints = positions.shape[:2]

    # Identify foot joints
    if foot_joint_names is None:
        foot_joint_names = ["left_foot", "right_foot", "left_ankle", "right_ankle"]

    foot_indices = []
    if foot_joint_names:  # Check if not None and not empty
        for foot_name in foot_joint_names:
            try:
                idx = motion.skeleton.get_joint_index(foot_name)
                foot_indices.append(idx)
            except ValueError:
                continue  # Joint not found, skip

    if not foot_indices:
        # If no foot joints found, use last two joints as fallback
        foot_indices = [num_joints - 2, num_joints - 1] if num_joints >= 2 else [0]

    # Get velocities from motion (cached if available)
    velocities = motion.velocities

    # Compute skating metric for each foot joint
    skating_per_frame = np.zeros(num_frames)
    skating_per_joint = np.zeros(len(foot_indices))

    for joint_idx, foot_idx in enumerate(foot_indices):
        foot_positions = positions[:, foot_idx, :]  # Shape: (frames, 3)
        foot_velocities = velocities[:, foot_idx, :]

        # Determine ground contact using specified method
        if use_displacement_contact:
            # StableMotion-style: contact when displacement is small (< 1cm/frame)
            foot_displacements = np.linalg.norm(foot_positions[1:] - foot_positions[:-1], axis=1)
            # Pad with False for first frame (no previous frame to compare)
            in_contact = np.concatenate([[False], foot_displacements < displacement_threshold])
        else:
            # Height-based: foot is close to ground (low Y coordinate)
            foot_heights = foot_positions[:, 1]  # Y coordinate
            in_contact = foot_heights <= contact_height_threshold

        # Calculate horizontal velocity magnitude when in contact
        horizontal_velocities = np.linalg.norm(foot_velocities[:, [0, 2]], axis=1)

        # Skating occurs when foot is in contact but moving horizontally
        skating_violations = in_contact & (horizontal_velocities > velocity_threshold)
        skating_magnitudes = np.where(skating_violations, horizontal_velocities, 0.0)

        # Aggregate results
        skating_per_frame += skating_magnitudes
        skating_per_joint[joint_idx] = np.mean(skating_magnitudes)

    # Calculate frame averages and sequence average
    frame_average = skating_per_frame / len(foot_indices)
    sequence_average = float(np.mean(frame_average))

    return MetricResult(
        per_frame=skating_per_frame,
        per_joint=skating_per_joint,
        frame_average=frame_average,
        sequence_average=sequence_average,
    )


def calculate_jerk(motion: Motion) -> MetricResult:
    """Calculate motion jerk (third derivative of position) for smoothness assessment.

    Jerk measures the rate of change of acceleration, providing insight into
    motion smoothness. High jerk values indicate abrupt changes that may appear
    unnatural or mechanically implausible.

    Args:
        motion: Motion object to analyze

    Returns:
        MetricResult with jerk measurements:
        - per_frame: RMS jerk across all joints per frame
        - per_joint: Average RMS jerk per joint across sequence
        - frame_average: Average jerk across joints per frame (same as per_frame)
        - sequence_average: Overall RMS jerk for entire sequence

    Examples:
        >>> jerk_result = calculate_jerk(motion)
        >>> print(f"Motion smoothness (lower is better): {jerk_result.sequence_average:.3f}")
        >>>
        >>> # Check which joints have highest jerk
        >>> joint_names = motion.skeleton.joint_names
        >>> for i, jerk_val in enumerate(jerk_result.per_joint):
        ...     print(f"{joint_names[i]}: {jerk_val:.3f}")
    """
    # Motion class already provides all needed functionality

    positions = motion.positions  # Shape: (frames, joints, 3)
    num_frames, num_joints = positions.shape[:2]
    dt = 1.0 / motion.framerate

    if num_frames < 4:
        # Need at least 4 frames to compute jerk (3rd derivative)
        return MetricResult(
            per_frame=np.zeros(num_frames),
            per_joint=np.zeros(num_joints),
            frame_average=np.zeros(num_frames),
            sequence_average=0.0,
        )

    # Calculate jerk using finite differences
    # jerk = d³x/dt³ ≈ (x[t+2] - 3*x[t+1] + 3*x[t] - x[t-1]) / dt³
    jerk = np.zeros((num_frames - 3, num_joints, 3))

    for i in range(num_frames - 3):
        jerk[i] = (positions[i + 3] - 3 * positions[i + 2] + 3 * positions[i + 1] - positions[i]) / (dt**3)

    # Calculate RMS jerk for each frame and joint
    jerk_magnitudes = np.linalg.norm(jerk, axis=2)  # Shape: (frames-3, joints)

    # Pad with zeros for first and last frames where jerk can't be computed
    jerk_per_frame = np.zeros(num_frames)
    jerk_per_frame[1:-2] = np.sqrt(np.mean(jerk_magnitudes**2, axis=1))

    # Per-joint average jerk
    jerk_per_joint = np.sqrt(np.mean(jerk_magnitudes**2, axis=0))

    # Frame average (same as per_frame for this metric)
    frame_average = jerk_per_frame.copy()

    # Sequence average
    sequence_average = np.sqrt(np.mean(jerk_magnitudes**2))

    return MetricResult(
        per_frame=jerk_per_frame,
        per_joint=jerk_per_joint,
        frame_average=frame_average,
        sequence_average=sequence_average,
    )


def calculate_ground_penetration(
    motion: Motion, ground_height: float = 0.0, foot_joint_names: list[str] | None = None
) -> MetricResult:
    """Calculate ground penetration violations for physical plausibility.

    Ground penetration occurs when body parts (especially feet) go below
    the ground plane, which is physically impossible. This metric measures
    the extent and frequency of such violations.

    Args:
        motion: Motion object to analyze
        ground_height: Y-coordinate of the ground plane (default: 0.0)
        foot_joint_names: Names of joints to check for ground penetration.
                         If None, checks all joints but prioritizes feet

    Returns:
        MetricResult with penetration measurements:
        - per_frame: Maximum penetration depth across joints per frame
        - per_joint: Average penetration depth per joint across sequence
        - frame_average: Average penetration across joints per frame
        - sequence_average: Overall penetration metric for sequence

    Examples:
        >>> penetration = calculate_ground_penetration(motion, ground_height=0.0)
        >>> if penetration.sequence_average > 0.01:  # 1cm threshold
        ...     print(f"Warning: Ground penetration detected ({penetration.sequence_average:.3f}m)")
        ...
        >>> # Find most problematic frames
        >>> problem_frames = np.where(penetration.per_frame > 0.02)[0]
        >>> print(f"Problematic frames: {problem_frames}")
    """
    # Motion class already provides all needed functionality

    positions = motion.positions  # Shape: (frames, joints, 3)
    num_frames, num_joints = positions.shape[:2]

    # Identify joints to check (prioritize feet if specified)
    if foot_joint_names is not None:
        check_indices = []
        for joint_name in foot_joint_names:
            try:
                idx = motion.skeleton.get_joint_index(joint_name)
                check_indices.append(idx)
            except ValueError:
                continue
        if not check_indices:  # Fallback to all joints
            check_indices = list(range(num_joints))
    else:
        check_indices = list(range(num_joints))

    # Calculate penetration depths (negative Y values below ground)
    penetration_depths = np.maximum(0.0, ground_height - positions[:, check_indices, 1])

    # Per-frame maximum penetration
    penetration_per_frame = np.max(penetration_depths, axis=1)

    # Per-joint average penetration
    penetration_per_joint = np.mean(penetration_depths, axis=0)

    # Frame average penetration
    frame_average = np.mean(penetration_depths, axis=1)

    # Sequence average
    sequence_average = float(np.mean(penetration_depths))

    return MetricResult(
        per_frame=penetration_per_frame,
        per_joint=penetration_per_joint,
        frame_average=frame_average,
        sequence_average=sequence_average,
    )
