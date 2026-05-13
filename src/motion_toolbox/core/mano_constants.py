"""Constants for the bimanual MANO hand representation.

Centralizes all magic numbers and structural constants for the 393-dimensional
bimanual hand motion format and the MANO skeleton topology, analogous to body
skeleton constants in visualization/constants.py.

The format_hand (393D) layout:
  [0:3]     Root velocity 3D (midpoint of wrists in canonical frame)
  [3:9]     Root angular velocity 6D (orientation delta of hand system)
  [9:105]   Joint positions: 32 joints × 3 = 96 (relative to root, normalized)
  [105:201] Joint velocities: 32 joints × 3 = 96
  [201:393] Joint rotations: 32 joints × 6D = 192

Joint ordering: left hand (16 joints) then right hand (16 joints).
Per-hand order: wrist, index1-3, middle1-3, ring1-3, pinky1-3, thumb1-3.
"""

import numpy as np

# =============================================================================
# Joint Counts and Format Dimensions
# =============================================================================

NUM_JOINTS_PER_HAND: int = 16
NUM_JOINTS_BIMANUAL: int = NUM_JOINTS_PER_HAND * 2  # 32

# 3 (root vel) + 6 (root ang vel) + 32*3 (pos) + 32*3 (vel) + 32*6 (rot) = 393
HAND_FORMAT_DIM: int = 3 + 6 + NUM_JOINTS_BIMANUAL * 3 + NUM_JOINTS_BIMANUAL * 3 + NUM_JOINTS_BIMANUAL * 6
assert HAND_FORMAT_DIM == 393, f"Unexpected HAND_FORMAT_DIM: {HAND_FORMAT_DIM}"


# =============================================================================
# MANO Joint Names
# =============================================================================

MANO_JOINT_NAMES_LEFT: tuple[str, ...] = (
    "left_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
)

MANO_JOINT_NAMES_RIGHT: tuple[str, ...] = (
    "right_wrist",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
)

MANO_JOINT_NAMES_BIMANUAL: tuple[str, ...] = MANO_JOINT_NAMES_LEFT + MANO_JOINT_NAMES_RIGHT

assert len(MANO_JOINT_NAMES_LEFT) == NUM_JOINTS_PER_HAND
assert len(MANO_JOINT_NAMES_RIGHT) == NUM_JOINTS_PER_HAND
assert len(MANO_JOINT_NAMES_BIMANUAL) == NUM_JOINTS_BIMANUAL


# =============================================================================
# MANO Kinematic Topology (Parent Indices)
# =============================================================================
# Per-hand parent chain. Index 0 is the wrist (root, parent=None).
# Finger chains: wrist → proximal → middle → distal phalanx.
#   index:  joints 1-3   (parent chain: 0→1→2→3)
#   middle: joints 4-6   (parent chain: 0→4→5→6)
#   ring:   joints 7-9   (parent chain: 0→7→8→9)
#   pinky:  joints 10-12 (parent chain: 0→10→11→12)
#   thumb:  joints 13-15 (parent chain: 0→13→14→15)

MANO_PARENT_INDICES_SINGLE: tuple[int | None, ...] = (
    None,  # 0  wrist (root)
    0,
    1,
    2,  # 1-3  index
    0,
    4,
    5,  # 4-6  middle
    0,
    7,
    8,  # 7-9  ring
    0,
    10,
    11,  # 10-12 pinky
    0,
    13,
    14,  # 13-15 thumb
)

# Bimanual FK topology: both hands are kinematically independent (two roots).
# The right wrist (joint 16) has parent=None — this is correct for FK.
# NOTE: Skeleton.from_bimanual_mano() uses a *different* topology for
# visualization purposes, parenting the right wrist to the left wrist (index 0)
# so the skeleton graph is connected. Do NOT use MANO_PARENT_INDICES_BIMANUAL
# for visualization; use Skeleton.from_bimanual_mano() instead.
MANO_PARENT_INDICES_BIMANUAL: tuple[int | None, ...] = MANO_PARENT_INDICES_SINGLE + tuple(
    None if p is None else p + NUM_JOINTS_PER_HAND for p in MANO_PARENT_INDICES_SINGLE
)

assert len(MANO_PARENT_INDICES_SINGLE) == NUM_JOINTS_PER_HAND
assert len(MANO_PARENT_INDICES_BIMANUAL) == NUM_JOINTS_BIMANUAL


# =============================================================================
# MANO T-Pose Joint Offsets (approximate anatomical rest pose)
# =============================================================================
# Offsets are in meters, defined as local translation from parent joint.
# These are approximate values derived from average hand anatomy and used
# when a real MANO model file is not available (e.g. for visualization).
#
# Left hand: lateral spread is positive X (fingers splay to the left).
# Right hand: lateral spread is negative X (mirror of left).

MANO_T_POSE_OFFSETS_LEFT: np.ndarray = np.array(
    [
        [0.000, 0.000, 0.000],  # 0  wrist (root, no offset)
        [0.010, -0.030, 0.020],  # 1  index1
        [0.000, -0.025, 0.000],  # 2  index2
        [0.000, -0.020, 0.000],  # 3  index3
        [0.005, -0.035, 0.005],  # 4  middle1
        [0.000, -0.030, 0.000],  # 5  middle2
        [0.000, -0.025, 0.000],  # 6  middle3
        [0.005, -0.030, -0.005],  # 7  ring1
        [0.000, -0.025, 0.000],  # 8  ring2
        [0.000, -0.020, 0.000],  # 9  ring3
        [0.010, -0.025, -0.015],  # 10 pinky1
        [0.000, -0.020, 0.000],  # 11 pinky2
        [0.000, -0.015, 0.000],  # 12 pinky3
        [0.020, -0.020, 0.010],  # 13 thumb1
        [0.015, -0.020, 0.000],  # 14 thumb2
        [0.000, -0.015, 0.000],  # 15 thumb3
    ],
    dtype=np.float32,
)

# Right hand is the mirror of left along the X axis.
MANO_T_POSE_OFFSETS_RIGHT: np.ndarray = MANO_T_POSE_OFFSETS_LEFT.copy()
MANO_T_POSE_OFFSETS_RIGHT[:, 0] *= -1.0  # flip lateral component

assert MANO_T_POSE_OFFSETS_LEFT.shape == (NUM_JOINTS_PER_HAND, 3)
assert MANO_T_POSE_OFFSETS_RIGHT.shape == (NUM_JOINTS_PER_HAND, 3)


# =============================================================================
# Bimanual Skeleton: Right Wrist Root Offset
# =============================================================================
# When building a connected bimanual skeleton (Skeleton.from_bimanual_mano),
# the right wrist is parented to the left wrist. This offset represents a
# typical wrist-to-wrist distance for an adult in a natural manipulation pose.
BIMANUAL_WRIST_SEPARATION_X: float = 0.3  # metres, right wrist offset from left


# =============================================================================
# MANO Rest-Pose Wrist Positions (from the MANO model T-pose)
# =============================================================================
# These are the J_regressor-derived wrist joint positions in the MANO model's
# canonical T-pose (zero shape, zero pose). Used in HandRepToMANOConverter to
# recover world translations from canonical-frame positions.
# Source: MANO_LEFT.pkl / MANO_RIGHT.pkl J_regressor applied to v_template.
MANO_REST_WRIST_LEFT: np.ndarray = np.array([-0.09566991, 0.00638343, 0.0061863], dtype=np.float32)
MANO_REST_WRIST_RIGHT: np.ndarray = np.array([0.09566991, 0.00638343, 0.0061863], dtype=np.float32)


# =============================================================================
# Numerical Thresholds
# =============================================================================
# Minimum sine of the angle between wrist-to-wrist vector and world-up axis.
# Below this threshold the reference frame is considered degenerate (wrists
# nearly aligned vertically) and a fallback orientation is used.
DEGENERATE_FRAME_THRESHOLD: float = 0.05  # sin(~3°)
