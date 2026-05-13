"""Constants for motion format detection and visualization.

This module defines all magic numbers and constants used in format detection
and compatibility checking, making them centralized and well-documented.
"""

# =============================================================================
# Format Dimensions
# =============================================================================
# Dimensions for different motion representation formats

FORMAT_DIMS = {
    "263d": 263,  # HumanML3D: joint positions, velocities, and kinematic features
    "272d": 272,  # Extended representation with additional features
    "393d": 393,  # Bimanual hand format: 32 MANO joints (2 × 16)
    "205d": 205,  # STMC (Spatial Temporal Motion Capture) representation
    "232d": 232,  # ALIGN: StableMotion's rotation-invariant representation
    "align": 232,  # ALIGN: StableMotion's rotation-invariant representation (alias)
    "smpl_poses": 72,  # SMPL body model: 24 joints × 3 rotation dimensions (axis-angle)
}

# Legacy constants for backward compatibility (deprecated, use FORMAT_DIMS instead)
HUMANML3D_263D_DIM = FORMAT_DIMS["263d"]
FORMAT_272D_DIM = FORMAT_DIMS["272d"]
STMC_205D_DIM = FORMAT_DIMS["205d"]
SMPL_POSES_72D_DIM = FORMAT_DIMS["smpl_poses"]


# =============================================================================
# Joint Counts by Representation/Skeleton Type
# =============================================================================

JOINT_COUNTS = {
    # Research motion representations
    "humanml3d": 22,
    "263d": 22,
    "272d": 22,
    "393d": 32,  # Bimanual hand: 2 × 16 MANO joints
    "205d": 22,
    "stmc": 22,
    # Hand representations
    "mano": 16,  # Single MANO hand
    "bimanual_mano": 32,  # Both hands
    "hand": 32,  # Alias for bimanual
    # SMPL body model family
    "smpl": 24,
    "smpl_compatible": 22,  # Compatible subset
    "smplh": 52,  # SMPL + hands
    "smplx": 54,  # SMPL + hands + face (10,475 vertices, 54 joints)
    # Other skeleton types
    "h36m": 19,  # Human3.6M dataset
    "r15": 16,  # Roblox R15
    "mhr": 53,  # Meta Human Rig (approximate, can vary)
}


# =============================================================================
# Confidence Levels
# =============================================================================
# These represent how confident the system is about format detection or
# conversion success. Lower confidence indicates potential for:
# - Detection ambiguity (format could be misidentified)
# - Conversion quality loss (fidelity may be reduced)
# - Untested/experimental combinations

CONFIDENCE_LEVELS = {
    "certain": 1.0,  # Perfect confidence: Exact match, no ambiguity
    "high": 0.9,  # Very high confidence: Known format/conversion, well-tested
    "good": 0.8,  # Good confidence: Works reliably but may lose some fidelity
    "moderate": 0.7,  # Moderate confidence: Works but with known limitations
    "low": 0.5,  # Low confidence: Experimental or ambiguous
}

# Legacy constants for backward compatibility (deprecated, use CONFIDENCE_LEVELS instead)
CONFIDENCE_CERTAIN = CONFIDENCE_LEVELS["certain"]
CONFIDENCE_HIGH = CONFIDENCE_LEVELS["high"]
CONFIDENCE_GOOD = CONFIDENCE_LEVELS["good"]
CONFIDENCE_MODERATE = CONFIDENCE_LEVELS["moderate"]
CONFIDENCE_LOW = CONFIDENCE_LEVELS["low"]


# =============================================================================
# Conversion Time Estimates (seconds)
# =============================================================================
# Approximate time for different conversion types

TIME_ESTIMATES = {
    "instant": 0.1,  # Trivial conversions
    "fast": 0.5,  # Fast direct conversions
    "simple": 1.0,  # Simple format conversions
    "moderate": 2.0,  # More complex conversions (e.g., retargeting)
    "smplify": 20.0,  # SMPLify optimization (slow)
    "smplify_chain": 25.0,  # SMPLify + additional conversion
}

# Legacy constants for backward compatibility (deprecated, use TIME_ESTIMATES instead)
TIME_INSTANT = TIME_ESTIMATES["instant"]
TIME_FAST = TIME_ESTIMATES["fast"]
TIME_SIMPLE = TIME_ESTIMATES["simple"]
TIME_MODERATE = TIME_ESTIMATES["moderate"]
TIME_SMPLIFY = TIME_ESTIMATES["smplify"]
TIME_SMPLIFY_CHAIN = TIME_ESTIMATES["smplify_chain"]


# =============================================================================
# Visualization Defaults
# =============================================================================

# Default frames per second for motion visualization
DEFAULT_FPS = 20

# Skeleton comparison side-by-side offset (in meters)
SKELETON_COMPARISON_OFFSET_X = 2.0  # Distance between skeletons in X direction


# =============================================================================
# Kinematic Chains for Skeleton Visualization
# =============================================================================
# Kinematic chains define how joints are connected to form bones in skeleton visualizations.
# Each chain is a list of joint indices that form a connected path.

KINEMATIC_CHAINS = {
    # HumanML3D skeleton (22 joints)
    "humanml3d": [
        [0, 2, 5, 8, 11],  # Right leg
        [0, 1, 4, 7, 10],  # Left leg
        [0, 3, 6, 9, 12, 15],  # Spine
        [9, 14, 17, 19, 21],  # Right arm
        [9, 13, 16, 18, 20],  # Left arm
    ],
    # SMPL skeleton (24 joints - includes hand joints)
    "smpl": [
        [0, 2, 5, 8, 11],  # Right leg: pelvis -> r_hip -> r_knee -> r_ankle -> r_foot
        [0, 1, 4, 7, 10],  # Left leg: pelvis -> l_hip -> l_knee -> l_ankle -> l_foot
        [0, 3, 6, 9, 12, 15],  # Spine: pelvis -> spine1 -> spine2 -> spine3 -> neck -> head
        [9, 14, 17, 19, 21, 23],  # Right arm: spine3 -> r_collar -> r_shoulder -> r_elbow -> r_wrist -> r_hand
        [9, 13, 16, 18, 20, 22],  # Left arm: spine3 -> l_collar -> l_shoulder -> l_elbow -> l_wrist -> l_hand
    ],
    # R15 skeleton (16 joints - Roblox)
    "r15": [
        [0, 1, 2, 3],  # Spine/torso
        [3, 4, 5, 6],  # Left arm
        [3, 7, 8, 9],  # Right arm
        [0, 10, 11, 12],  # Left leg
        [0, 13, 14, 15],  # Right leg
    ],
    # SOMA30 skeleton (30 joints — Kimodo / NVIDIA SOMA compact)
    # Joint indices match Skeleton.from_soma30():
    #   0=Hips, 1=Spine1, 2=Spine2, 3=Chest, 4=Neck1, 5=Neck2, 6=Head
    #   7=Jaw, 8=LeftEye, 9=RightEye
    #   10=LeftShoulder, 11=LeftArm, 12=LeftForeArm, 13=LeftHand
    #   14=LeftHandThumbEnd, 15=LeftHandMiddleEnd
    #   16=RightShoulder, 17=RightArm, 18=RightForeArm, 19=RightHand
    #   20=RightHandThumbEnd, 21=RightHandMiddleEnd
    #   22=LeftLeg, 23=LeftShin, 24=LeftFoot, 25=LeftToeBase
    #   26=RightLeg, 27=RightShin, 28=RightFoot, 29=RightToeBase
    "soma30": [
        [0, 26, 27, 28, 29],  # Right leg: Hips -> RightLeg -> RightShin -> RightFoot -> RightToeBase
        [0, 22, 23, 24, 25],  # Left leg: Hips -> LeftLeg -> LeftShin -> LeftFoot -> LeftToeBase
        [0, 1, 2, 3, 4, 5, 6],  # Spine: Hips -> Spine1 -> Spine2 -> Chest -> Neck1 -> Neck2 -> Head
        [3, 16, 17, 18, 19],  # Right arm: Chest -> RightShoulder -> RightArm -> RightForeArm -> RightHand
        [3, 10, 11, 12, 13],  # Left arm: Chest -> LeftShoulder -> LeftArm -> LeftForeArm -> LeftHand
    ],
    # SOMA77 skeleton (77 joints — full detail with finger/toe chains)
    # Spine and limb chains follow the same pattern as SOMA30.
    # Joint indices match Skeleton.from_soma77():
    #   0=Hips, 1=Spine1, 2=Spine2, 3=Chest, 4=Neck1, 5=Neck2, 6=Head, 7=HeadEnd
    #   8=Jaw, 9=LeftEye, 10=RightEye
    #   11=LeftShoulder, 12=LeftArm, 13=LeftForeArm, 14=LeftHand
    #   15-38: Left hand finger chains (5 fingers × 4/5 joints each)
    #   39=RightShoulder, 40=RightArm, 41=RightForeArm, 42=RightHand
    #   43-66: Right hand finger chains
    #   67=LeftLeg, 68=LeftShin, 69=LeftFoot, 70=LeftToeBase, 71=LeftToeEnd
    #   72=RightLeg, 73=RightShin, 74=RightFoot, 75=RightToeBase, 76=RightToeEnd
    "soma77": [
        [0, 72, 73, 74, 75, 76],  # Right leg
        [0, 67, 68, 69, 70, 71],  # Left leg
        [0, 1, 2, 3, 4, 5, 6, 7],  # Spine + head
        [3, 39, 40, 41, 42],  # Right arm (main chain)
        [3, 11, 12, 13, 14],  # Left arm (main chain)
        # Right hand finger chains
        [42, 43, 44, 45, 46],  # Right thumb
        [42, 47, 48, 49, 50, 51],  # Right index
        [42, 52, 53, 54, 55, 56],  # Right middle
        [42, 57, 58, 59, 60, 61],  # Right ring
        [42, 62, 63, 64, 65, 66],  # Right pinky
        # Left hand finger chains
        [14, 15, 16, 17, 18],  # Left thumb
        [14, 19, 20, 21, 22, 23],  # Left index
        [14, 24, 25, 26, 27, 28],  # Left middle
        [14, 29, 30, 31, 32, 33],  # Left ring
        [14, 34, 35, 36, 37, 38],  # Left pinky
    ],
    # Alias for Kimodo NPZ (uses SOMA30)
    "kimodo_npz": [
        [0, 26, 27, 28, 29],
        [0, 22, 23, 24, 25],
        [0, 1, 2, 3, 4, 5, 6],
        [3, 16, 17, 18, 19],
        [3, 10, 11, 12, 13],
    ],
    # Bimanual MANO hand (32 joints: left 0-15, right 16-31)
    # Per hand: wrist=0, index=1-3, middle=4-6, ring=7-9, pinky=10-12, thumb=13-15
    "hand": [
        # Left hand
        [0, 1, 2, 3],  # Left index
        [0, 4, 5, 6],  # Left middle
        [0, 7, 8, 9],  # Left ring
        [0, 10, 11, 12],  # Left pinky
        [0, 13, 14, 15],  # Left thumb
        # Right hand
        [16, 17, 18, 19],  # Right index
        [16, 20, 21, 22],  # Right middle
        [16, 23, 24, 25],  # Right ring
        [16, 26, 27, 28],  # Right pinky
        [16, 29, 30, 31],  # Right thumb
    ],
    "bimanual_mano": [
        [0, 1, 2, 3],
        [0, 4, 5, 6],
        [0, 7, 8, 9],
        [0, 10, 11, 12],
        [0, 13, 14, 15],
        [16, 17, 18, 19],
        [16, 20, 21, 22],
        [16, 23, 24, 25],
        [16, 26, 27, 28],
        [16, 29, 30, 31],
    ],
    "mano": [
        [0, 1, 2, 3],
        [0, 4, 5, 6],
        [0, 7, 8, 9],
        [0, 10, 11, 12],
        [0, 13, 14, 15],
    ],
}

# Default colors for kinematic chain visualization
DEFAULT_CHAIN_COLORS = [
    "#ff6b6b",  # Right leg - red
    "#4ecdc4",  # Left leg - cyan
    "#45b7d1",  # Spine - blue
    "#f9ca24",  # Right arm - yellow
    "#6c5ce7",  # Left arm - purple
]

HAND_CHAIN_COLORS = [
    # Left hand
    "#ff6b6b",  # Left index - red
    "#4ecdc4",  # Left middle - teal
    "#45b7d1",  # Left ring - blue
    "#f9ca24",  # Left pinky - yellow
    "#6c5ce7",  # Left thumb - purple
    # Right hand
    "#e74c3c",  # Right index - dark red
    "#1abc9c",  # Right middle - dark teal
    "#2980b9",  # Right ring - dark blue
    "#f39c12",  # Right pinky - orange
    "#8e44ad",  # Right thumb - dark purple
]


# =============================================================================
# Foot Joint Indices
# =============================================================================
# Indices of foot joints for different skeleton types, used for ground alignment

FOOT_INDICES = {
    "humanml3d": [10, 11],  # left_foot, right_foot
    "smpl": [10, 11],  # left_foot, right_foot (same as HumanML3D)
    "272d": [10, 11],  # Uses HumanML3D joint layout
}


# =============================================================================
# SMPL Vertex Indices for Orientation Detection
# =============================================================================
# These vertex indices are used to detect if an SMPL mesh is upside-down
# by comparing head vs feet positions. Indices are for the neutral SMPL model.

SMPL_VERTEX_INDICES = {
    "head": 411,  # Vertex on top of head
    "left_foot": 3392,  # Vertex on left foot
    "right_foot": 6730,  # Vertex on right foot
}


# =============================================================================
# ALIGN Format Constants
# =============================================================================

ALIGN_DEFAULT_FPS = 20.0  # Default framerate for ALIGN format (from StableMotion)


# =============================================================================
# MHR (Meta Human Rig) Specific Constants
# =============================================================================

MHR_FACE_EXPR_DIMS = 72  # Number of face expression coefficients for MHR
