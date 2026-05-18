"""Motion Toolbox - Advanced tools for motion analysis and processing."""

# Core classes
# Motion converters
from .converters import (
    BaseMotionConverter,
    GVHMRToMotionConverter,
    HumanML3DToMotionConverter,
    MotionToHumanML3DConverter,
    MotionToSMPLConverter,
    SMPLToMotionConverter,
    get_motion_converter,
    list_converters,
    load_smpl_npz,
)
from .core.motion import Motion
from .core.skeleton import Skeleton

# Datasets / dataloaders
from .data import (
    Format272Dataset,
    HumanML3DDataset,
    MotionFolderDataset,
    motion_collate,
)

# IO — unified file loader
from .io import load_motion

# Utility functions
from .utils.quaternion import (
    angle_between_vectors,
    axis_angle_to_matrix,  # Alias for axis_angle_to_rotation_matrix
    axis_angle_to_quaternion,
    axis_angle_to_rotation_matrix,
    matrix_to_axis_angle,  # Alias for rotation_matrix_to_axis_angle
    normalize_quaternion,
    q_inv,
    q_mul,
    q_rotate,
    q_slerp,
    # Axis-angle conversions
    quaternion_to_axis_angle,
    rotation_matrix_to_axis_angle,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Skeleton",
    "Motion",
    # Quaternion utilities
    "q_mul",
    "q_inv",
    "q_rotate",
    "q_slerp",
    "normalize_quaternion",
    # Axis-angle utilities
    "quaternion_to_axis_angle",
    "axis_angle_to_quaternion",
    "axis_angle_to_rotation_matrix",
    "rotation_matrix_to_axis_angle",
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "angle_between_vectors",
    # Motion converters
    "BaseMotionConverter",
    "MotionToHumanML3DConverter",
    "HumanML3DToMotionConverter",
    "SMPLToMotionConverter",
    "load_smpl_npz",
    "MotionToSMPLConverter",
    "GVHMRToMotionConverter",
    "get_motion_converter",
    "list_converters",
    # IO
    "load_motion",
    # Data
    "MotionFolderDataset",
    "Format272Dataset",
    "HumanML3DDataset",
    "motion_collate",
]


# Pre-built skeletons for convenience
def create_smpl_skeleton():
    """Create SMPL skeleton (24 joints)."""
    return Skeleton.from_smpl()


def create_smplh_skeleton():
    """Create SMPL+H skeleton with hands (52 joints)."""
    return Skeleton.from_smplh()


def create_smplx_skeleton():
    """Create SMPL-X skeleton with hands and face (55 joints)."""
    return Skeleton.from_smplx()


def create_humanml3d_skeleton():
    """Create HumanML3D compatible skeleton (22 joints)."""
    return Skeleton.from_humanml3d()


def create_h36m_skeleton():
    """Create Human3.6M skeleton (32 joints)."""
    return Skeleton.from_h36m()


# Add convenience functions to __all__
__all__.extend(
    [
        "create_smpl_skeleton",
        "create_smplh_skeleton",
        "create_smplx_skeleton",
        "create_humanml3d_skeleton",
        "create_h36m_skeleton",
    ]
)
