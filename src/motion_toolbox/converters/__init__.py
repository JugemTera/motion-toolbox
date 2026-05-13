"""Motion representation converters for transforming between different motion formats.

This module provides converters for transforming between motion representations.

✅ **Currently Implemented:**
- GVHMR → Motion (fully implemented and tested)
- Motion → GVHMR (conversion for pipeline integration)
- Motion → SMPL (basic implementation for axis-angle export)
- SMPL → R15 (convert SMPL motion to Roblox R15 format)
- HumanML3D ↔ SMPL (bidirectional conversion with SMPLify3D)

For a complete list of converters and their status, use `list_converters()`.
"""

# ALIGN converter
from .align_converter import ALIGNConverter
from .base_converter import BaseMotionConverter

# HumanML3D ↔ SMPL converters
from .humanml3d_to_smpl import HumanML3DToSMPLConverter, recover_from_ric
from .motion_converters import (
    AVAILABLE_CONVERTERS,
    Format272ToMotionConverter,
    GVHMRToMotionConverter,
    HumanML3DTo272Converter,
    HumanML3DToMotionConverter,
    MHRToMotionConverter,
    # Implemented converters
    MotionToGVHMRConverter,
    # Unimplemented converters (raise NotImplementedError)
    MotionToHumanML3DConverter,
    MotionToMHRConverter,
    MotionToSMPLConverter,
    SMPLToMotionConverter,
    SMPLToR15Converter,
    # Utility functions
    get_motion_converter,
    list_converters,
    load_smpl_npz,
)
from .smpl_to_humanml3d import SMPLToHumanML3DConverter

# STMC converter
from .stmc_converter import STMC205Converter

# Export only implemented converters in __all__ to avoid confusion
__all__ = [
    "BaseMotionConverter",
    # ✅ Implemented and ready to use
    "Format272ToMotionConverter",
    "GVHMRToMotionConverter",
    "MotionToGVHMRConverter",
    "MotionToSMPLConverter",
    "MHRToMotionConverter",
    "MotionToMHRConverter",
    "SMPLToR15Converter",
    # HumanML3D ↔ SMPL
    "HumanML3DToSMPLConverter",
    "SMPLToHumanML3DConverter",
    "recover_from_ric",
    # STMC
    "STMC205Converter",
    # ALIGN
    "ALIGNConverter",
    # HumanML3D ↔ 272D conversion
    "HumanML3DTo272Converter",
    # 🔄 Available but raise NotImplementedError (for contributors)
    "MotionToHumanML3DConverter",
    "HumanML3DToMotionConverter",
    "SMPLToMotionConverter",
    # TODO: Implement MotionToSTMCConverter and STMCToMotionConverter https://github.com/nv-tlabs/stmc
    # Utility functions
    "get_motion_converter",
    "list_converters",
    "load_smpl_npz",
    "AVAILABLE_CONVERTERS",
]
