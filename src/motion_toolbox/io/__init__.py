"""File I/O for motion data.

Provides a unified :func:`load_motion` entry-point that reads a motion file
from disk, auto-detects its format (or uses an explicit hint), and returns a
:class:`~motion_toolbox.core.motion.Motion` object ready for visualization or
further processing.

Supported file types
--------------------
==========  =====================  ============================
Extension   Detected formats       Converter used
==========  =====================  ============================
``.npz``    SMPL, 272D             SMPLToMotionConverter /
                                   Format272ToMotionConverter
``.npy``    263D, 272D, 205D,      HumanML3DToMotionConverter /
            232D (ALIGN)           Format272ToMotionConverter /
                                   STMC205Converter /
                                   ALIGNConverter
``.pt``     GVHMR                  GVHMRToMotionConverter
==========  =====================  ============================
"""

from .loader import load_motion

__all__ = ["load_motion"]
