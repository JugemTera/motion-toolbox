"""joints2smpl module for fitting SMPL parameters to 3D joint positions.

Ported from the MDM (Motion Diffusion Model) repository:
https://github.com/GuyTevet/motion-diffusion-model
"""

from .smplify import Joints2SMPL, SMPLify3D

__all__ = ["SMPLify3D", "Joints2SMPL"]
