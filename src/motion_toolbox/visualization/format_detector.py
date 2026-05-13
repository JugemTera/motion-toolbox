"""Format detection for motion data.

This module provides automatic detection of motion representation formats
based on data shape, type, and properties.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.motion import Motion
from .constants import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_GOOD,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MODERATE,
    FORMAT_272D_DIM,
    FORMAT_DIMS,
    HUMANML3D_263D_DIM,
    JOINT_COUNTS,
    SMPL_POSES_72D_DIM,
    STMC_205D_DIM,
)


@dataclass
class DetectedFormat:
    """Result of format detection.

    Attributes:
        format_type: Type of format detected (e.g., '263d', '272d', 'motion', etc.)
        confidence: Confidence level (0.0 to 1.0)
        num_frames: Number of frames detected
        num_joints: Number of joints detected (if applicable)
        skeleton_type: Type of skeleton (e.g., 'humanml3d', 'smpl', etc.)
        metadata: Additional format-specific metadata
    """

    format_type: str
    confidence: float
    num_frames: int
    num_joints: int | None = None
    skeleton_type: str | None = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FormatDetector:
    """Detects motion representation formats from data.

    Supports detection of:
    - HumanML3D 263D representation
    - 272D representation
    - 205D (STMC) representation
    - 232D (ALIGN) representation
    - Motion objects
    - Joint positions (T, N, 3)
    - SMPL parameters (dict with 'poses', 'trans', etc.)

    Examples:
        >>> detector = FormatDetector()
        >>>
        >>> # Detect from HumanML3D data
        >>> data = np.random.randn(120, 263)
        >>> result = detector.detect(data)
        >>> print(result.format_type)  # '263d'
        >>>
        >>> # Detect from Motion object
        >>> from motion_toolbox.core import Motion
        >>> result = detector.detect(motion_obj)
        >>> print(result.format_type)  # 'motion'
    """

    # Known joint counts for different skeleton types (imported from constants)
    SKELETON_JOINT_COUNTS = JOINT_COUNTS

    def detect(self, data: Any, format_hint: str | None = None) -> DetectedFormat:
        """Detect the format of motion data.

        Args:
            data: Motion data to detect format from
            format_hint: Optional hint about the format (overrides auto-detection)

        Returns:
            DetectedFormat object with detection results

        Raises:
            ValueError: If format cannot be detected
        """
        if format_hint:
            # If user provides hint, validate and use it
            return self._validate_format_hint(data, format_hint)

        # Try detection in order of specificity
        # 1. Check if it's a Motion object
        if isinstance(data, Motion):
            return self._detect_motion_object(data)

        # 2. Check if it's a dictionary (might be SMPL parameters)
        if isinstance(data, dict):
            return self._detect_dict_format(data)

        # 3. Check if it's a numpy array or tensor
        if isinstance(data, (np.ndarray, list)):
            if isinstance(data, list):
                data = np.array(data)
            return self._detect_array_format(data)

        # Unknown format
        raise ValueError(
            f"Unable to detect format for data of type {type(data)}. "
            f"Supported types: Motion objects, numpy arrays, dicts, lists."
        )

    def _detect_motion_object(self, motion: Motion) -> DetectedFormat:
        """Detect format from Motion object."""
        # Determine skeleton type by matching joint count
        skeleton_type = None
        num_joints = motion.skeleton.num_joints

        # Try exact match first
        for skel_name, joint_count in JOINT_COUNTS.items():
            if num_joints == joint_count:
                skeleton_type = skel_name
                break

        # Special case for MHR which can vary
        if skeleton_type is None and num_joints >= 50:
            skeleton_type = "mhr"

        # Check if body_model_type is set (takes precedence)
        if motion.body_model_type:
            skeleton_type = motion.body_model_type

        return DetectedFormat(
            format_type="motion",
            confidence=CONFIDENCE_CERTAIN,
            num_frames=motion.num_frames,
            num_joints=motion.num_joints,
            skeleton_type=skeleton_type,
            metadata={
                "framerate": motion.framerate,
                "up_axis": motion.up_axis,
                "has_identity": motion.identity is not None,
            },
        )

    def _detect_dict_format(self, data: dict) -> DetectedFormat:
        """Detect format from dictionary (likely SMPL or MHR parameters)."""
        data_keys = set(data.keys())

        # Check for MHR parameter keys FIRST (more specific)
        mhr_keys = {"model_parameters", "identity_coeffs", "face_expr_coeffs"}
        if mhr_keys.issubset(data_keys):
            num_frames = data["model_parameters"].shape[0] if hasattr(data["model_parameters"], "shape") else 1
            return DetectedFormat(
                format_type="mhr",
                confidence=CONFIDENCE_HIGH,
                num_frames=num_frames,
                num_joints=JOINT_COUNTS.get("mhr", 53),
                skeleton_type="mhr",
                metadata={"representation": "mhr_parameters"},
            )

        # Check for SMPL parameter keys
        smpl_keys = {"poses", "pose", "trans", "transl", "betas", "global_orient", "body_pose"}

        if smpl_keys & data_keys:  # If any SMPL keys present
            # Determine number of frames
            num_frames = None
            for key in ["poses", "pose", "trans", "transl"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, (np.ndarray, list)):
                        if isinstance(val, list):
                            val = np.array(val)
                        if val.ndim >= 1:
                            num_frames = val.shape[0]
                            break

            return DetectedFormat(
                format_type="smpl_params",
                confidence=CONFIDENCE_HIGH,
                num_frames=num_frames or 1,
                num_joints=JOINT_COUNTS["smpl"],
                skeleton_type="smpl",
                metadata={"keys": list(data_keys)},
            )

        raise ValueError(
            f"Dictionary format not recognized. Keys: {list(data_keys)}. Expected SMPL parameter keys like: {smpl_keys}"
        )

    def _detect_array_format(self, data: np.ndarray) -> DetectedFormat:
        """Detect format from numpy array."""
        shape = data.shape

        # Must be at least 2D
        if data.ndim < 2:
            raise ValueError(
                f"Array must be at least 2D, got shape {shape}. Expected shapes like (T, 263), (T, 272), or (T, N, 3)"
            )

        num_frames = shape[0]

        # Check for specific dimensional representations
        if data.ndim == 2:
            dim = shape[1]

            # HumanML3D 263D representation
            if dim == HUMANML3D_263D_DIM:
                return DetectedFormat(
                    format_type="263d",
                    confidence=CONFIDENCE_CERTAIN,
                    num_frames=num_frames,
                    num_joints=JOINT_COUNTS["263d"],
                    skeleton_type="humanml3d",
                    metadata={"representation": "humanml3d_263d"},
                )

            # 272D representation
            elif dim == FORMAT_272D_DIM:
                return DetectedFormat(
                    format_type="272d",
                    confidence=CONFIDENCE_CERTAIN,
                    num_frames=num_frames,
                    num_joints=JOINT_COUNTS["272d"],
                    skeleton_type="smpl_compatible",
                    metadata={"representation": "272d"},
                )

            # STMC 205D representation
            elif dim == STMC_205D_DIM:
                return DetectedFormat(
                    format_type="205d",
                    confidence=CONFIDENCE_CERTAIN,
                    num_frames=num_frames,
                    num_joints=JOINT_COUNTS["205d"],
                    skeleton_type="smpl_compatible",
                    metadata={"representation": "stmc_205d"},
                )

            # ALIGN 232D representation (StableMotion)
            elif dim == FORMAT_DIMS["align"]:
                return DetectedFormat(
                    format_type="align",
                    confidence=CONFIDENCE_CERTAIN,
                    num_frames=num_frames,
                    num_joints=JOINT_COUNTS["smpl"],  # ALIGN uses SMPL skeleton
                    skeleton_type="smpl",
                    metadata={"representation": "align_232d"},
                )

            # Bimanual hand 393D representation
            elif dim == FORMAT_DIMS.get("393d", 393):
                return DetectedFormat(
                    format_type="393d",
                    confidence=CONFIDENCE_CERTAIN,
                    num_frames=num_frames,
                    num_joints=JOINT_COUNTS["393d"],
                    skeleton_type="hand",
                    metadata={"representation": "hand_393d"},
                )

            # SMPL pose parameters (72D = 24 joints * 3)
            elif dim == SMPL_POSES_72D_DIM:
                return DetectedFormat(
                    format_type="smpl_poses",
                    confidence=CONFIDENCE_GOOD,
                    num_frames=num_frames,
                    num_joints=JOINT_COUNTS["smpl"],
                    skeleton_type="smpl",
                    metadata={"representation": "smpl_72d_poses"},
                )

            else:
                raise ValueError(
                    f"Unrecognized 2D array shape {shape}. "
                    f"Expected dimensions: {HUMANML3D_263D_DIM} (HumanML3D), {FORMAT_272D_DIM}, "
                    f"393 (hand bimanual), "
                    f"{STMC_205D_DIM} (STMC), or {SMPL_POSES_72D_DIM} (SMPL poses)"
                )

        # Check for joint positions (T, N, 3)
        elif data.ndim == 3 and shape[2] == 3:
            num_joints = shape[1]

            # Try to identify skeleton type from joint count
            skeleton_type = None
            # Lower confidence for joint positions (could be custom skeleton)
            confidence = CONFIDENCE_MODERATE

            for skel_name, joint_count in self.SKELETON_JOINT_COUNTS.items():
                if num_joints == joint_count:
                    skeleton_type = skel_name
                    # Higher confidence when we match a known skeleton
                    confidence = CONFIDENCE_HIGH
                    break

            if skeleton_type is None:
                # Unknown skeleton type but valid joint format
                skeleton_type = f"unknown_{num_joints}joints"

            return DetectedFormat(
                format_type="joint_positions",
                confidence=confidence,
                num_frames=num_frames,
                num_joints=num_joints,
                skeleton_type=skeleton_type,
                metadata={"shape": shape},
            )

        else:
            raise ValueError(
                f"Unrecognized array shape {shape}. "
                f"Expected (T, {HUMANML3D_263D_DIM}), (T, {FORMAT_272D_DIM}), "
                f"(T, {STMC_205D_DIM}), or (T, N, 3) for joint positions"
            )

    def _validate_format_hint(self, data: Any, format_hint: str) -> DetectedFormat:
        """Validate user-provided format hint against data."""
        # First detect normally to get metadata
        try:
            detected = self.detect(data, format_hint=None)
        except ValueError as e:
            # If auto-detection fails, trust the hint but with low confidence
            return DetectedFormat(
                format_type=format_hint,
                confidence=CONFIDENCE_LOW,
                num_frames=len(data) if hasattr(data, "__len__") else 1,
                metadata={"user_hint": True, "detection_failed": str(e)},
            )

        # Check if hint matches detection
        if detected.format_type == format_hint:
            # Perfect match, return with full confidence
            return detected

        # Hint doesn't match, but use hint anyway with moderate confidence
        return DetectedFormat(
            format_type=format_hint,
            confidence=CONFIDENCE_MODERATE - 0.1,  # 0.6
            num_frames=detected.num_frames,
            num_joints=detected.num_joints,
            skeleton_type=detected.skeleton_type,
            metadata={
                **detected.metadata,
                "user_hint": True,
                "auto_detected": detected.format_type,
            },
        )


def detect_format(data: Any, format_hint: str | None = None) -> DetectedFormat:
    """Convenience function to detect motion data format.

    Args:
        data: Motion data to detect
        format_hint: Optional format hint

    Returns:
        DetectedFormat object

    Examples:
        >>> result = detect_format(np.random.randn(120, 263))
        >>> print(result.format_type)  # '263d'
    """
    detector = FormatDetector()
    return detector.detect(data, format_hint)
