"""Compatibility matrix for visualization modes and motion formats.

This module defines which formats can be visualized with which modes
and body models, and what conversions are required.
"""

from dataclasses import dataclass
from enum import Enum

from .constants import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_GOOD,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MODERATE,
    TIME_FAST,
    TIME_INSTANT,
    TIME_MODERATE,
    TIME_SIMPLE,
    TIME_SMPLIFY,
    TIME_SMPLIFY_CHAIN,
)
from .format_detector import DetectedFormat


class VisualizationMode(Enum):
    """Available visualization modes."""

    SKELETON = "skeleton"
    MESH = "mesh"
    AUTO = "auto"


class BodyModel(Enum):
    """Available body models for mesh visualization."""

    SMPL = "smpl"
    R15 = "r15"
    MHR = "mhr"
    AUTO = "auto"


class ConversionRequirement(Enum):
    """Types of conversions that might be required."""

    NONE = "none"  # Direct visualization
    SIMPLE = "simple"  # Simple format conversion
    SMPLIFY = "smplify"  # Requires SMPLify optimization
    NOT_SUPPORTED = "not_supported"  # Not supported


@dataclass
class CompatibilityResult:
    """Result of compatibility check.

    Attributes:
        compatible: Whether the combination is compatible
        confidence: Confidence level (0.0 to 1.0)
        conversion_required: Type of conversion required
        warnings: List of warning messages
        suggestions: List of suggestion messages
        estimated_time: Estimated time for conversion (in seconds, if applicable)
    """

    compatible: bool
    confidence: float
    conversion_required: ConversionRequirement
    warnings: list[str]
    suggestions: list[str]
    estimated_time: float | None = None


class CompatibilityMatrix:
    """Defines compatibility between formats, visualization modes, and body models.

    This class checks if a given motion format can be visualized with a specific
    visualization mode and body model, and provides information about required
    conversions.

    Examples:
        >>> matrix = CompatibilityMatrix()
        >>>
        >>> # Check if 263D can be visualized as skeleton
        >>> result = matrix.check_compatibility('263d', 'skeleton')
        >>> print(result.compatible)  # True
        >>>
        >>> # Check if 263D can be visualized as SMPL mesh
        >>> result = matrix.check_compatibility('263d', 'mesh', 'smpl')
        >>> print(result.conversion_required)  # SMPLIFY
    """

    # Define which formats work with skeleton visualization (all do!)
    SKELETON_COMPATIBLE = {
        "263d",
        "272d",
        "205d",
        "232d",
        "align",
        "motion",
        "joint_positions",
        "smpl_params",
        "smpl_poses",
        "mhr",
    }

    # Define which formats work directly with SMPL mesh
    SMPL_DIRECT = {
        "motion",  # If Motion has SMPL skeleton
        "smpl_params",
    }

    SMPL_SIMPLE_CONVERSION = {
        "272d",  # Via converter
        "205d",  # Via converter
        "232d",  # ALIGN via converter
        "align",  # ALIGN via converter (alias)
        "smpl_poses",  # Just need to add translation
    }

    SMPL_SMPLIFY_REQUIRED = {
        "263d",  # HumanML3D requires SMPLify optimization
        "joint_positions",  # Generic joint positions require SMPLify
    }

    # Define which formats work with R15
    R15_DIRECT = set()  # None work directly

    R15_SIMPLE_CONVERSION = {
        "motion",  # Via SMPL → R15 converter
        "272d",  # Via converter chain
        "205d",  # Via converter chain
        "smpl_params",  # Via SMPL → R15
    }

    R15_SMPLIFY_REQUIRED = {
        "263d",  # HumanML3D → SMPL → R15
    }

    # Define which formats work with MHR
    MHR_DIRECT = {
        "motion",  # If Motion has MHR skeleton
        "mhr",  # MHR native format (model_parameters)
    }

    MHR_SIMPLE_CONVERSION = {
        "272d",  # Via converter
        "205d",  # Via converter
    }

    MHR_NOT_SUPPORTED = {
        "263d",  # HumanML3D not compatible with MHR
        "joint_positions",  # Generic positions not compatible
    }

    def check_compatibility(
        self,
        format_type: str,
        mode: str,
        body_model: str | None = None,
        skeleton_type: str | None = None,
    ) -> CompatibilityResult:
        """Check if a format is compatible with a visualization mode and body model.

        Args:
            format_type: Detected format type (e.g., '263d', 'motion')
            mode: Visualization mode ('skeleton', 'mesh', 'auto')
            body_model: Body model for mesh visualization ('smpl', 'r15', 'mhr', 'auto')
            skeleton_type: Detected skeleton type (e.g., 'humanml3d', 'smpl')

        Returns:
            CompatibilityResult with compatibility information
        """
        # Handle AUTO mode
        if mode == "auto":
            # Auto mode: prefer skeleton for simplicity
            return self._check_skeleton_compatibility(format_type)

        # Check skeleton visualization
        if mode == "skeleton":
            return self._check_skeleton_compatibility(format_type)

        # Check mesh visualization
        if mode == "mesh":
            if not body_model or body_model == "auto":
                # Default to SMPL
                body_model = "smpl"

            if body_model == "smpl":
                return self._check_smpl_compatibility(format_type, skeleton_type)
            elif body_model == "r15":
                return self._check_r15_compatibility(format_type, skeleton_type)
            elif body_model == "mhr":
                return self._check_mhr_compatibility(format_type, skeleton_type)
            else:
                return CompatibilityResult(
                    compatible=False,
                    confidence=CONFIDENCE_CERTAIN,
                    conversion_required=ConversionRequirement.NOT_SUPPORTED,
                    warnings=[f"Unknown body model: {body_model}"],
                    suggestions=["Use 'smpl', 'r15', or 'mhr' as body_model"],
                )

        return CompatibilityResult(
            compatible=False,
            confidence=CONFIDENCE_CERTAIN,
            conversion_required=ConversionRequirement.NOT_SUPPORTED,
            warnings=[f"Unknown visualization mode: {mode}"],
            suggestions=["Use 'skeleton', 'mesh', or 'auto' as mode"],
        )

    def _check_skeleton_compatibility(self, format_type: str) -> CompatibilityResult:
        """Check compatibility with skeleton visualization."""
        if format_type in self.SKELETON_COMPATIBLE:
            return CompatibilityResult(
                compatible=True,
                confidence=CONFIDENCE_CERTAIN,
                conversion_required=ConversionRequirement.SIMPLE,
                warnings=[],
                suggestions=[],
                estimated_time=TIME_INSTANT,
            )

        return CompatibilityResult(
            compatible=False,
            confidence=CONFIDENCE_CERTAIN,
            conversion_required=ConversionRequirement.NOT_SUPPORTED,
            warnings=[f"Format '{format_type}' cannot be visualized as skeleton"],
            suggestions=["Check format detection"],
        )

    def _check_smpl_compatibility(self, format_type: str, skeleton_type: str | None) -> CompatibilityResult:
        """Check compatibility with SMPL mesh visualization."""
        # Direct support
        if format_type in self.SMPL_DIRECT:
            # Check if Motion has SMPL-compatible skeleton
            if format_type == "motion" and skeleton_type not in ["smpl", "humanml3d", "smpl_compatible"]:
                return CompatibilityResult(
                    compatible=True,
                    confidence=CONFIDENCE_MODERATE,
                    conversion_required=ConversionRequirement.SIMPLE,
                    warnings=[f"Motion has {skeleton_type} skeleton, conversion may lose fidelity"],
                    suggestions=["Consider using skeleton visualization for this skeleton type"],
                    estimated_time=TIME_SIMPLE,
                )

            return CompatibilityResult(
                compatible=True,
                confidence=CONFIDENCE_CERTAIN,
                conversion_required=ConversionRequirement.NONE,
                warnings=[],
                suggestions=[],
                estimated_time=TIME_FAST,
            )

        # Simple conversion
        if format_type in self.SMPL_SIMPLE_CONVERSION:
            return CompatibilityResult(
                compatible=True,
                confidence=CONFIDENCE_HIGH,
                conversion_required=ConversionRequirement.SIMPLE,
                warnings=[],
                suggestions=[],
                estimated_time=TIME_SIMPLE,
            )

        # Requires SMPLify optimization
        if format_type in self.SMPL_SMPLIFY_REQUIRED:
            return CompatibilityResult(
                compatible=True,
                confidence=CONFIDENCE_GOOD,
                conversion_required=ConversionRequirement.SMPLIFY,
                warnings=[
                    "This requires SMPLify optimization which can be slow (10-30 seconds per sequence)",
                    "Results may not perfectly match original joint positions",
                ],
                suggestions=[
                    "Consider using skeleton visualization for faster results",
                    "Use on_incompatible='auto_skeleton' to automatically fall back to skeleton",
                ],
                estimated_time=TIME_SMPLIFY,
            )

        return CompatibilityResult(
            compatible=False,
            confidence=CONFIDENCE_CERTAIN,
            conversion_required=ConversionRequirement.NOT_SUPPORTED,
            warnings=[f"Format '{format_type}' cannot be converted to SMPL"],
            suggestions=["Use skeleton visualization instead"],
        )

    def _check_r15_compatibility(self, format_type: str, skeleton_type: str | None) -> CompatibilityResult:
        """Check compatibility with R15 mesh visualization."""
        # Simple conversion
        if format_type in self.R15_SIMPLE_CONVERSION:
            return CompatibilityResult(
                compatible=True,
                confidence=CONFIDENCE_GOOD,
                conversion_required=ConversionRequirement.SIMPLE,
                warnings=["Conversion to R15 may lose some joint detail"],
                suggestions=[],
                estimated_time=TIME_MODERATE,
            )

        # Requires SMPLify first
        if format_type in self.R15_SMPLIFY_REQUIRED:
            return CompatibilityResult(
                compatible=True,
                confidence=CONFIDENCE_MODERATE,
                conversion_required=ConversionRequirement.SMPLIFY,
                warnings=[
                    "Requires conversion to SMPL first (via SMPLify), then SMPL → R15",
                    "This can be slow (10-30 seconds) and may lose fidelity",
                ],
                suggestions=[
                    "Consider using skeleton visualization",
                    "Use on_incompatible='auto_skeleton' for faster results",
                ],
                estimated_time=TIME_SMPLIFY_CHAIN,
            )

        return CompatibilityResult(
            compatible=False,
            confidence=CONFIDENCE_CERTAIN,
            conversion_required=ConversionRequirement.NOT_SUPPORTED,
            warnings=[f"Format '{format_type}' cannot be converted to R15"],
            suggestions=["Use skeleton visualization or convert to SMPL first"],
        )

    def _check_mhr_compatibility(self, format_type: str, skeleton_type: str | None) -> CompatibilityResult:
        """Check compatibility with MHR mesh visualization."""
        # Direct support
        if format_type in self.MHR_DIRECT:
            if format_type == "mhr":
                return CompatibilityResult(
                    compatible=True,
                    confidence=1.0,
                    conversion_required=ConversionRequirement.NONE,
                    warnings=[],
                    suggestions=[],
                    estimated_time=1.0,
                )
            if format_type == "motion" and skeleton_type == "mhr":
                return CompatibilityResult(
                    compatible=True,
                    confidence=CONFIDENCE_CERTAIN,
                    conversion_required=ConversionRequirement.NONE,
                    warnings=[],
                    suggestions=[],
                    estimated_time=TIME_SIMPLE,
                )

        # Simple conversion
        if format_type in self.MHR_SIMPLE_CONVERSION:
            return CompatibilityResult(
                compatible=True,
                confidence=CONFIDENCE_GOOD,
                conversion_required=ConversionRequirement.SIMPLE,
                warnings=["Conversion to MHR may require retargeting"],
                suggestions=[],
                estimated_time=TIME_MODERATE,
            )

        # Not supported
        if format_type in self.MHR_NOT_SUPPORTED:
            return CompatibilityResult(
                compatible=False,
                confidence=CONFIDENCE_CERTAIN,
                conversion_required=ConversionRequirement.NOT_SUPPORTED,
                warnings=[f"Format '{format_type}' is not compatible with MHR"],
                suggestions=[
                    "Use skeleton visualization",
                    "Convert to SMPL or 272D format first",
                ],
            )

        # Unknown format
        return CompatibilityResult(
            compatible=False,
            confidence=CONFIDENCE_LOW,
            conversion_required=ConversionRequirement.NOT_SUPPORTED,
            warnings=[f"Compatibility of '{format_type}' with MHR is unknown"],
            suggestions=["Use skeleton visualization or test with SMPL first"],
        )

    def get_recommended_mode(self, format_type: str, skeleton_type: str | None = None) -> tuple[str, str | None]:
        """Get recommended visualization mode and body model for a format.

        Args:
            format_type: Detected format type
            skeleton_type: Detected skeleton type

        Returns:
            Tuple of (mode, body_model) recommendations
        """
        # For formats that work well with SMPL
        if format_type in (self.SMPL_DIRECT | self.SMPL_SIMPLE_CONVERSION):
            return ("mesh", "smpl")

        # For MHR-specific motion
        if format_type == "motion" and skeleton_type == "mhr":
            return ("mesh", "mhr")

        # For formats requiring SMPLify, recommend skeleton for speed
        if format_type in self.SMPL_SMPLIFY_REQUIRED:
            return ("skeleton", None)

        # Default to skeleton (always works)
        return ("skeleton", None)


def check_compatibility(
    detected_format: DetectedFormat,
    mode: str,
    body_model: str | None = None,
) -> CompatibilityResult:
    """Convenience function to check compatibility.

    Args:
        detected_format: Format detection result
        mode: Visualization mode
        body_model: Optional body model

    Returns:
        CompatibilityResult

    Examples:
        >>> from motion_toolbox.visualization.format_detector import detect_format
        >>> detected = detect_format(np.random.randn(120, 263))
        >>> result = check_compatibility(detected, 'skeleton')
        >>> print(result.compatible)  # True
    """
    matrix = CompatibilityMatrix()
    return matrix.check_compatibility(
        detected_format.format_type,
        mode,
        body_model,
        detected_format.skeleton_type,
    )
