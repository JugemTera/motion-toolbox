"""Root orientation change metric for detecting rotation instability.

This module implements a metric to measure the maximum rotation angle change
of the root joint between consecutive frames. This is useful for detecting
sudden orientation changes, jittery rotations, or tracking instability.
"""

from typing import Any

import numpy as np

from ...core.motion import Motion
from ...utils.quaternion import axis_angle_to_rotation_matrix, extract_angle_from_rotation_matrix
from ..base_metric import EvalMetric, MetricConvention, MetricRegistry, MetricResult


class RootOrientationChangeMetric(EvalMetric):
    """Root orientation change measurement for rotation stability analysis.

    This metric computes the rotation angle between consecutive frames for the
    root joint. Large values indicate sudden orientation changes which may
    suggest tracking errors, jitter, or unnatural motion. This method follows the MotionMillion dataset conventions.

    The metric works by:
    1. Converting root axis-angle rotations to rotation matrices
    2. Computing relative rotation between consecutive frames: R_rel = R_i @ R_(i-1)^T
    3. Extracting rotation angle using: angle = arccos((trace(R_rel) - 1) / 2)
    4. Reporting the maximum angle change across all frame pairs

    Supported Conventions:
    - MAX_ROOT_ROTATION: Maximum rotation angle change between consecutive frames

    Examples:
        >>> from motion_toolbox.evaluation.metrics import RootOrientationChangeMetric
        >>> from motion_toolbox.evaluation.base_metric import MetricConvention
        >>>
        >>> # Create metric instance
        >>> metric = RootOrientationChangeMetric(
        ...     convention=MetricConvention.MAX_ROOT_ROTATION
        ... )
        >>>
        >>> # Compute for a motion sequence
        >>> result = metric.compute(motion)
        >>> print(f"Max root rotation change: {result.sequence_average:.4f} radians")
        >>> print(f"Max root rotation change: {np.degrees(result.sequence_average):.2f} degrees")
        >>>
        >>> # Check per-frame values to find problematic frames
        >>> problem_frames = np.where(result.per_frame > np.radians(30))[0]
        >>> print(f"Frames with >30 degree rotation: {problem_frames}")
    """

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        """Get list of supported conventions for root orientation change."""
        return [MetricConvention.MAX_ROOT_ROTATION]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        """Get default parameters for each convention."""
        defaults = {
            MetricConvention.MAX_ROOT_ROTATION: {
                "rotation_threshold_degrees": 45.0,  # Threshold in degrees for ratio calculation
            },
        }
        return defaults.get(convention, {})

    def _validate_params(self) -> None:
        """Validate parameters for the selected convention."""
        # No parameters to validate for MAX_ROOT_ROTATION
        pass

    def compute(self, motion: Motion) -> MetricResult:
        """Compute root orientation change metric.

        Args:
            motion: Motion object to analyze

        Returns:
            MetricResult with rotation angle measurements:
            - per_frame: Rotation angle change at each frame (radians)
            - per_joint: Single value for root joint (max angle)
            - frame_average: Same as per_frame (only one joint)
            - sequence_average: Maximum rotation angle change across sequence

        Raises:
            ValueError: If motion has fewer than 2 frames
        """
        num_frames = motion.num_frames

        # Need at least 2 frames to compute rotation change
        if num_frames < 2:
            return MetricResult(
                per_frame=np.zeros(num_frames),
                per_joint=np.zeros(1),  # Only root joint
                frame_average=np.zeros(num_frames),
                sequence_average=0.0,
                metadata={
                    "convention": self.convention.value,
                    "parameters": self.params,
                    "note": "Insufficient frames (need at least 2)",
                },
            )

        # Get root joint rotations (index 0) - shape: (num_frames, 3)
        # Make a copy to ensure we have a writable array
        root_rotations = np.array(motion.joint_rotations[:, 0, :])

        # Convert all root axis-angles to rotation matrices - shape: (num_frames, 3, 3)
        root_matrices = axis_angle_to_rotation_matrix(root_rotations)

        # Compute rotation angles between consecutive frames
        rotation_angles = np.zeros(num_frames, dtype=np.float64)

        for i in range(1, num_frames):
            # Compute relative rotation: R_rel = R_i @ R_(i-1)^T
            R_curr = root_matrices[i]
            R_prev = root_matrices[i - 1]
            R_relative = R_curr @ R_prev.T

            # Extract rotation angle using trace formula (returns degrees)
            rotation_angles[i] = extract_angle_from_rotation_matrix(R_relative)

        # First frame has no previous frame, so angle change is 0 (already initialized to 0)
        # Note: rotation_angles are already in DEGREES from extract_angle_from_rotation_matrix

        # Get rotation threshold from parameters (in degrees)
        rotation_threshold_degrees = self.params.get("rotation_threshold_degrees", 45.0)

        # Compute ratio of frames exceeding threshold (excluding first frame with 0 rotation)
        # Both rotation_angles and threshold are in degrees
        frames_above_threshold = rotation_angles[1:] > rotation_threshold_degrees
        ratio = float(np.sum(frames_above_threshold) / max(num_frames - 1, 1))

        # rotation_angles are already in degrees, no conversion needed
        per_frame_degrees = rotation_angles
        per_joint_degrees = np.array([np.max(per_frame_degrees)])  # Max angle for root joint
        frame_average_degrees = per_frame_degrees.copy()  # Same as per_frame (only one joint)
        sequence_average_degrees = float(np.max(per_frame_degrees))  # Maximum angle change

        return MetricResult(
            per_frame=per_frame_degrees,  # In degrees for easier interpretation
            per_joint=per_joint_degrees,  # In degrees
            frame_average=frame_average_degrees,  # In degrees
            sequence_average=sequence_average_degrees,  # In degrees
            metadata={
                "convention": self.convention.value,
                "parameters": self.params,
                "max_angle_degrees": sequence_average_degrees,
                "mean_angle_degrees": float(np.mean(per_frame_degrees[1:])),  # Exclude first frame
                "rotation_threshold_degrees": rotation_threshold_degrees,
                "frames_above_threshold": int(np.sum(frames_above_threshold)),
                "total_frames_analyzed": num_frames - 1,  # Exclude first frame
                # Flag to indicate single joint metric (no per-joint plot needed)
                "single_joint_metric": True,
            },
            ratio=ratio,  # Ratio of frames exceeding rotation threshold
        )


# Register the metric
MetricRegistry.register("root_orientation_change", RootOrientationChangeMetric)
