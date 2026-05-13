"""Dynamic Score Metric for Motion Analysis.

This module implements metrics to measure how dynamic (vs static) a motion is,
combining temporal activity (velocity-based) and spatial coverage (range-based) measures.
"""

from typing import Any

import numpy as np

from motion_toolbox.core.motion import Motion
from motion_toolbox.evaluation.base_metric import EvalMetric, MetricConvention, MetricRegistry, MetricResult


class DynamicScoreMetric(EvalMetric):
    """Measures motion dynamism combining temporal and spatial activity.

    This metric evaluates how dynamic a motion is by combining:
    1. Temporal activity: How fast joints move (velocity-based)
    2. Spatial activity: How much space the motion covers (range-based)
    3. Activity ratio: Percentage of frames with significant motion

    A completely static motion scores 0, while highly dynamic motions score higher.
    The metric is normalized to provide meaningful comparisons across different motions.

    Conventions:
        - HYBRID_VELOCITY_RANGE: Combines velocity and range components (default)
        - VELOCITY_ONLY: Pure velocity-based dynamics
        - RANGE_ONLY: Pure spatial range-based dynamics
        - WEIGHTED_JOINTS: Uses joint importance weights
    """

    def __init__(self, convention: MetricConvention = MetricConvention.HYBRID_VELOCITY_RANGE, **params):
        """Initialize dynamic score metric.

        Args:
            convention: Computation method to use
            **params: Additional parameters:
                - velocity_weight: Weight for velocity component (default: 0.7)
                - range_weight: Weight for range component (default: 0.3)
                - activity_threshold: Threshold for "active" motion (default: 0.01 m/s)
                - normalize_by_duration: Whether to normalize by motion duration (default: True)
                - joint_weights: Dict of joint importance weights (optional)
        """
        super().__init__(convention, **params)

        # Set default parameters
        self.params.setdefault("velocity_weight", 0.7)
        self.params.setdefault("range_weight", 0.3)
        self.params.setdefault("activity_threshold", 0.01)  # 1 cm/s threshold
        self.params.setdefault("normalize_by_duration", True)

        # Validate weights sum to 1 for hybrid approach
        if convention == MetricConvention.HYBRID_VELOCITY_RANGE:
            vel_weight = self.params["velocity_weight"]
            range_weight = self.params["range_weight"]
            if abs(vel_weight + range_weight - 1.0) > 1e-6:
                raise ValueError(f"Velocity and range weights must sum to 1.0, got {vel_weight + range_weight}")

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        """Get list of supported conventions for dynamic score calculation."""
        return [
            MetricConvention.HYBRID_VELOCITY_RANGE,
            MetricConvention.VELOCITY_ONLY,
            MetricConvention.RANGE_ONLY,
            MetricConvention.WEIGHTED_JOINTS,
        ]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        """Get default parameters for the specified convention."""
        if convention == MetricConvention.HYBRID_VELOCITY_RANGE:
            return {
                "velocity_weight": 0.7,
                "range_weight": 0.3,
                "activity_threshold": 0.01,
                "normalize_by_duration": True,
            }
        elif convention == MetricConvention.VELOCITY_ONLY:
            return {
                "activity_threshold": 0.01,
            }
        elif convention == MetricConvention.RANGE_ONLY:
            return {
                "normalize_by_duration": True,
            }
        elif convention == MetricConvention.WEIGHTED_JOINTS:
            return {
                "activity_threshold": 0.01,
                "joint_weights": {
                    "pelvis": 1.0,
                    "spine": 0.8,
                    "head": 0.3,
                    "left_foot": 0.7,
                    "right_foot": 0.7,
                },
            }
        else:
            return {}

    def _validate_params(self) -> dict[str, Any]:
        """Validate metric parameters."""
        required_params = {}

        if self.convention == MetricConvention.HYBRID_VELOCITY_RANGE:
            required_params.update(
                {
                    "velocity_weight": "Weight for velocity component",
                    "range_weight": "Weight for range component",
                    "activity_threshold": "Threshold for active motion detection",
                }
            )
        elif self.convention == MetricConvention.VELOCITY_ONLY:
            required_params.update({"activity_threshold": "Threshold for active motion detection"})
        elif self.convention == MetricConvention.RANGE_ONLY:
            required_params.update({"normalize_by_duration": "Whether to normalize by motion duration"})
        elif self.convention == MetricConvention.WEIGHTED_JOINTS:
            required_params.update(
                {
                    "joint_weights": "Dictionary of joint importance weights",
                    "activity_threshold": "Threshold for active motion detection",
                }
            )

        return required_params

    def compute(self, motion: Motion) -> MetricResult:
        """Compute dynamic score for the given motion.

        Args:
            motion: Motion object to analyze

        Returns:
            MetricResult with dynamic scores and activity ratio
        """
        positions = motion.positions  # Shape: (frames, joints, 3)
        num_frames, num_joints = positions.shape[:2]

        if num_frames < 2:
            # Need at least 2 frames to compute dynamics
            return MetricResult(
                per_frame=np.zeros(num_frames),
                per_joint=np.zeros(num_joints),
                frame_average=np.zeros(num_frames),
                sequence_average=0.0,
                metadata={
                    "convention": self.convention.value,
                    "insufficient_frames": True,
                    "min_frames_required": 2,
                },
                ratio=0.0,
            )

        # Dispatch to appropriate computation method
        if self.convention == MetricConvention.HYBRID_VELOCITY_RANGE:
            return self._compute_hybrid_dynamics(motion, positions, num_frames, num_joints)
        elif self.convention == MetricConvention.VELOCITY_ONLY:
            return self._compute_velocity_dynamics(motion, positions, num_frames, num_joints)
        elif self.convention == MetricConvention.RANGE_ONLY:
            return self._compute_range_dynamics(motion, positions, num_frames, num_joints)
        elif self.convention == MetricConvention.WEIGHTED_JOINTS:
            return self._compute_weighted_dynamics(motion, positions, num_frames, num_joints)
        else:
            raise ValueError(f"Unsupported convention: {self.convention}")

    def _compute_hybrid_dynamics(
        self, motion: Motion, positions: np.ndarray, num_frames: int, num_joints: int
    ) -> MetricResult:
        """Compute dynamics using hybrid velocity + range approach."""
        velocities = motion.velocities  # Shape: (frames, joints, 3)

        # 1. Temporal component: velocity-based activity
        velocity_magnitudes = np.linalg.norm(velocities, axis=2)  # (frames, joints)
        temporal_per_frame = np.mean(velocity_magnitudes, axis=1)  # Average across joints per frame
        temporal_per_joint = np.mean(velocity_magnitudes, axis=0)  # Average across frames per joint
        temporal_score = np.mean(velocity_magnitudes)  # Overall temporal activity

        # 2. Spatial component: range-based activity
        joint_ranges = np.max(positions, axis=0) - np.min(positions, axis=0)  # (joints, 3)
        range_magnitudes = np.linalg.norm(joint_ranges, axis=1)  # (joints,)
        spatial_per_joint = range_magnitudes
        # Broadcast spatial component to per-frame (constant across time)
        spatial_per_frame = np.full(num_frames, np.mean(range_magnitudes))
        spatial_score = np.mean(range_magnitudes)

        # 3. Combine components
        velocity_weight = self.params["velocity_weight"]
        range_weight = self.params["range_weight"]

        combined_per_frame = velocity_weight * temporal_per_frame + range_weight * spatial_per_frame
        combined_per_joint = velocity_weight * temporal_per_joint + range_weight * spatial_per_joint
        combined_score = velocity_weight * temporal_score + range_weight * spatial_score

        # 4. Calculate activity ratio (percentage of frames with significant motion)
        activity_threshold = self.params["activity_threshold"]
        active_frames = temporal_per_frame > activity_threshold
        activity_ratio = np.sum(active_frames) / num_frames if num_frames > 0 else 0.0

        return MetricResult(
            per_frame=combined_per_frame,
            per_joint=combined_per_joint,
            frame_average=combined_per_frame,  # Same as per_frame for this metric
            sequence_average=combined_score,
            metadata={
                "convention": self.convention.value,
                "velocity_weight": velocity_weight,
                "range_weight": range_weight,
                "temporal_score": temporal_score,
                "spatial_score": spatial_score,
                "activity_threshold": activity_threshold,
                "active_frames": int(np.sum(active_frames)),
                "total_frames": num_frames,
            },
            ratio=activity_ratio,
        )

    def _compute_velocity_dynamics(
        self, motion: Motion, positions: np.ndarray, num_frames: int, num_joints: int
    ) -> MetricResult:
        """Compute dynamics using only velocity information."""
        velocities = motion.velocities
        velocity_magnitudes = np.linalg.norm(velocities, axis=2)

        per_frame = np.mean(velocity_magnitudes, axis=1)
        per_joint = np.mean(velocity_magnitudes, axis=0)
        sequence_average = np.mean(velocity_magnitudes)

        # Activity ratio
        activity_threshold = self.params["activity_threshold"]
        active_frames = per_frame > activity_threshold
        activity_ratio = np.sum(active_frames) / num_frames if num_frames > 0 else 0.0

        return MetricResult(
            per_frame=per_frame,
            per_joint=per_joint,
            frame_average=per_frame,
            sequence_average=sequence_average,
            metadata={
                "convention": self.convention.value,
                "activity_threshold": activity_threshold,
                "active_frames": int(np.sum(active_frames)),
                "total_frames": num_frames,
            },
            ratio=activity_ratio,
        )

    def _compute_range_dynamics(
        self, motion: Motion, positions: np.ndarray, num_frames: int, num_joints: int
    ) -> MetricResult:
        """Compute dynamics using only spatial range information."""
        joint_ranges = np.max(positions, axis=0) - np.min(positions, axis=0)
        range_magnitudes = np.linalg.norm(joint_ranges, axis=1)

        # For range-based, per_frame is constant (spatial doesn't change over time)
        per_frame = np.full(num_frames, np.mean(range_magnitudes))
        per_joint = range_magnitudes
        sequence_average = np.mean(range_magnitudes)

        # Normalize by duration if requested
        if self.params.get("normalize_by_duration", True):
            duration = num_frames / motion.framerate
            sequence_average = sequence_average / duration
            per_frame = per_frame / duration

        # For spatial metrics, ratio could be "joints with significant range"
        range_threshold = 0.05  # 5cm minimum range to be considered "active"
        active_joints = range_magnitudes > range_threshold
        joint_activity_ratio = np.sum(active_joints) / num_joints if num_joints > 0 else 0.0

        return MetricResult(
            per_frame=per_frame,
            per_joint=per_joint,
            frame_average=per_frame,
            sequence_average=sequence_average,
            metadata={
                "convention": self.convention.value,
                "normalize_by_duration": self.params.get("normalize_by_duration", True),
                "range_threshold": range_threshold,
                "active_joints": int(np.sum(active_joints)),
                "total_joints": num_joints,
            },
            ratio=joint_activity_ratio,
        )

    def _compute_weighted_dynamics(
        self, motion: Motion, positions: np.ndarray, num_frames: int, num_joints: int
    ) -> MetricResult:
        """Compute dynamics using joint importance weights."""
        velocities = motion.velocities
        velocity_magnitudes = np.linalg.norm(velocities, axis=2)

        # Get joint weights
        joint_weights = self.params.get("joint_weights", {})
        joint_names = motion.skeleton.joint_names

        # Create weight array
        weights = np.ones(num_joints)
        for i, joint_name in enumerate(joint_names):
            if joint_name in joint_weights:
                weights[i] = joint_weights[joint_name]

        # Apply weights
        weighted_velocities = velocity_magnitudes * weights[np.newaxis, :]  # Broadcast weights

        per_frame = np.sum(weighted_velocities, axis=1) / np.sum(weights)  # Weighted average per frame
        per_joint = np.mean(weighted_velocities, axis=0)  # Keep weights for per-joint
        sequence_average = np.sum(weighted_velocities) / (num_frames * np.sum(weights))

        # Activity ratio with weights
        activity_threshold = self.params["activity_threshold"]
        active_frame_scores = np.sum(weighted_velocities > activity_threshold, axis=1)
        active_frames = active_frame_scores > 0
        activity_ratio = np.sum(active_frames) / num_frames if num_frames > 0 else 0.0

        return MetricResult(
            per_frame=per_frame,
            per_joint=per_joint,
            frame_average=per_frame,
            sequence_average=sequence_average,
            metadata={
                "convention": self.convention.value,
                "joint_weights_applied": len(joint_weights),
                "total_weight": float(np.sum(weights)),
                "activity_threshold": activity_threshold,
                "active_frames": int(np.sum(active_frames)),
                "total_frames": num_frames,
            },
            ratio=activity_ratio,
        )


# Register the metric
MetricRegistry.register("dynamic_score", DynamicScoreMetric)
