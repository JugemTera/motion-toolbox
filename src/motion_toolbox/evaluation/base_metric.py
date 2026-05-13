"""Base classes for motion evaluation metrics.

This module provides the foundational architecture for implementing different
evaluation metric conventions and variants. Each metric can have multiple
implementations following different research standards or practical needs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..core.motion import Motion


@dataclass
class MetricResult:
    """Container for metric computation results.

    Args:
        per_frame: Metric values for each frame, shape (num_frames,)
        per_joint: Metric values for each joint averaged across frames, shape (num_joints,)
        frame_average: Average metric value per frame, shape (num_frames,)
        sequence_average: Single average value for entire sequence
        metadata: Additional information about the computation (thresholds, methods, etc.)
        ratio: Optional metric-specific ratio (e.g., skating during contact, acceleration above threshold)

    Examples:
        >>> result = MetricResult(
        ...     per_frame=np.array([0.1, 0.2, 0.15]),
        ...     per_joint=np.array([0.12, 0.18]),
        ...     frame_average=np.array([0.15, 0.19, 0.135]),
        ...     sequence_average=0.16,
        ...     metadata={"method": "height_based", "threshold": 0.05},
        ...     ratio=0.25
        ... )
    """

    per_frame: np.ndarray
    per_joint: np.ndarray
    frame_average: np.ndarray
    sequence_average: float
    metadata: dict[str, Any]
    ratio: float | None = None  # Metric-specific ratio (e.g., skating during contact, acceleration above threshold)


class MetricConvention(Enum):
    """Standard conventions for metric implementations.

    Different research papers and implementations may use varying approaches
    for the same underlying metric concept. This enum helps identify which
    convention is being used.
    """

    # Foot skating conventions
    HOIDINI_FOOT_SKATING = "hoidini_foot_skating"  # HOIDINI paper implementation
    CLOSD_FOOT_SLIDING = "closd_foot_sliding"  # CLoSD/PhysDiff implementation
    GMD_SMOOTHED_SKATING = "gmd_smoothed_skating"  # GMD velocity-based with smoothing
    HOIDINI_MESH_SKATING = "hoidini_mesh_skating"  # HOIDINI approach using mesh vertices
    CLOSD_MESH_SLIDING = "closd_mesh_sliding"  # CLoSD approach using mesh vertices

    # Jerk conventions
    MEAN_JERK = "mean_jerk"  # Mean jerk (StableMotion/GVHMR standard)
    RMS_JERK = "rms_jerk"  # Root mean square jerk
    PEAK_JERK = "peak_jerk"  # Peak jerk analysis (FlowMDM style)
    GRADIENT_JITTER = "gradient_jitter"  # Jitter using np.gradient (source TBD)

    # Ground penetration conventions
    SIMPLE_HEIGHT_CHECK = "simple_height_check"
    MESH_INTERSECTION = "mesh_intersection"
    CAPSULE_COLLISION = "capsule_collision"
    CLOSD_PENETRATION = "closd_penetration"  # CLoSD/PhysDiff implementation
    HOIDINI_MESH_PENETRATION = "hoidini_mesh_penetration"  # HOIDINI approach using mesh vertices
    CLOSD_MESH_PENETRATION = "closd_mesh_penetration"  # CLoSD approach using mesh vertices

    # Floating conventions
    CLOSD_FLOATING = "closd_floating"  # CLoSD/PhysDiff implementation

    # Acceleration conventions
    PEAK_DETECTION = "peak_detection"
    THRESHOLD_EXCEEDANCE = "threshold_exceedance"
    STATISTICAL_OUTLIERS = "statistical_outliers"

    # HOIDINI paper conventions (for comparison/validation)
    HOIDINI_FLOOR_METRICS = "hoidini_floor_metrics"

    # Dynamic score conventions
    HYBRID_VELOCITY_RANGE = "hybrid_velocity_range"  # Combines velocity and spatial range
    VELOCITY_ONLY = "velocity_only"  # Pure velocity-based dynamics
    RANGE_ONLY = "range_only"  # Pure spatial range-based dynamics
    WEIGHTED_JOINTS = "weighted_joints"  # Uses joint importance weights

    # Root orientation change conventions
    MAX_ROOT_ROTATION = "max_root_rotation"  # Maximum rotation angle between consecutive frames

    # Hand-specific conventions
    HAND_JOINT_LIMITS = "hand_joint_limits"  # Finger bone-length plausibility check
    HAND_PENETRATION = "hand_penetration"  # Inter-hand collision detection
    HAND_SMOOTHNESS = "hand_smoothness"  # Mean jerk of finger joints (m/s³)
    HAND_FLEXION = "hand_flexion"  # Finger flexion distribution


class EvalMetric(ABC):
    """Abstract base class for motion evaluation metrics.

    This class defines the interface that all evaluation metrics must implement.
    Each metric can have multiple variants following different conventions
    or research standards.

    Args:
        convention: The implementation convention to use
        **kwargs: Convention-specific parameters

    Examples:
        >>> # Height-based foot skating detection
        >>> foot_metric = FootSkatingMetric(
        ...     convention=MetricConvention.HEIGHT_BASED_CONTACT,
        ...     velocity_threshold=0.02,
        ...     contact_height_threshold=0.05
        ... )
        >>>
        >>> # StableMotion-style displacement-based detection
        >>> foot_metric_stable = FootSkatingMetric(
        ...     convention=MetricConvention.DISPLACEMENT_BASED_CONTACT,
        ...     velocity_threshold=0.02,
        ...     displacement_threshold=0.01
        ... )
        >>>
        >>> result = foot_metric.compute(motion)
        >>> print(f"Skating score: {result.sequence_average:.4f}")
        >>> print(f"Method used: {result.metadata['convention']}")
    """

    def __init__(self, convention: MetricConvention, **kwargs):
        """Initialize the metric with a specific convention.

        Args:
            convention: The implementation convention to use
            **kwargs: Convention-specific parameters
        """
        self.convention = convention
        self.params = kwargs
        self._validate_params()

    @abstractmethod
    def compute(self, motion: Motion) -> MetricResult:
        """Compute the metric for the given motion.

        Args:
            motion: Motion object to analyze

        Returns:
            MetricResult containing the computed values and metadata
        """
        pass

    @abstractmethod
    def _validate_params(self) -> None:
        """Validate that required parameters are provided for the convention.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        pass

    @classmethod
    @abstractmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        """Get list of conventions supported by this metric class.

        Returns:
            List of supported MetricConvention values
        """
        pass

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        """Get default parameters for a specific convention.

        Args:
            convention: The convention to get defaults for

        Returns:
            Dictionary of default parameter values
        """
        return {}

    def get_info(self) -> dict[str, Any]:
        """Get information about this metric instance.

        Returns:
            Dictionary containing metric name, convention, and parameters
        """
        return {
            "metric_name": self.__class__.__name__,
            "convention": self.convention.value,
            "parameters": self.params,
        }


class MetricRegistry:
    """Registry for managing different metric implementations.

    This class provides a centralized way to register, discover, and instantiate
    different metric implementations and their conventions.

    Examples:
        >>> # Register a new metric implementation
        >>> MetricRegistry.register("foot_skating", FootSkatingMetric)
        >>>
        >>> # Get available conventions for a metric
        >>> conventions = MetricRegistry.get_conventions("foot_skating")
        >>> print(conventions)  # [MetricConvention.HEIGHT_BASED_CONTACT, ...]
        >>>
        >>> # Create metric instance with specific convention
        >>> metric = MetricRegistry.create_metric(
        ...     "foot_skating",
        ...     MetricConvention.HEIGHT_BASED_CONTACT,
        ...     velocity_threshold=0.02
        ... )
    """

    _registry: dict[str, type[EvalMetric]] = {}

    @classmethod
    def register(cls, name: str, metric_class: type[EvalMetric]) -> None:
        """Register a metric implementation.

        Args:
            name: Name to register the metric under
            metric_class: The metric class to register
        """
        cls._registry[name] = metric_class

    @classmethod
    def get_metric_class(cls, name: str) -> type[EvalMetric]:
        """Get a registered metric class by name.

        Args:
            name: Name of the metric class

        Returns:
            The metric class

        Raises:
            KeyError: If metric is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Metric '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def get_conventions(cls, name: str) -> list[MetricConvention]:
        """Get supported conventions for a metric.

        Args:
            name: Name of the metric

        Returns:
            List of supported conventions
        """
        metric_class = cls.get_metric_class(name)
        return metric_class.get_supported_conventions()

    @classmethod
    def create_metric(cls, name: str, convention: MetricConvention, **kwargs) -> EvalMetric:
        """Create a metric instance with specific convention.

        Args:
            name: Name of the metric
            convention: Convention to use
            **kwargs: Convention-specific parameters

        Returns:
            Initialized metric instance
        """
        metric_class = cls.get_metric_class(name)
        return metric_class(convention=convention, **kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all registered metric names.

        Returns:
            List of registered metric names
        """
        return list(cls._registry.keys())

    @classmethod
    def get_metric_info(cls, name: str) -> dict[str, Any]:
        """Get information about a metric including supported conventions.

        Args:
            name: Name of the metric

        Returns:
            Dictionary with metric information
        """
        metric_class = cls.get_metric_class(name)
        conventions = metric_class.get_supported_conventions()

        # Get default parameters for each convention
        convention_defaults = {}
        for conv in conventions:
            convention_defaults[conv.value] = metric_class.get_default_params(conv)

        return {
            "name": name,
            "class": metric_class.__name__,
            "supported_conventions": [conv.value for conv in conventions],
            "default_parameters": convention_defaults,
        }
