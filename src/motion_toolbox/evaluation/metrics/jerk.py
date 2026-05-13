"""Jerk metric implementations following different conventions.

This module implements various approaches to measure motion smoothness
using jerk (third derivative of position) with different calculation methods.
"""

from typing import Any

import numpy as np

from ...core.motion import Motion
from ..base_metric import EvalMetric, MetricConvention, MetricRegistry, MetricResult


class JerkMetric(EvalMetric):
    """Motion jerk (smoothness) measurement with multiple conventions.

    Jerk measures the rate of change of acceleration, providing insight into
    motion smoothness. High jerk values indicate abrupt changes that may appear
    unnatural or mechanically implausible.

    Supported Conventions:
    - MEAN_JERK: StableMotion/GVHMR standard (mean(norm(jerk))/10.0) - research standard
    - RMS_JERK: Root mean square jerk calculation (in HOIDINI DNO loss, not eval)
    - PEAK_JERK: Peak jerk analysis with smoothing (FlowMDM, MotionMillion style) - research standard
    - GRADIENT_JITTER: Jitter using np.gradient method (source TBD)

    Note: RMS_JERK convention produces identical results to HOIDINI paper implementation
    (validated with <0.000053 difference, which is floating-point precision).

    Examples:
        >>> # Mean jerk calculation (StableMotion/GVHMR standard)
        >>> metric = JerkMetric(convention=MetricConvention.MEAN_JERK)
        >>>
        >>> # RMS jerk for better statistical properties
        >>> metric_rms = JerkMetric(convention=MetricConvention.RMS_JERK)
        >>>
        >>> # Peak jerk analysis (FlowMDM style)
        >>> metric_peak = JerkMetric(
        ...     convention=MetricConvention.PEAK_JERK,
        ...     jerk_threshold=100.0,
        ...     smoothing_window=1
        ... )
        >>>
        >>> result = metric.compute(motion)
        >>> print(f"Jerk score: {result.sequence_average:.4f}")
    """

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        """Get list of supported conventions for jerk calculation."""
        return [
            MetricConvention.MEAN_JERK,
            MetricConvention.RMS_JERK,
            MetricConvention.PEAK_JERK,
            MetricConvention.GRADIENT_JITTER,
        ]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        """Get default parameters for each convention."""
        defaults = {
            MetricConvention.MEAN_JERK: {
                "jerk_threshold": 10.0,  # StableMotion/GVHMR use /10.0 scaling, adjusted threshold
            },
            MetricConvention.RMS_JERK: {
                "jerk_threshold": 500.0,  # m/s³ - threshold for "high jerk" detection (motion capture)
            },
            MetricConvention.PEAK_JERK: {
                "jerk_threshold": 100.0,  # m/s³ - threshold for peak jerk detection (not scaled)
                "smoothing_window": 1,  # No smoothing in research implementations
            },
            MetricConvention.GRADIENT_JITTER: {
                "jerk_threshold": 10.0,  # Same threshold as mean jerk for consistency
            },
        }
        return defaults.get(convention, {})

    def _validate_params(self) -> None:
        """Validate parameters for the selected convention."""
        pass  # No validation needed for current conventions

    def compute(self, motion: Motion) -> MetricResult:
        """Compute jerk metric using the specified convention.

        Args:
            motion: Motion object to analyze

        Returns:
            MetricResult with jerk measurements and metadata
        """
        positions = motion.positions  # Shape: (frames, joints, 3)
        num_frames, num_joints = positions.shape[:2]
        dt = 1.0 / motion.framerate

        if num_frames < 4:
            # Need at least 4 frames to compute jerk (3rd derivative)
            return MetricResult(
                per_frame=np.zeros(num_frames),
                per_joint=np.zeros(num_joints),
                frame_average=np.zeros(num_frames),
                sequence_average=0.0,
                metadata={
                    "convention": self.convention.value,
                    "insufficient_frames": True,
                    "min_frames_required": 4,
                },
            )

        # Calculate jerk using finite differences
        # jerk = d³x/dt³ ≈ (x[t+3] - 3*x[t+2] + 3*x[t+1] - x[t]) / dt³
        jerk = np.zeros((num_frames - 3, num_joints, 3))

        for i in range(num_frames - 3):
            jerk[i] = (positions[i + 3] - 3 * positions[i + 2] + 3 * positions[i + 1] - positions[i]) / (dt**3)

        # Apply convention-specific processing
        if self.convention == MetricConvention.MEAN_JERK:
            result = self._compute_mean_jerk(jerk, num_frames, num_joints)
        elif self.convention == MetricConvention.RMS_JERK:
            result = self._compute_rms_jerk(jerk, num_frames, num_joints)
        elif self.convention == MetricConvention.PEAK_JERK:
            result = self._compute_peak_jerk(jerk, positions, num_frames, num_joints)
        elif self.convention == MetricConvention.GRADIENT_JITTER:
            result = self._compute_gradient_jitter(positions, num_frames, num_joints, dt)
        else:
            raise ValueError(f"Unsupported convention: {self.convention}")

        # Add metadata
        result.metadata.update(
            {
                "convention": self.convention.value,
                "parameters": self.params,
                "dt": dt,
                "frames_used": num_frames - 3,
            }
        )

        return result

    def _compute_mean_jerk(self, jerk: np.ndarray, num_frames: int, num_joints: int) -> MetricResult:
        """Compute mean jerk calculation (StableMotion/GVHMR standard).

        This implements the standard jerk calculation used in both StableMotion and GVHMR papers:
        - Calculate jerk magnitudes: norm(jerk, axis=2)
        - Average across joints: mean(axis=1)
        - Apply scaling: /10.0

        Note: StableMotion originally used feet units and applied a FEET2METER conversion factor.
        Since our Motion class assumes meters, we only apply the /10.0 scaling factor.

        References:
        - StableMotion: Uses this algorithm with feet->meter conversion
        - GVHMR: Uses this exact algorithm without unit conversion
        """
        jerk_threshold = self.params.get("jerk_threshold", 1.0)  # Lower threshold due to /10.0 scaling

        # Calculate jerk magnitudes (equivalent to torch.norm(..., dim=2))
        jerk_magnitudes = np.linalg.norm(jerk, axis=2)  # Shape: (frames-3, joints)

        # Average across joints (equivalent to .mean(dim=-1))
        jitter_per_frame_computed = np.mean(jerk_magnitudes, axis=1)  # Shape: (frames-3,)

        # Apply StableMotion/GVHMR scaling factor
        jitter_per_frame_computed = jitter_per_frame_computed / 10.0

        # Pad with zeros for frames where jerk can't be computed
        jitter_per_frame = np.zeros(num_frames)
        jitter_per_frame[1:-2] = jitter_per_frame_computed

        # Per-joint jitter (average over time, then scale)
        jitter_per_joint = np.mean(jerk_magnitudes, axis=0) / 10.0  # Shape: (joints,)

        # Frame average is the jitter per frame
        frame_average = jitter_per_frame.copy()

        # Sequence average: overall mean jitter (StableMotion/GVHMR style)
        sequence_average = float(np.mean(jitter_per_frame_computed))

        # Calculate high jitter ratio
        frames_with_high_jitter = np.sum(jitter_per_frame_computed > jerk_threshold)
        total_frames = num_frames
        high_jitter_ratio = frames_with_high_jitter / total_frames if total_frames > 0 else 0.0

        return MetricResult(
            per_frame=jitter_per_frame,
            per_joint=jitter_per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "calculation_method": "mean_jerk_stablemotion_gvhmr",
                "reference": "StableMotion and GVHMR papers",
                "scaling_factor": 10.0,
                "note": "StableMotion uses FEET2METER conversion which we omit (Motion class assumes meters)",
                "high_jitter_threshold": jerk_threshold,
                "frames_with_high_jitter": int(frames_with_high_jitter),
                "total_frames": int(total_frames),
                "finite_differences": True,
            },
            ratio=float(high_jitter_ratio),
        )

    def _compute_rms_jerk(self, jerk: np.ndarray, num_frames: int, num_joints: int) -> MetricResult:
        """Compute RMS jerk for better statistical properties."""
        # Calculate RMS jerk magnitudes
        jerk_magnitudes = np.linalg.norm(jerk, axis=2)  # Shape: (frames-3, joints)

        # RMS calculation
        jerk_rms_per_frame = np.zeros(num_frames)
        jerk_rms_per_frame[1:-2] = np.sqrt(np.mean(jerk_magnitudes**2, axis=1))

        # Per-joint RMS jerk
        jerk_per_joint = np.sqrt(np.mean(jerk_magnitudes**2, axis=0))

        # Frame average
        frame_average = jerk_rms_per_frame.copy()

        # Sequence RMS average
        sequence_average = float(np.sqrt(np.mean(jerk_magnitudes**2)))

        # Calculate high jerk ratio (percentage of total frames with RMS jerk above threshold)
        jerk_threshold = self.params.get("jerk_threshold", 10.0)  # m/s³ - configurable threshold for "high jerk"
        frames_with_high_jerk = np.sum(np.sqrt(np.mean(jerk_magnitudes**2, axis=1)) > jerk_threshold)
        total_frames = num_frames  # Use total frames, not just frames where jerk can be computed
        high_jerk_ratio = frames_with_high_jerk / total_frames if total_frames > 0 else 0.0

        return MetricResult(
            per_frame=jerk_rms_per_frame,
            per_joint=jerk_per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "calculation_method": "rms",
                "high_jerk_threshold": jerk_threshold,
                "frames_with_high_jerk": int(frames_with_high_jerk),
                "total_frames": int(total_frames),
                "jerk_computation_frames": int(len(jerk_magnitudes)),  # Frames where jerk could be computed
            },
            ratio=float(high_jerk_ratio),
        )

    def _compute_peak_jerk(
        self, jerk: np.ndarray, positions: np.ndarray, num_frames: int, num_joints: int
    ) -> MetricResult:
        """Compute peak jerk analysis (FlowMDM style).

        This method focuses on peak jerk values which are more indicative of motion quality
        issues than average values. It applies smoothing to reduce noise sensitivity and
        identifies frames with exceptionally high jerk values.
        """
        smoothing_window = self.params.get("smoothing_window", 5)
        jerk_threshold = self.params.get("jerk_threshold", 100.0)

        # Calculate jerk magnitudes
        jerk_magnitudes = np.linalg.norm(jerk, axis=2)  # Shape: (frames-3, joints)

        # Apply smoothing to reduce noise sensitivity
        if smoothing_window > 1:
            try:
                from scipy.ndimage import uniform_filter1d

                # Smooth along the time axis (axis=0)
                jerk_magnitudes_smooth = uniform_filter1d(jerk_magnitudes, size=smoothing_window, axis=0)
            except ImportError:
                # Fallback to simple moving average if scipy not available
                jerk_magnitudes_smooth = self._simple_moving_average(jerk_magnitudes, smoothing_window)
        else:
            jerk_magnitudes_smooth = jerk_magnitudes

        # Find peak jerk per frame (maximum across all joints)
        peak_jerk_per_frame_computed = np.max(jerk_magnitudes_smooth, axis=1)  # Shape: (frames-3,)

        # Pad with zeros for frames where jerk can't be computed
        peak_jerk_per_frame = np.zeros(num_frames)
        peak_jerk_per_frame[1:-2] = peak_jerk_per_frame_computed

        # Per-joint peak jerk (maximum over time for each joint)
        peak_jerk_per_joint = np.max(jerk_magnitudes_smooth, axis=0)

        # Frame average is the peak jerk per frame
        frame_average = peak_jerk_per_frame.copy()

        # Sequence average: mean of peak jerk values (not overall peak)
        sequence_average = float(np.mean(peak_jerk_per_frame_computed))

        # Calculate high jerk ratio (percentage of total frames with peak jerk above threshold)
        frames_with_high_jerk = np.sum(peak_jerk_per_frame_computed > jerk_threshold)
        total_frames = num_frames
        high_jerk_ratio = frames_with_high_jerk / total_frames if total_frames > 0 else 0.0

        # Overall peak jerk (maximum across all frames and joints)
        overall_peak_jerk = float(np.max(jerk_magnitudes_smooth))

        return MetricResult(
            per_frame=peak_jerk_per_frame,
            per_joint=peak_jerk_per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "calculation_method": "peak_jerk",
                "reference": "FlowMDM style peak jerk analysis",
                "smoothing_window": smoothing_window,
                "high_jerk_threshold": jerk_threshold,
                "frames_with_high_jerk": int(frames_with_high_jerk),
                "total_frames": int(total_frames),
                "jerk_computation_frames": int(len(jerk_magnitudes)),
                "overall_peak_jerk": overall_peak_jerk,
            },
            ratio=float(high_jerk_ratio),
        )

    def _simple_moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Simple moving average fallback when scipy is not available."""
        if window_size <= 1:
            return data

        # Pad the data to handle edges
        pad_width = [(window_size // 2, window_size // 2)] + [(0, 0)] * (data.ndim - 1)
        padded_data = np.pad(data, pad_width, mode="edge")

        # Apply moving average along the first axis (time)
        smoothed = np.zeros_like(data)
        for i in range(data.shape[0]):
            start_idx = i
            end_idx = i + window_size
            smoothed[i] = np.mean(padded_data[start_idx:end_idx], axis=0)

        return smoothed

    def _compute_gradient_jitter(
        self, positions: np.ndarray, num_frames: int, num_joints: int, dt: float
    ) -> MetricResult:
        """Compute gradient-based jitter using np.gradient method.

        This method uses np.gradient for smoother derivative calculation but follows
        the same formulation as mean jerk for consistency:
        - joint_velocities = np.gradient(positions, dt, axis=0)
        - joint_accelerations = np.gradient(velocities, dt, axis=0)
        - joint_jerks = np.gradient(accelerations, dt, axis=0)
        - Calculate jerk magnitudes: norm(jerk, axis=2)
        - Average across joints: mean(axis=1)
        - Apply scaling: /10.0

        The only difference from mean jerk is using np.gradient instead of finite differences.
        """
        jerk_threshold = self.params.get("jerk_threshold", 10.0)  # Same threshold as mean jerk

        # Gradient-based calculation using np.gradient
        # positions shape: (frames, joints, 3)
        joint_velocities = np.gradient(positions, dt, axis=0)  # (frames, joints, 3)
        joint_accelerations = np.gradient(joint_velocities, dt, axis=0)  # (frames, joints, 3)
        joint_jerks = np.gradient(joint_accelerations, dt, axis=0)  # (frames, joints, 3)

        # Calculate jerk magnitudes (same as mean jerk: norm(jerk, axis=2))
        jerk_magnitudes = np.linalg.norm(joint_jerks, axis=2)  # Shape: (frames, joints)

        # Average across joints (same as mean jerk: mean(axis=1))
        jitter_per_frame_computed = np.mean(jerk_magnitudes, axis=1)  # Shape: (frames,)

        # Apply StableMotion/GVHMR scaling factor (same as mean jerk: /10.0)
        jitter_per_frame_computed = jitter_per_frame_computed / 10.0

        # Frame average is the jitter per frame
        frame_average = jitter_per_frame_computed.copy()

        # Per-joint jitter (average over time, then scale - same as mean jerk)
        jitter_per_joint = np.mean(jerk_magnitudes, axis=0) / 10.0  # Shape: (joints,)

        # Sequence average: overall mean jitter (same as mean jerk)
        sequence_average = float(np.mean(jitter_per_frame_computed))

        # Calculate high jitter ratio (same as mean jerk)
        frames_with_high_jerk = np.sum(jitter_per_frame_computed > jerk_threshold)
        total_frames = num_frames
        high_jerk_ratio = frames_with_high_jerk / total_frames if total_frames > 0 else 0.0

        # Debug information
        max_jerk = np.max(jitter_per_frame_computed) if len(jitter_per_frame_computed) > 0 else 0.0
        mean_jerk = np.mean(jitter_per_frame_computed) if len(jitter_per_frame_computed) > 0 else 0.0

        return MetricResult(
            per_frame=jitter_per_frame_computed,
            per_joint=jitter_per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "calculation_method": "gradient_based_jitter",
                "reference": "Gradient-based jitter with same formulation as mean jerk",
                "scaling_factor": 10.0,
                "note": "Same formulation as mean jerk, only difference is np.gradient vs finite differences",
                "high_jitter_threshold": jerk_threshold,
                "frames_with_high_jerk": int(frames_with_high_jerk),
                "total_frames": int(total_frames),
                "max_jerk_value": float(max_jerk),
                "mean_jerk_value": float(mean_jerk),
                "uses_gradient": True,
            },
            ratio=float(high_jerk_ratio),
        )


# Register the metric
MetricRegistry.register("jerk", JerkMetric)
