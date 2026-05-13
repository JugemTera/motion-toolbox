"""Ground floating detection metrics.

This module implements floating detection using various approaches:
- CLoSD-style floating detection (PhysDiff implementation)
- Enhanced with two-threshold system and frame-by-frame analysis
"""

from typing import Any

import numpy as np
import torch

from ...core.motion import Motion
from ..base_metric import EvalMetric, MetricConvention, MetricRegistry, MetricResult


class FloatingMetric(EvalMetric):
    """Metric for detecting when feet are floating above ground.

    Floating occurs when the lowest point of the body is above a threshold distance
    from the ground, indicating unrealistic hovering or floating behavior.

    Supports multiple conventions:
    - CLoSD floating: Based on PhysDiff implementation with tolerance-based detection
    """

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        """Get supported floating detection conventions."""
        return [
            MetricConvention.CLOSD_FLOATING,
        ]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        """Get default parameters for each convention."""
        defaults = {
            MetricConvention.CLOSD_FLOATING: {
                "tolerance": 0.005,  # 5mm tolerance (from CLoSD/PhysDiff)
                "contact_threshold": 0.05,  # 50mm contact threshold (same as other metrics)
                "floating_threshold": 0.05,  # 50mm floating threshold (same as contact - floating is non-contact)
            },
        }
        return defaults.get(convention, {})

    def _validate_params(self) -> None:
        """Validate parameters for the selected convention."""
        required_params = {
            MetricConvention.CLOSD_FLOATING: ["tolerance"],
        }

        required = required_params.get(self.convention, [])
        missing = [param for param in required if param not in self.params]

        if missing:
            raise ValueError(f"Missing required parameters for {self.convention.value}: {missing}")

    def compute(self, motion: Motion) -> MetricResult:
        """Compute floating metric based on the selected convention."""
        if self.convention == MetricConvention.CLOSD_FLOATING:
            result = self._compute_closd_floating(motion)
        else:
            raise ValueError(f"Unsupported convention: {self.convention}")

        return result

    def _compute_closd_floating(self, motion: Motion) -> MetricResult:
        """Compute CLoSD-style floating metric.

        Based on CLoSD/PhysDiff implementation:
        https://github.com/GuyTevet/CLoSD/blob/d8950bdb78312fcd2844f336b3fc079e7368c87f/closd/diffusion_planner/data_loaders/humanml/utils/metrics.py

        CLoSD approach:
        - Find lowest joint height per frame across all joints
        - Apply tolerance (5mm default)
        - Clamp floating values (only positive values above tolerance count)
        - Convert to millimeters for output
        """
        # Get joint positions from motion
        positions = motion.positions  # Shape: (frames, joints, 3)
        num_frames, num_joints = positions.shape[:2]
        tolerance = self.params["tolerance"]  # Original CLoSD tolerance for floating calculation
        contact_threshold = self.params.get("contact_threshold", 0.05)  # 50mm contact threshold
        floating_threshold = self.params.get(
            "floating_threshold", contact_threshold
        )  # Same as contact threshold for consistency

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]

        # Convert to torch for compatibility with CLoSD implementation
        positions_torch = torch.from_numpy(positions).float()

        # Get heights of all joints: shape (frames, joints)
        joint_heights = positions_torch[:, :, up_axis_idx]  # B x n_joints x L equivalent

        # Find lowest joint height per frame: shape (frames,)
        lowest_heights = joint_heights.min(dim=1)[0]  # B x L equivalent

        floating_list = []
        contact_frames = []
        floating_frames = []
        floating_events = []  # Track all positive distance events

        for frame_idx in range(num_frames):
            lowest_height = lowest_heights[frame_idx].item()

            # CLoSD formula: (lowest_height - tolerance).clamp_min(0) * 1000
            floating_value = max(lowest_height - tolerance, 0.0) * 1000.0 if lowest_height - tolerance > 0 else 0.0

            # Track contact and floating using proper thresholds
            is_contact = lowest_height <= contact_threshold  # Within contact threshold
            is_floating = lowest_height > floating_threshold  # Above floating threshold (significant floating)
            is_positive_distance = lowest_height > 0  # Any positive distance (floating event)

            contact_frames.append(is_contact)
            floating_frames.append(is_floating)
            floating_events.append(is_positive_distance)
            floating_list.append(floating_value)

        floating_per_frame = np.array(floating_list)
        contact_frames = np.array(contact_frames)
        floating_frames = np.array(floating_frames)
        floating_events = np.array(floating_events)

        # Calculate ratio (percentage of total frames with floating)
        total_frames = num_frames
        floating_frames_count = np.sum(floating_frames)
        floating_ratio = floating_frames_count / total_frames if total_frames > 0 else 0.0

        # Aggregate results (CLoSD style returns mean across all frames)
        sequence_average = float(np.mean(floating_per_frame))

        # For per_joint, we can't easily determine which joint contributed to each frame's minimum
        # So we distribute the floating evenly across all joints
        per_joint = np.full(num_joints, sequence_average / num_joints)

        # Frame average is same as per_frame for this metric
        frame_average = floating_per_frame

        return MetricResult(
            per_frame=floating_per_frame,
            per_joint=per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "reference": "CLoSD/PhysDiff implementation",
                "source": "https://github.com/GuyTevet/CLoSD",
                "aggregation": "lowest_joint_per_frame",
                "measurement": "floating_height_mm",
                "tolerance": tolerance,
                "total_frames": int(total_frames),
                "floating_frames_count": int(floating_frames_count),
                # Frame-by-frame data for visualization
                "contact_frames_bool": contact_frames.tolist(),
                "floating_frames_bool": floating_frames.tolist(),
                "floating_events_bool": floating_events.tolist(),
                "foot_heights": lowest_heights.tolist(),
                "contact_threshold": contact_threshold,
                "floating_threshold": floating_threshold,
            },
            ratio=float(floating_ratio),
        )


# Register the metric
MetricRegistry.register("floating", FloatingMetric)
