"""Hand-specific evaluation metrics for bimanual hand motion generation.

Metrics include:
- Finger bone-length plausibility check (proxy for joint limit violations)
- Inter-hand penetration (collision between hands)
- Hand jerk / smoothness (temporal smoothness of finger trajectories)
- Finger flexion distribution (statistical properties of finger angles)
"""

from typing import Any

import numpy as np

from ...core.motion import Motion
from ..base_metric import EvalMetric, MetricConvention, MetricResult


class _HandMetricBase(EvalMetric):
    """Base for hand metrics.

    Uses the first supported convention as the default so callers can
    instantiate without passing an explicit convention argument.
    """

    def __init__(self, convention: MetricConvention | None = None, **kwargs):
        supported = self.get_supported_conventions()
        if convention is None:
            convention = supported[0]
        super().__init__(convention, **kwargs)

    def _validate_params(self) -> None:
        pass


class HandJointLimitMetric(_HandMetricBase):
    """Check finger bone lengths against anatomical plausibility limits.

    Operates on joint positions (T, 32, 3) and measures what fraction of
    joint-frames have bone segments that exceed the maximum anatomically
    plausible length for each level of the finger chain.

    This is a position-space proxy for joint-angle limit violations: an
    over-extended bone segment is a necessary (though not sufficient)
    consequence of exceeding the flexion/extension range of the joint.
    For a full rotation-space check, the 6D rotations stored in the 393D
    format_hand representation should be used instead.

    Bone length thresholds (in metres, approximate 99th-percentile adult):
        MCP segment (wrist → proximal):  0.050 m
        PIP segment (proximal → middle): 0.040 m
        DIP segment (middle → distal):   0.030 m

    Conventions:
    - HAND_JOINT_LIMITS: fraction of (frame, joint) pairs with over-long bones
    """

    # Maximum plausible bone lengths (metres) for each segment level.
    # These are generous 99th-percentile adult values; violations indicate
    # the generator produced physically impossible poses.
    _MAX_BONE_LEN = {
        "mcp": 0.050,  # wrist → proximal phalanx
        "pip": 0.040,  # proximal → middle phalanx
        "dip": 0.030,  # middle → distal phalanx
    }

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        return [MetricConvention.HAND_JOINT_LIMITS]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        return {}

    def compute(self, motion: Motion, **kwargs) -> MetricResult:
        """Compute bone-length violations from a Motion object."""
        return self.compute_from_positions(motion.positions, **kwargs)

    def compute_from_positions(self, positions: np.ndarray, **kwargs) -> MetricResult:
        """Compute bone-length violations from joint positions (T, J, 3).

        Args:
            positions: (T, J, 3) joint positions. J must be 32 (bimanual) or 16
                (single hand, hand_offset=0 only).

        Returns:
            MetricResult where sequence_average is the fraction of (frame, joint)
            pairs with at least one over-long bone segment.
        """
        T, J = positions.shape[:2]
        violations = self._compute_violations(positions)
        per_frame = violations.mean(axis=1)
        per_joint = violations.mean(axis=0)
        ratio = float(violations.mean())
        return MetricResult(
            per_frame=per_frame,
            per_joint=per_joint,
            frame_average=per_frame,
            sequence_average=ratio,
            metadata={
                "metric": "hand_bone_length_violations",
                "thresholds_m": self._MAX_BONE_LEN,
                "num_joints": J,
            },
            ratio=ratio,
        )

    def _compute_violations(self, positions: np.ndarray) -> np.ndarray:
        """Return (T, J) binary array: 1 where the bone above joint j is too long."""
        T, J = positions.shape[:2]
        violations = np.zeros((T, J), dtype=np.float32)

        max_mcp = self._MAX_BONE_LEN["mcp"]
        max_pip = self._MAX_BONE_LEN["pip"]
        max_dip = self._MAX_BONE_LEN["dip"]

        hand_offsets = [0] if J == 16 else [0, 16]
        for hand_offset in hand_offsets:
            wrist = positions[:, hand_offset, :]
            for finger_start in range(1, 16, 3):
                idx = hand_offset + finger_start
                if idx + 2 >= J:
                    break
                p1 = positions[:, idx, :]
                p2 = positions[:, idx + 1, :]
                p3 = positions[:, idx + 2, :]

                violations[:, idx] = np.linalg.norm(p1 - wrist, axis=-1) > max_mcp
                violations[:, idx + 1] = np.linalg.norm(p2 - p1, axis=-1) > max_pip
                violations[:, idx + 2] = np.linalg.norm(p3 - p2, axis=-1) > max_dip

        return violations


class InterHandPenetrationMetric(_HandMetricBase):
    """Measure collision/penetration between left and right hands.

    Checks if joints from one hand are unrealistically close to or inside
    the volume of the other hand. Uses vectorised pairwise distance
    computation so performance is independent of sequence length.
    """

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        return [MetricConvention.HAND_PENETRATION]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        return {"penetration_threshold": 0.005}

    def compute(self, motion: Motion, **kwargs) -> MetricResult:
        return self.compute_from_positions(motion.positions, **kwargs)

    def compute_from_positions(self, positions: np.ndarray, **kwargs) -> MetricResult:
        """Compute inter-hand penetration from joint positions (T, 32, 3)."""
        threshold = kwargs.get("penetration_threshold", 0.005)
        T = positions.shape[0]

        left_joints = positions[:, :16, :]  # (T, 16, 3)
        right_joints = positions[:, 16:, :]  # (T, 16, 3)

        # Vectorised pairwise distances: (T, 16, 16)
        diffs = left_joints[:, :, None, :] - right_joints[:, None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)

        # Sum of depth-of-penetration for all interpenetrating joint pairs.
        depth = np.maximum(0.0, threshold - dists)  # (T, 16, 16)
        penetration_per_frame = depth.sum(axis=(1, 2)).astype(np.float32)

        penetration_ratio = float((penetration_per_frame > 0).mean())

        return MetricResult(
            per_frame=penetration_per_frame,
            per_joint=np.zeros(32),
            frame_average=penetration_per_frame,
            sequence_average=float(penetration_per_frame.mean()),
            metadata={"metric": "inter_hand_penetration", "threshold": threshold},
            ratio=penetration_ratio,
        )


class HandSmoothnessMetric(_HandMetricBase):
    """Measure temporal smoothness of hand motion via mean jerk.

    Computes the mean jerk (third derivative of position, in m/s³) across
    all 32 finger joints. Higher values indicate jerkier, less natural motion.

    Registers under HAND_SMOOTHNESS to avoid collision with the body-level
    MEAN_JERK convention used by JerkMetric.
    """

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        return [MetricConvention.HAND_SMOOTHNESS]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        return {"fps": 30}

    def compute(self, motion: Motion, **kwargs) -> MetricResult:
        return self.compute_from_positions(motion.positions, **kwargs)

    def compute_from_positions(self, positions: np.ndarray, **kwargs) -> MetricResult:
        """Compute mean jerk (m/s³) from joint positions (T, J, 3).

        Args:
            positions: (T, J, 3) joint positions
            fps: Capture frame-rate used to scale finite differences (default 30)

        Returns:
            MetricResult where sequence_average is mean jerk in m/s³.
        """
        fps = kwargs.get("fps", 30)
        T, J = positions.shape[:2]

        if T < 4:
            return MetricResult(
                per_frame=np.zeros(T),
                per_joint=np.zeros(J),
                frame_average=np.zeros(T),
                sequence_average=0.0,
                metadata={"metric": "hand_smoothness", "note": "too few frames"},
            )

        dt = 1.0 / fps
        vel = np.diff(positions, axis=0) / dt  # (T-1, J, 3)
        acc = np.diff(vel, axis=0) / dt  # (T-2, J, 3)
        jerk = np.diff(acc, axis=0) / dt  # (T-3, J, 3)  units: m/s³

        jerk_magnitude = np.linalg.norm(jerk, axis=-1)  # (T-3, J)

        per_frame = np.zeros(T, dtype=np.float32)
        per_frame[3:] = jerk_magnitude.mean(axis=1)  # align to frame index

        per_joint = jerk_magnitude.mean(axis=0)
        seq_avg = float(jerk_magnitude.mean())  # m/s³, no scaling

        return MetricResult(
            per_frame=per_frame,
            per_joint=per_joint,
            frame_average=per_frame,
            sequence_average=seq_avg,
            metadata={"metric": "hand_smoothness", "fps": fps, "unit": "m/s³"},
        )


class FingerFlexionMetric(_HandMetricBase):
    """Analyze finger flexion distribution.

    Measures the wrist-to-fingertip distance for each of the 10 fingers
    (5 per hand). Larger values mean the finger is more extended; smaller
    values mean it is more curled. Useful for checking whether generated
    motions have a realistic spread of finger poses.
    """

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        return [MetricConvention.HAND_FLEXION]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        return {}

    def compute(self, motion: Motion, **kwargs) -> MetricResult:
        return self.compute_from_positions(motion.positions, **kwargs)

    def compute_from_positions(self, positions: np.ndarray, **kwargs) -> MetricResult:
        """Compute wrist-to-tip distances for all fingers.

        Args:
            positions: (T, J, 3) joint positions. J must be 32 (bimanual) or
                16 (single hand).

        Returns:
            MetricResult where sequence_average is the mean wrist-to-tip
            distance across all fingers and frames (in metres).
        """
        T, J = positions.shape[:2]

        flexion_scores = []
        hand_offsets = [0] if J == 16 else [0, 16]
        for hand_offset in hand_offsets:
            wrist = positions[:, hand_offset, :]
            for finger_start in range(1, 16, 3):
                idx = hand_offset + finger_start
                if idx + 2 >= J:
                    break
                tip = positions[:, idx + 2, :]
                dist = np.linalg.norm(tip - wrist, axis=-1)  # (T,)
                flexion_scores.append(dist)

        flexion = np.array(flexion_scores)  # (N_fingers, T)
        per_frame = flexion.mean(axis=0)  # (T,)
        per_joint = np.zeros(J, dtype=np.float32)

        return MetricResult(
            per_frame=per_frame,
            per_joint=per_joint,
            frame_average=per_frame,
            sequence_average=float(per_frame.mean()),
            metadata={
                "metric": "finger_flexion",
                "unit": "metres (wrist-to-tip distance)",
                "mean_extension": float(flexion.mean()),
                "std_extension": float(flexion.std()),
                "min_extension": float(flexion.min()),
                "max_extension": float(flexion.max()),
            },
        )
