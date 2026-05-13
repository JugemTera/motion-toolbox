"""Ground penetration metric implementations following different conventions.

This module implements various approaches to detect ground penetration violations
in motion data, supporting different detection methods and body representations.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ...core.motion import Motion
from ..base_metric import EvalMetric, MetricConvention, MetricRegistry, MetricResult


class GroundPenetrationMetric(EvalMetric):
    """Ground penetration detection with multiple implementation conventions.

    Ground penetration occurs when body parts go below the ground plane, which is
    physically impossible. Different conventions use various methods to detect
    and measure these violations.

    Supported Conventions:
    - SIMPLE_HEIGHT_CHECK: Basic Y-coordinate comparison with ground height
    - MESH_INTERSECTION: Check for mesh-ground intersections (requires mesh data)
    - CAPSULE_COLLISION: Use capsule approximations for body parts
    - HOIDINI_FLOOR_METRICS: HOIDINI paper implementation (preserved for compatibility)
    - CLOSD_PENETRATION: CLoSD/PhysDiff-based implementation with lowest joint detection

    References:
    - HOIDINI: Original implementation for compatibility
    - CLoSD: Based on https://github.com/GuyTevet/CLoSD implementation

    Examples:
        >>> # Simple height-based detection
        >>> metric = GroundPenetrationMetric(
        ...     convention=MetricConvention.SIMPLE_HEIGHT_CHECK,
        ...     ground_height=0.0,
        ...     check_joints=["left_foot", "right_foot"]
        ... )
        >>>
        >>> # Capsule collision detection for more accurate results
        >>> metric_capsule = GroundPenetrationMetric(
        ...     convention=MetricConvention.CAPSULE_COLLISION,
        ...     ground_height=0.0,
        ...     capsule_radius=0.02,
        ...     check_joints=["left_foot", "right_foot", "left_hand", "right_hand"]
        ... )
        >>>
        >>> result = metric.compute(motion)
        >>> print(f"Penetration score: {result.sequence_average:.4f}")
    """

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        """Get list of supported conventions for ground penetration detection."""
        return [
            MetricConvention.SIMPLE_HEIGHT_CHECK,
            MetricConvention.MESH_INTERSECTION,
            MetricConvention.CAPSULE_COLLISION,
            MetricConvention.HOIDINI_FLOOR_METRICS,
            MetricConvention.CLOSD_PENETRATION,
            MetricConvention.HOIDINI_MESH_PENETRATION,
            MetricConvention.CLOSD_MESH_PENETRATION,
        ]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        """Get default parameters for each convention."""
        defaults = {
            MetricConvention.SIMPLE_HEIGHT_CHECK: {
                "ground_height": 0.0,
                "check_joints": None,  # None = check all joints
            },
            MetricConvention.MESH_INTERSECTION: {
                "ground_height": 0.0,
                "mesh_vertices": None,  # Will use motion mesh if available
                "ground_normal": [0.0, 1.0, 0.0],  # Up vector
            },
            MetricConvention.CAPSULE_COLLISION: {
                "ground_height": 0.0,
                "capsule_radius": 0.02,  # 2cm radius
                "check_joints": ["left_foot", "right_foot", "left_hand", "right_hand"],
            },
            MetricConvention.HOIDINI_FLOOR_METRICS: {
                "floor_height": 0.0,
                "foot_joint_names": None,  # Uses automatic detection from skeleton
                "contact_threshold": 0.05,  # 50mm contact threshold (same as foot contact)
                "penetration_threshold": -0.01,  # -10mm penetration threshold (negative = below ground)
            },
            MetricConvention.CLOSD_PENETRATION: {
                "tolerance": 0.005,  # 5mm tolerance (from CLoSD) - for penetration calculation
                "contact_threshold": 0.05,  # 50mm contact threshold (same as foot contact)
                "penetration_threshold": -0.01,  # -10mm penetration threshold
            },
            MetricConvention.HOIDINI_MESH_PENETRATION: {
                "floor_height": 0.0,  # Floor level for HOIDINI
                "body_model": "smpl",  # Default to SMPL body model
                "contact_threshold": 0.05,  # 50mm contact threshold (same as foot contact)
                "penetration_threshold": -0.01,  # -10mm penetration threshold (negative = below ground)
            },
            MetricConvention.CLOSD_MESH_PENETRATION: {
                "tolerance": 0.005,  # 5mm tolerance (from CLoSD) - for penetration calculation
                "contact_threshold": 0.05,  # 50mm contact threshold (same as foot contact)
                "penetration_threshold": -0.01,  # -10mm penetration threshold
                "body_model": "smpl",  # Default to SMPL body model
            },
        }
        return defaults.get(convention, {})

    def _validate_params(self) -> None:
        """Validate parameters for the selected convention."""
        required_params = {
            MetricConvention.SIMPLE_HEIGHT_CHECK: ["ground_height"],
            MetricConvention.MESH_INTERSECTION: ["ground_height", "ground_normal"],
            MetricConvention.CAPSULE_COLLISION: ["ground_height", "capsule_radius"],
            MetricConvention.HOIDINI_FLOOR_METRICS: ["floor_height"],
            MetricConvention.CLOSD_PENETRATION: ["tolerance"],
            MetricConvention.HOIDINI_MESH_PENETRATION: ["floor_height", "body_model"],
            MetricConvention.CLOSD_MESH_PENETRATION: ["tolerance", "body_model"],
        }

        required = required_params.get(self.convention, [])
        missing = [param for param in required if param not in self.params]

        if missing:
            raise ValueError(f"Missing required parameters for {self.convention.value}: {missing}")

    def compute(self, motion: Motion) -> MetricResult:
        """Compute ground penetration metric using the specified convention.

        Args:
            motion: Motion object to analyze

        Returns:
            MetricResult with penetration measurements and metadata
        """
        positions = motion.positions  # Shape: (frames, joints, 3)
        num_frames, num_joints = positions.shape[:2]

        # Identify joints to check
        check_joints = self.params.get("check_joints", None)
        check_indices = self._get_check_indices(motion, check_joints, num_joints)

        # Compute penetration using the specified convention
        if self.convention == MetricConvention.SIMPLE_HEIGHT_CHECK:
            result = self._compute_simple_height_check(positions, check_indices, num_frames, motion)
        elif self.convention == MetricConvention.MESH_INTERSECTION:
            result = self._compute_mesh_intersection(motion, positions, check_indices, num_frames)
        elif self.convention == MetricConvention.CAPSULE_COLLISION:
            result = self._compute_capsule_collision(positions, check_indices, num_frames)
        elif self.convention == MetricConvention.HOIDINI_FLOOR_METRICS:
            result = self._compute_hoidini_floor_metrics(positions, motion)
        elif self.convention == MetricConvention.CLOSD_PENETRATION:
            result = self._compute_closd_penetration(positions, motion)
        elif self.convention == MetricConvention.HOIDINI_MESH_PENETRATION:
            result = self._compute_hoidini_mesh_penetration(motion)
        elif self.convention == MetricConvention.CLOSD_MESH_PENETRATION:
            result = self._compute_closd_mesh_penetration(motion)
        else:
            raise ValueError(f"Unsupported convention: {self.convention}")

        # Add metadata
        result.metadata.update(
            {
                "convention": self.convention.value,
                "parameters": self.params,
                "check_indices": check_indices,
                "num_checked_joints": len(check_indices),
            }
        )

        return result

    def _get_check_indices(self, motion: Motion, check_joints: list[str] | None, num_joints: int) -> list[int]:
        """Get indices of joints to check for penetration."""
        if check_joints is None:
            # Check all joints
            return list(range(num_joints))

        check_indices = []
        for joint_name in check_joints:
            try:
                idx = motion.skeleton.get_joint_index(joint_name)
                check_indices.append(idx)
            except ValueError:
                continue  # Joint not found, skip

        if not check_indices:
            # Fallback: check all joints
            check_indices = list(range(num_joints))

        return check_indices

    def _get_foot_indices_from_skeleton(self, motion: Motion) -> list[int]:
        """Get foot joint indices automatically from skeleton using common foot joint names.

        This method tries multiple common foot joint naming conventions to find
        foot joints in the skeleton, making it robust across different skeleton formats.

        Args:
            motion: Motion object with skeleton information

        Returns:
            List of foot joint indices

        Raises:
            ValueError: If no foot joints can be found in the skeleton
        """
        # Common foot joint naming conventions across different skeleton formats
        foot_joint_patterns = [
            # Primary patterns (most common)
            ["left_foot", "right_foot"],
            ["left_ankle", "right_ankle"],
            ["LeftFoot", "RightFoot"],
            ["LeftAnkle", "RightAnkle"],
            # SMPL/SMPL-H/SMPL-X specific
            ["L_Ankle", "R_Ankle"],
            ["l_ankle", "r_ankle"],
            # Motion capture systems
            ["LANK", "RANK"],  # Some BVH formats
            ["LAnkle", "RAnkle"],
            # Mixamo/game engine formats
            ["mixamorig:LeftFoot", "mixamorig:RightFoot"],
            ["mixamorig:LeftToeBase", "mixamorig:RightToeBase"],
            # Other common patterns
            ["left_toe", "right_toe"],
            ["LeftToe", "RightToe"],
        ]

        skeleton = motion.skeleton
        num_joints = len(skeleton.joint_names)

        # Try each pattern until we find one that works
        for pattern in foot_joint_patterns:
            foot_indices = []

            for foot_name in pattern:
                try:
                    idx = skeleton.get_joint_index(foot_name)
                    foot_indices.append(idx)
                except ValueError:
                    break  # This pattern doesn't work, try next

            # If we found all joints in this pattern, use it
            if len(foot_indices) == len(pattern):
                return foot_indices

        # Fallback: look for joints with "foot" or "ankle" in the name
        foot_indices = []
        joint_names_lower = [name.lower() for name in skeleton.joint_names]

        for i, name_lower in enumerate(joint_names_lower):
            if any(keyword in name_lower for keyword in ["foot", "ankle", "toe"]):
                foot_indices.append(i)

        # If we found some foot-like joints, try to pick left/right pairs
        if len(foot_indices) >= 2:
            # Sort by name to get consistent left/right ordering
            foot_indices.sort(key=lambda i: skeleton.joint_names[i])

            # Take first two (or up to 4 if we have left/right foot + ankle)
            foot_indices = foot_indices[: min(4, len(foot_indices))]
            return foot_indices

        # Last resort: use last two joints (often feet in hierarchical skeletons)
        if num_joints >= 2:
            return [num_joints - 2, num_joints - 1]
        elif num_joints == 1:
            return [0]
        else:
            raise ValueError(
                f"Cannot find foot joints in skeleton. "
                f"Joint names: {skeleton.joint_names}. "
                f"Please ensure skeleton contains recognizable foot joint names."
            )

    def _compute_simple_height_check(
        self, positions: np.ndarray, check_indices: list[int], num_frames: int, motion: Motion
    ) -> MetricResult:
        """Compute penetration using simple height comparison."""
        ground_height = self.params["ground_height"]
        contact_height_threshold = self.params.get("contact_height_threshold", 0.05)  # Default 5cm for contact

        # Get up-axis index based on motion coordinate system
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]

        # Calculate penetration depths (coordinate system agnostic)
        penetration_depths = np.maximum(0.0, ground_height - positions[:, check_indices, up_axis_idx])

        # Per-frame maximum penetration
        penetration_per_frame = np.max(penetration_depths, axis=1)

        # Per-joint average penetration
        penetration_per_joint = np.mean(penetration_depths, axis=0)

        # Frame average penetration
        frame_average = np.mean(penetration_depths, axis=1)

        # Sequence average
        sequence_average = float(np.mean(penetration_depths))

        # Calculate penetration ratio for foot joints only
        penetration_ratio = 0.0
        foot_joint_names = self.params.get("foot_joint_names", ["left_foot", "right_foot"])
        if foot_joint_names:
            foot_indices = []
            for joint_name in foot_joint_names:
                try:
                    foot_indices.append(motion.skeleton.joint_names.index(joint_name))
                except ValueError:
                    # Try common alternatives
                    alternatives = {
                        "left_foot": ["left_ankle", "LeftFoot", "LeftAnkle", "L_Ankle"],
                        "right_foot": ["right_ankle", "RightFoot", "RightAnkle", "R_Ankle"],
                    }
                    for alt_name in alternatives.get(joint_name, []):
                        try:
                            foot_indices.append(motion.skeleton.joint_names.index(alt_name))
                            break
                        except ValueError:
                            continue

            if foot_indices:
                # Track contact and penetration for feet only
                any_foot_contact = np.zeros(num_frames, dtype=bool)
                any_foot_penetration_during_contact = np.zeros(num_frames, dtype=bool)

                for foot_idx in foot_indices:
                    if foot_idx < positions.shape[1]:  # Ensure index is valid
                        foot_heights = positions[:, foot_idx, up_axis_idx]
                        in_contact = foot_heights <= (ground_height + contact_height_threshold)
                        penetrating = foot_heights < ground_height

                        any_foot_contact |= in_contact
                        any_foot_penetration_during_contact |= in_contact & penetrating

                total_contact_frames = np.sum(any_foot_contact)
                if total_contact_frames > 0:
                    penetration_ratio = np.sum(any_foot_penetration_during_contact) / total_contact_frames

        return MetricResult(
            per_frame=penetration_per_frame,
            per_joint=penetration_per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "detection_method": "simple_height_check",
                "total_contact_frames": int(np.sum(any_foot_contact)) if "any_foot_contact" in locals() else 0,
                "penetration_contact_frames": int(np.sum(any_foot_penetration_during_contact))
                if "any_foot_penetration_during_contact" in locals()
                else 0,
            },
            ratio=float(penetration_ratio),
        )

    def _compute_mesh_intersection(
        self, motion: Motion, positions: np.ndarray, check_indices: list[int], num_frames: int
    ) -> MetricResult:
        """Compute penetration using mesh intersection (placeholder implementation)."""
        # Note: This would require actual mesh data and intersection algorithms
        # For now, fall back to height-based check with a warning
        ground_height = self.params["ground_height"]

        # TODO: Implement actual mesh intersection
        # This is a placeholder that uses height-based detection
        penetration_depths = np.maximum(0.0, ground_height - positions[:, check_indices, 1])

        # Per-frame maximum penetration
        penetration_per_frame = np.max(penetration_depths, axis=1)

        # Per-joint average penetration
        penetration_per_joint = np.mean(penetration_depths, axis=0)

        # Frame average penetration
        frame_average = np.mean(penetration_depths, axis=1)

        # Sequence average
        sequence_average = float(np.mean(penetration_depths))

        return MetricResult(
            per_frame=penetration_per_frame,
            per_joint=penetration_per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "detection_method": "mesh_intersection_fallback",
                "warning": "Mesh intersection not implemented, using height-based fallback",
            },
        )

    def _compute_capsule_collision(
        self, positions: np.ndarray, check_indices: list[int], num_frames: int
    ) -> MetricResult:
        """Compute penetration using capsule collision detection."""
        ground_height = self.params["ground_height"]
        capsule_radius = self.params["capsule_radius"]

        # Calculate penetration depths considering capsule radius
        # Capsule center must be at least radius above ground
        effective_ground = ground_height + capsule_radius
        penetration_depths = np.maximum(0.0, effective_ground - positions[:, check_indices, 1])

        # Per-frame maximum penetration
        penetration_per_frame = np.max(penetration_depths, axis=1)

        # Per-joint average penetration
        penetration_per_joint = np.mean(penetration_depths, axis=0)

        # Frame average penetration
        frame_average = np.mean(penetration_depths, axis=1)

        # Sequence average
        sequence_average = float(np.mean(penetration_depths))

        return MetricResult(
            per_frame=penetration_per_frame,
            per_joint=penetration_per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "detection_method": "capsule_collision",
                "effective_ground_height": effective_ground,
            },
        )

    def _compute_hoidini_floor_metrics(self, positions: np.ndarray, motion: Motion) -> MetricResult:
        """Compute HOIDINI-style floor metrics.

        HOIDINI approach:
        - Automatically detects foot joints from skeleton
        - Uses only the lowest foot per frame
        - Computes penetration/floating distances and frame ratios
        - Multiple contact thresholds (5mm, 1cm, 1.5cm)
        """
        num_frames, num_joints = positions.shape[:2]
        floor_height = self.params.get("floor_height", 0.0)

        # Get foot indices dynamically from skeleton
        try:
            foot_indices = self._get_foot_indices_from_skeleton(motion)
        except ValueError as e:
            # Fallback to last two joints if automatic detection fails
            if num_joints >= 2:
                foot_indices = [num_joints - 2, num_joints - 1]
            else:
                raise ValueError(f"Cannot determine foot joints for HOIDINI floor metrics: {e}")

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]
        foot_heights = positions[:, foot_indices, up_axis_idx]  # Shape: (frames, 2)
        lower_foot_heights = np.min(foot_heights, axis=1) - floor_height  # Shape: (frames,)

        # Separate penetration and floating
        penetration_mask = lower_foot_heights < 0
        floating_mask = lower_foot_heights > 0

        penetration_heights = lower_foot_heights[penetration_mask]
        floating_heights = lower_foot_heights[floating_mask]

        # Distance metrics
        mean_penetration = abs(np.mean(penetration_heights)) if len(penetration_heights) > 0 else 0.0
        mean_floating = np.mean(floating_heights) if len(floating_heights) > 0 else 0.0

        # Frame ratio metrics with different thresholds
        total_frames = len(lower_foot_heights)
        thresholds = [0.005, 0.01, 0.015]  # 5mm, 1cm, 1.5cm

        # Calculate penetration per frame for compatibility
        penetration_per_frame = np.abs(np.minimum(lower_foot_heights, 0.0))

        # Calculate per-joint values (same for both feet in HOIDINI approach)
        penetration_per_joint = np.array([mean_penetration, mean_penetration])

        # Frame average
        frame_average = penetration_per_frame

        # Calculate contact and penetration ratio
        # For HOIDINI, we consider "contact" as being close to the floor (within threshold)
        contact_threshold = self.params.get("contact_threshold", 0.05)  # Default 50mm - same as foot contact
        penetration_threshold = self.params.get(
            "penetration_threshold", -0.01
        )  # Default -10mm below ground for consistent detection
        contact_frames = (
            np.abs(lower_foot_heights) <= contact_threshold
        )  # Either touching or slightly penetrating/floating
        penetration_frames_bool = lower_foot_heights < penetration_threshold  # Actually below penetration threshold
        penetration_events_bool = lower_foot_heights < 0  # Any negative distance (penetration event)
        penetration_during_contact = contact_frames & penetration_frames_bool  # Penetrating while in contact

        total_contact_frames = np.sum(contact_frames)
        penetration_contact_frames = np.sum(penetration_during_contact)
        penetration_ratio = penetration_contact_frames / total_contact_frames if total_contact_frames > 0 else 0.0

        # Aggregate results with threshold-based metrics
        metrics_dict = {
            "mean_penetration_floor": mean_penetration,
            "floating_floor": mean_floating,
            "total_contact_frames": int(total_contact_frames),
            "penetration_contact_frames": int(penetration_contact_frames),
        }

        # Add threshold-based frame ratios
        for threshold in thresholds:
            threshold_str = f"{int(threshold * 1000)}mm"
            penetration_frames = np.sum(lower_foot_heights < -threshold)
            floating_frames = np.sum(lower_foot_heights > threshold)

            metrics_dict[f"penetration_ratio_{threshold_str}"] = penetration_frames / total_frames
            metrics_dict[f"floating_ratio_{threshold_str}"] = floating_frames / total_frames

        # Sequence average (overall penetration severity)
        sequence_average = mean_penetration

        return MetricResult(
            per_frame=penetration_per_frame,
            per_joint=penetration_per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "reference": "HOIDINI paper implementation",
                "aggregation": "lowest_foot_per_frame",
                "thresholds": thresholds,
                "detailed_metrics": metrics_dict,
                "foot_indices": foot_indices,
                "detected_foot_joints": [motion.skeleton.joint_names[i] for i in foot_indices],
                # Frame-by-frame data for visualization
                "contact_frames_bool": contact_frames.tolist(),
                "penetration_frames_bool": penetration_frames_bool.tolist(),
                "penetration_events_bool": penetration_events_bool.tolist(),
                "foot_heights": lower_foot_heights.tolist(),
                "contact_threshold": contact_threshold,
                "penetration_threshold": penetration_threshold,
            },
            ratio=float(penetration_ratio),
        )

    @staticmethod
    def _load_body_model_config(body_model: str) -> dict:
        """Load body model configuration from comprehensive config file.

        Args:
            body_model: Name of the body model (e.g., "smpl")

        Returns:
            Body model configuration dictionary

        Raises:
            ValueError: If body model configuration is not found
        """
        # Try to find the comprehensive config file
        config_paths = [
            Path(__file__).resolve().parents[4] / "configs" / "evaluation" / "comprehensive.yaml",
            Path("configs/evaluation/comprehensive.yaml"),
        ]

        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            raise ValueError(
                f"Could not find comprehensive.yaml config file. Searched paths: {[str(p) for p in config_paths]}"
            )

        # Load the config file
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config file {config_path}: {e}")

        # Extract body model configurations
        body_models = config_data.get("body_models", {})
        if body_model not in body_models:
            available = list(body_models.keys())
            raise ValueError(f"Body model '{body_model}' not found in configuration. Available models: {available}")

        return body_models[body_model]

    def _get_foot_vertex_indices(self, body_model: str, motion: Motion = None) -> tuple[list[int], list[int]]:
        """Get foot vertex indices from body model configuration.

        Args:
            body_model: Name of the body model (e.g., "smpl")
            motion: Optional motion object for vertex count validation

        Returns:
            Tuple of (left_foot_vertices, right_foot_vertices) lists
        """
        body_model_config = self._load_body_model_config(body_model)
        foot_vertices = body_model_config.get("foot_vertices", {})
        left_foot_indices = foot_vertices.get("left_foot", [])
        right_foot_indices = foot_vertices.get("right_foot", [])

        if not left_foot_indices or not right_foot_indices:
            raise ValueError(
                f"Missing foot vertex indices in body model '{body_model}' configuration. "
                f"Expected 'foot_vertices.left_foot' and 'foot_vertices.right_foot' lists."
            )

        # Validate vertex indices against model bounds if motion is provided
        if motion is not None and motion.body_model_path is not None:
            try:
                vertices = motion._get_smpl_vertices()
                total_vertices = vertices.shape[1] if len(vertices.shape) > 1 else 0

                # Check all vertex indices are within bounds
                all_indices = left_foot_indices + right_foot_indices
                invalid_indices = [idx for idx in all_indices if idx < 0 or idx >= total_vertices]

                if invalid_indices:
                    raise ValueError(
                        f"Invalid vertex indices for body model '{body_model}': {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}. "
                        f"Model has {total_vertices} vertices (valid range: 0-{total_vertices - 1})."
                    )

                # Check for duplicate indices between left and right foot
                left_set = set(left_foot_indices)
                right_set = set(right_foot_indices)
                overlap = left_set & right_set
                if overlap:
                    # If there's significant overlap, it might be a unified foot vertex set
                    if len(overlap) == len(left_foot_indices) == len(right_foot_indices):
                        print(
                            f"Info: '{body_model}' uses unified foot vertex set ({len(left_foot_indices)} vertices for both feet)"
                        )
                    else:
                        print(
                            f"Warning: Found {len(overlap)} overlapping vertex indices between left and right foot in '{body_model}' configuration"
                        )

            except Exception as e:
                print(f"Warning: Could not validate vertex indices for body model '{body_model}': {e}")

        return left_foot_indices, right_foot_indices

    def _compute_hoidini_mesh_penetration(self, motion: Motion) -> MetricResult:
        """Compute HOIDINI-style ground penetration using mesh vertices instead of joints.

        HOIDINI approach adapted for mesh:
        - Find lowest foot vertex per frame (instead of lowest joint)
        - Penetration detection based on vertex height below floor
        - Apply multiple thresholds for analysis
        """
        if motion.body_model_path is None:
            raise ValueError("Mesh-based penetration detection requires body_model_path for SMPL forward kinematics")

        # Get SMPL vertices for all frames
        vertices = motion._get_smpl_vertices()
        num_frames = vertices.shape[0]

        # Get foot vertex indices from body model configuration
        body_model = self.params["body_model"]
        left_foot_vertices, right_foot_vertices = self._get_foot_vertex_indices(body_model, motion)
        all_foot_vertices = left_foot_vertices + right_foot_vertices

        floor_height = self.params["floor_height"]

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]

        # Extract foot vertex heights
        foot_vertex_positions = vertices[:, all_foot_vertices, :]  # Shape: (frames, 42, 3)
        foot_vertex_heights = foot_vertex_positions[:, :, up_axis_idx]  # Shape: (frames, 42)

        # Find lowest foot vertex per frame (HOIDINI-style)
        lowest_foot_heights = np.min(foot_vertex_heights, axis=1)  # Shape: (frames,)

        # Compute penetration amounts (heights below floor)
        penetration_amounts = np.maximum(floor_height - lowest_foot_heights, 0.0)

        # HOIDINI-style multi-threshold analysis
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]  # 1mm, 5mm, 1cm, 2cm, 5cm
        metrics_dict = {}

        # Calculate basic penetration metrics
        total_frames = num_frames
        penetration_frames = np.sum(penetration_amounts > 0)
        mean_penetration = np.mean(penetration_amounts)
        max_penetration = np.max(penetration_amounts)

        # Contact and penetration for ratio calculation
        contact_threshold = self.params.get("contact_threshold", 0.05)  # Default 50mm - same as foot contact
        penetration_threshold = self.params.get(
            "penetration_threshold", -0.01
        )  # Default -10mm - negative means below ground
        contact_frames = lowest_foot_heights <= (floor_height + contact_threshold)  # Within contact threshold of floor
        penetration_frames_bool = lowest_foot_heights < (
            floor_height + penetration_threshold
        )  # Actually below penetration threshold
        penetration_events_bool = lowest_foot_heights < floor_height  # Any negative distance (penetration event)
        penetration_contact_frames = contact_frames & penetration_frames_bool  # Penetrating while in contact

        total_contact_frames = np.sum(contact_frames)
        penetration_ratio = (
            np.sum(penetration_contact_frames) / total_contact_frames if total_contact_frames > 0 else 0.0
        )

        metrics_dict.update(
            {
                "penetration_frames": penetration_frames,
                "mean_penetration_vertex": mean_penetration,
                "max_penetration_vertex": max_penetration,
                "total_contact_frames": int(total_contact_frames),
                "penetration_contact_frames": int(np.sum(penetration_contact_frames)),
            }
        )

        # Add threshold-based frame ratios
        for threshold in thresholds:
            threshold_str = f"{int(threshold * 1000)}mm"
            penetration_frames_thresh = np.sum(penetration_amounts > threshold)
            metrics_dict[f"penetration_ratio_{threshold_str}"] = penetration_frames_thresh / total_frames

        # Sequence average (overall penetration severity)
        sequence_average = mean_penetration

        # Per-joint penetration (distribute across foot vertices)
        per_joint = np.full(len(all_foot_vertices), sequence_average / len(all_foot_vertices))

        # Frame average
        frame_average = penetration_amounts

        return MetricResult(
            per_frame=penetration_amounts,
            per_joint=per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "reference": "HOIDINI paper implementation (mesh-based)",
                "aggregation": "lowest_foot_vertex_per_frame",
                "thresholds": thresholds,
                "detailed_metrics": metrics_dict,
                "foot_vertex_count": len(all_foot_vertices),
                # Frame-by-frame data for visualization
                "contact_frames_bool": contact_frames.tolist(),
                "penetration_frames_bool": penetration_frames_bool.tolist(),
                "penetration_events_bool": penetration_events_bool.tolist(),
                "foot_heights": lowest_foot_heights.tolist(),
                "contact_threshold": contact_threshold,
                "penetration_threshold": penetration_threshold,
            },
            ratio=float(penetration_ratio),
        )

    def _compute_closd_mesh_penetration(self, motion: Motion) -> MetricResult:
        """Compute CLoSD-style ground penetration using mesh vertices instead of joints.

        CLoSD approach adapted for mesh:
        - Find lowest FOOT vertex per frame (from foot vertex indices)
        - Apply tolerance before calculating penetration
        - Convert to millimeters for output
        """
        import torch

        if motion.body_model_path is None:
            raise ValueError("Mesh-based penetration detection requires body_model_path for SMPL forward kinematics")

        # Get SMPL vertices for all frames
        vertices = motion._get_smpl_vertices()
        num_frames = vertices.shape[0]
        num_vertices = vertices.shape[1]

        tolerance = self.params["tolerance"]  # Original CLoSD tolerance for penetration calculation
        contact_threshold = self.params.get("contact_threshold", 0.05)  # 50mm contact threshold
        penetration_threshold = self.params.get("penetration_threshold", -0.01)  # -10mm penetration threshold
        body_model = self.params["body_model"]  # Validate body model parameter

        # Get foot vertex indices from body model configuration
        left_foot_vertices, right_foot_vertices = self._get_foot_vertex_indices(body_model, motion)
        all_foot_vertices = left_foot_vertices + right_foot_vertices

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]

        # Convert to torch for compatibility with CLoSD implementation
        vertices_torch = torch.from_numpy(vertices).float()

        # Get heights of FOOT vertices only: shape (frames, foot_vertices)
        foot_vertex_heights = vertices_torch[:, all_foot_vertices, up_axis_idx]  # B x n_foot_vertices x L equivalent

        # Find lowest FOOT vertex height per frame: shape (frames,)
        lowest_heights = foot_vertex_heights.min(dim=1)[0]  # B x L equivalent

        penetration_list = []
        contact_frames = []
        penetration_frames = []
        penetration_events = []  # Track all negative distance events

        for frame_idx in range(num_frames):
            lowest_height = lowest_heights[frame_idx].item()

            # CLoSD formula: (lowest_height + tolerance).clamp_max(0) * 1000 * -1
            penetration_value = max(lowest_height + tolerance, 0.0) * -1000.0 if lowest_height + tolerance < 0 else 0.0

            # Track contact and penetration using proper thresholds
            is_contact = lowest_height <= contact_threshold  # Within 50mm contact threshold
            is_penetrating = lowest_height < penetration_threshold  # Below -10mm penetration threshold
            is_negative_distance = lowest_height < 0  # Any negative distance (penetration event)

            contact_frames.append(is_contact)
            penetration_frames.append(is_penetrating)
            penetration_events.append(is_negative_distance)
            penetration_list.append(penetration_value)

        penetration_per_frame = np.array(penetration_list)
        contact_frames = np.array(contact_frames)
        penetration_frames = np.array(penetration_frames)
        penetration_events = np.array(penetration_events)

        # Calculate ratio (percentage of contact frames with penetration)
        total_contact_frames = np.sum(contact_frames)
        penetration_contact_frames = np.sum(contact_frames & penetration_frames)
        penetration_ratio = penetration_contact_frames / total_contact_frames if total_contact_frames > 0 else 0.0

        # Aggregate results (CLoSD style returns mean across all frames)
        sequence_average = float(np.mean(penetration_per_frame))

        # For per_joint, we can't easily determine which vertex contributed to each frame's minimum
        # So we distribute the penetration evenly across foot vertices
        per_joint = np.full(len(all_foot_vertices), sequence_average / len(all_foot_vertices))

        # Frame average is same as per_frame for this metric
        frame_average = penetration_per_frame

        return MetricResult(
            per_frame=penetration_per_frame,
            per_joint=per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "reference": "CLoSD/PhysDiff implementation (mesh-based)",
                "source": "https://github.com/GuyTevet/CLoSD",
                "aggregation": "lowest_foot_vertex_per_frame",
                "measurement": "penetration_depth_mm",
                "tolerance": tolerance,
                "total_contact_frames": int(total_contact_frames),
                "penetration_contact_frames": int(penetration_contact_frames),
                "foot_vertex_count": len(all_foot_vertices),
                # Frame-by-frame data for visualization
                "contact_frames_bool": contact_frames.tolist(),
                "penetration_frames_bool": penetration_frames.tolist(),
                "penetration_events_bool": penetration_events.tolist(),
                "foot_heights": lowest_heights.tolist(),
                "contact_threshold": contact_threshold,
                "penetration_threshold": penetration_threshold,
                "margin": tolerance,
            },
            ratio=float(penetration_ratio),
        )

    def _compute_closd_penetration(self, positions: np.ndarray, motion: Motion) -> MetricResult:
        """Compute CLoSD-style ground penetration metric.

        Based on CLoSD implementation:
        https://github.com/GuyTevet/CLoSD/blob/d8950bdb78312fcd2844f336b3fc079e7368c87f/closd/diffusion_planner/data_loaders/humanml/utils/metrics.py#L212

        CLoSD approach:
        - Find lowest joint height per frame across all joints
        - Apply tolerance (5mm default)
        - Clamp penetration values (only negative values below tolerance count)
        - Convert to millimeters for output
        """
        import torch

        num_frames, num_joints = positions.shape[:2]
        tolerance = self.params["tolerance"]  # Original CLoSD tolerance for penetration calculation
        contact_threshold = self.params.get("contact_threshold", 0.05)  # 50mm contact threshold
        penetration_threshold = self.params.get("penetration_threshold", -0.01)  # -10mm penetration threshold

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]

        # Convert to torch for compatibility with CLoSD implementation
        positions_torch = torch.from_numpy(positions).float()

        # Get heights of all joints: shape (frames, joints)
        joint_heights = positions_torch[:, :, up_axis_idx]  # B x n_joints x L equivalent

        # Find lowest joint height per frame: shape (frames,)
        lowest_heights = joint_heights.min(dim=1)[0]  # B x L equivalent

        penetration_list = []
        contact_frames = []
        penetration_frames = []
        penetration_events = []  # Track all negative distance events

        for frame_idx in range(num_frames):
            lowest_height = lowest_heights[frame_idx].item()

            # CLoSD formula: (lowest_height + tolerance).clamp_max(0) * 1000 * -1
            penetration_value = max(lowest_height + tolerance, 0.0) * -1000.0 if lowest_height + tolerance < 0 else 0.0

            # Track contact and penetration using proper thresholds
            is_contact = lowest_height <= contact_threshold  # Within 50mm contact threshold
            is_penetrating = lowest_height < penetration_threshold  # Below -10mm penetration threshold
            is_negative_distance = lowest_height < 0  # Any negative distance (penetration event)

            contact_frames.append(is_contact)
            penetration_frames.append(is_penetrating)
            penetration_events.append(is_negative_distance)
            penetration_list.append(penetration_value)

        penetration_per_frame = np.array(penetration_list)
        contact_frames = np.array(contact_frames)
        penetration_frames = np.array(penetration_frames)
        penetration_events = np.array(penetration_events)

        # Calculate ratio (percentage of contact frames with penetration)
        total_contact_frames = np.sum(contact_frames)
        penetration_contact_frames = np.sum(contact_frames & penetration_frames)
        penetration_ratio = penetration_contact_frames / total_contact_frames if total_contact_frames > 0 else 0.0

        # Aggregate results (CLoSD style returns mean across all frames)
        sequence_average = float(np.mean(penetration_per_frame))

        # For per_joint, we can't easily determine which joint contributed to each frame's minimum
        # So we distribute the penetration evenly across all joints
        per_joint = np.full(num_joints, sequence_average / num_joints)

        # Frame average is same as per_frame for this metric
        frame_average = penetration_per_frame

        return MetricResult(
            per_frame=penetration_per_frame,
            per_joint=per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            metadata={
                "reference": "CLoSD/PhysDiff implementation",
                "source": "https://github.com/GuyTevet/CLoSD",
                "aggregation": "lowest_joint_per_frame",
                "measurement": "penetration_depth_mm",
                "tolerance": tolerance,
                "total_contact_frames": int(total_contact_frames),
                "penetration_contact_frames": int(penetration_contact_frames),
                # Frame-by-frame data for visualization
                "contact_frames_bool": contact_frames.tolist(),
                "penetration_frames_bool": penetration_frames.tolist(),
                "penetration_events_bool": penetration_events.tolist(),
                "foot_heights": lowest_heights.tolist(),
                "contact_threshold": contact_threshold,
                "penetration_threshold": penetration_threshold,
                "margin": tolerance,
            },
            ratio=float(penetration_ratio),
        )


# Register the metric
MetricRegistry.register("ground_penetration", GroundPenetrationMetric)
