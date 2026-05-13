"""Foot skating metric implementations following different conventions.

This module implements approaches to detect foot skating artifacts
in motion data, supporting different research standards.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ...core.motion import Motion
from ..base_metric import EvalMetric, MetricConvention, MetricRegistry, MetricResult


class FootSkatingMetric(EvalMetric):
    """Foot skating detection with multiple implementation conventions.

    Foot skating occurs when feet slide along the ground instead of lifting
    during locomotion. Different conventions use various methods to detect
    contact and measure skating violations.

    Supported Conventions:
    - HOIDINI_FOOT_SKATING: HOIDINI paper implementation (preserved for compatibility)
    - CLOSD_FOOT_SLIDING: CLoSD/PhysDiff-based implementation with enhanced detection

    References:
    - HOIDINI:Based on https://github.com/hoidini/HOIDiNi implementation
    - CLoSD: Based on https://github.com/GuyTevet/CLoSD implementation
    """

    @classmethod
    def get_supported_conventions(cls) -> list[MetricConvention]:
        """Get list of supported conventions for foot skating detection."""
        return [
            MetricConvention.HOIDINI_FOOT_SKATING,
            MetricConvention.CLOSD_FOOT_SLIDING,
            MetricConvention.GMD_SMOOTHED_SKATING,
            MetricConvention.HOIDINI_MESH_SKATING,
            MetricConvention.CLOSD_MESH_SLIDING,
        ]

    @classmethod
    def get_default_params(cls, convention: MetricConvention) -> dict[str, Any]:
        """Get default parameters for each convention."""
        defaults = {
            MetricConvention.HOIDINI_FOOT_SKATING: {
                "contact_threshold": 0.01,  # Original HOIDINI: 1cm threshold (0.01m)
                "skating_threshold": 0.001,  # Original: 1mm displacement = skating
                "foot_joint_names": None,  # Uses automatic detection from skeleton
            },
            MetricConvention.CLOSD_FOOT_SLIDING: {
                "margin": 0.005,  # Original CLoSD: 5mm contact margin (0.005m)
                "skating_threshold": 0.001,  # Original: 1mm displacement = skating
                "foot_joint_names": ["left_foot", "right_foot", "left_ankle", "right_ankle"],
            },
            MetricConvention.GMD_SMOOTHED_SKATING: {
                "thresh_height": 0.05,  # Original GMD: 50mm contact height (0.05m)
                "thresh_vel": 0.50,  # Original GMD: 50cm/s velocity threshold (0.5m/s)
                "avg_window": 5,  # Original GMD: 5-frame smoothing window
                "foot_joint_names": ["left_foot", "right_foot"],
            },
            MetricConvention.HOIDINI_MESH_SKATING: {
                "contact_threshold": 0.01,  # Original HOIDINI: 1cm threshold (0.01m)
                "skating_threshold": 0.001,  # Original: 1mm displacement = skating
                "body_model": "smpl",  # Default to SMPL body model
            },
            MetricConvention.CLOSD_MESH_SLIDING: {
                "margin": 0.005,  # Original CLoSD: 5mm margin (0.005m)
                "skating_threshold": 0.001,  # Original: 1mm displacement = skating
                "body_model": "smpl",  # Default to SMPL body model
            },
        }
        return defaults.get(convention, {})

    def _validate_params(self) -> None:
        """Validate parameters for the selected convention."""
        required_params = {
            MetricConvention.HOIDINI_FOOT_SKATING: ["contact_threshold", "skating_threshold"],
            MetricConvention.CLOSD_FOOT_SLIDING: ["margin", "skating_threshold"],
            MetricConvention.GMD_SMOOTHED_SKATING: ["thresh_height", "thresh_vel", "avg_window"],
            MetricConvention.HOIDINI_MESH_SKATING: ["contact_threshold", "skating_threshold", "body_model"],
            MetricConvention.CLOSD_MESH_SLIDING: ["margin", "skating_threshold", "body_model"],
        }

        required = required_params.get(self.convention, [])
        missing = [param for param in required if param not in self.params]

        if missing:
            raise ValueError(f"Missing required parameters for {self.convention.value}: {missing}")

    def compute(self, motion: Motion) -> MetricResult:
        """Compute foot skating metric using the specified convention.

        Args:
            motion: Motion object to analyze

        Returns:
            MetricResult with skating measurements and metadata
        """
        # Get joint positions
        positions = motion.positions  # Shape: (frames, joints, 3)
        num_frames, num_joints = positions.shape[:2]

        # Compute skating using the specified convention
        if self.convention == MetricConvention.HOIDINI_FOOT_SKATING:
            result = self._compute_hoidini_foot_skating(positions, motion)
        elif self.convention == MetricConvention.CLOSD_FOOT_SLIDING:
            result = self._compute_closd_foot_sliding(positions, motion)
        elif self.convention == MetricConvention.GMD_SMOOTHED_SKATING:
            result = self._compute_gmd_smoothed_skating(positions, motion)
        elif self.convention == MetricConvention.HOIDINI_MESH_SKATING:
            result = self._compute_hoidini_mesh_skating(motion)
        elif self.convention == MetricConvention.CLOSD_MESH_SLIDING:
            result = self._compute_closd_mesh_sliding(motion)
        else:
            raise ValueError(f"Unsupported convention: {self.convention}")

        # Add metadata
        result.metadata.update(
            {
                "convention": self.convention.value,
                "parameters": self.params,
            }
        )

        return result

    def _get_foot_indices(self, motion: Motion, foot_joint_names: list[str], num_joints: int) -> list[int]:
        """Get indices of foot joints in the skeleton."""
        foot_indices = []

        for foot_name in foot_joint_names:
            try:
                idx = motion.skeleton.get_joint_index(foot_name)
                foot_indices.append(idx)
            except ValueError:
                continue  # Joint not found, skip

        if not foot_indices:
            # Fallback: use last two joints
            foot_indices = [num_joints - 2, num_joints - 1] if num_joints >= 2 else [0]

        return foot_indices

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

    def _compute_hoidini_foot_skating(self, positions: np.ndarray, motion: Motion) -> MetricResult:
        """Compute HOIDINI-style foot skating metric.

        HOIDINI approach:
        - Automatically detects foot joints from skeleton
        - Uses only the lowest foot per frame
        - Contact detection based on height threshold
        - Measures XY displacement when both frames have foot below threshold

        Original HOIDINI implementation used contact_threshold=0.01m (1cm).
        Results are reported in millimeters for consistency with other metrics.
        """
        num_frames, num_joints = positions.shape[:2]
        contact_threshold = self.params["contact_threshold"]
        skating_threshold_mm = self.params["skating_threshold"] * 1000  # Convert to mm

        # Get foot indices dynamically from skeleton
        try:
            foot_indices = self._get_foot_indices_from_skeleton(motion)
        except ValueError as e:
            # Fallback to last two joints if automatic detection fails
            if num_joints >= 2:
                foot_indices = [num_joints - 2, num_joints - 1]
            else:
                raise ValueError(f"Cannot determine foot joints for HOIDINI foot skating: {e}")

        # Extract foot positions
        foot_positions = positions[:, foot_indices, :]  # Shape: (frames, 2, 3)

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]
        foot_heights = foot_positions[:, :, up_axis_idx]  # Shape: (frames, 2)

        # Horizontal plane indices (coordinate system agnostic)
        if motion.up_axis.lower() == "y":
            horizontal_indices = [0, 2]  # XZ plane
        elif motion.up_axis.lower() == "z":
            horizontal_indices = [0, 1]  # XY plane
        else:  # X-up
            horizontal_indices = [1, 2]  # YZ plane

        skating_list = []
        contact_frames = []
        skating_frames = []
        contact_heights = []  # Track height of contact foot for each frame

        for t in range(num_frames - 1):
            current_heights = foot_heights[t]
            contact_idx_local = torch.argmin(torch.from_numpy(current_heights)).item()
            current_height = current_heights[contact_idx_local]
            contact_heights.append(current_height)  # Store the lowest foot height
            next_height = foot_heights[t + 1, contact_idx_local]

            # Both frames must have foot below threshold
            is_contact = current_height <= contact_threshold and next_height <= contact_threshold
            contact_frames.append(is_contact)

            if is_contact:
                current_pos = foot_positions[t, contact_idx_local, horizontal_indices]
                next_pos = foot_positions[t + 1, contact_idx_local, horizontal_indices]
                offset = next_pos - current_pos
                skate_distance = np.linalg.norm(offset) * 1000  # Convert to millimeters
                # Consider skating if displacement > minimal threshold (e.g., 1mm)
                is_skating = skate_distance > skating_threshold_mm
                skating_frames.append(is_skating)
            else:
                skate_distance = 0.0
                skating_frames.append(False)

            skating_list.append(skate_distance)

        # Pad last frame
        skating_list.append(0.0)
        contact_frames.append(False)
        skating_frames.append(False)

        skating_per_frame = np.array(skating_list)
        contact_frames = np.array(contact_frames)
        skating_frames = np.array(skating_frames)

        # Calculate ratio (percentage of contact frames with skating)
        total_contact_frames = np.sum(contact_frames)
        skating_contact_frames = np.sum(contact_frames & skating_frames)
        skating_ratio = skating_contact_frames / total_contact_frames if total_contact_frames > 0 else 0.0

        # Aggregate results (HOIDINI style)
        sequence_average = float(np.mean(skating_per_frame))
        per_joint = np.array([sequence_average, sequence_average])  # Same for both feet

        return MetricResult(
            per_frame=skating_per_frame,
            per_joint=per_joint,
            frame_average=skating_per_frame,
            sequence_average=sequence_average,
            metadata={
                "reference": "HOIDINI paper implementation",
                "aggregation": "lowest_foot_per_frame",
                "measurement": "position_displacement",
                "total_contact_frames": int(total_contact_frames),
                "skating_contact_frames": int(skating_contact_frames),
                "foot_indices": foot_indices,
                "detected_foot_joints": [motion.skeleton.joint_names[i] for i in foot_indices],
                "contact_frames_bool": contact_frames.tolist(),
                "skating_frames_bool": skating_frames.tolist(),
                "foot_heights": contact_heights,
                "contact_threshold": contact_threshold,
            },
            ratio=float(skating_ratio),
        )

    def _compute_closd_foot_sliding(self, positions: np.ndarray, motion: Motion) -> MetricResult:
        """Compute CLoSD-style foot sliding metric.

        Based on CLoSD implementation:
        https://github.com/GuyTevet/CLoSD/blob/d8950bdb78312fcd2844f336b3fc079e7368c87f/closd/diffusion_planner/data_loaders/humanml/utils/metrics.py#L232

        CLoSD approach:
        - Find lowest foot joint per frame
        - Check if lowest foot is below margin in current and next frame
        - If both frames in contact, measure horizontal displacement
        - Results reported in millimeters

        Original CLoSD implementation used margin=0.005m (5mm).
        """
        num_frames, num_joints = positions.shape[:2]
        margin = self.params["margin"]
        skating_threshold_mm = self.params["skating_threshold"] * 1000  # Convert to mm

        # Get foot joint indices
        foot_joint_names = self.params.get("foot_joint_names", ["left_foot", "right_foot"])
        foot_indices = self._get_foot_indices(motion, foot_joint_names, num_joints)

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]

        # Horizontal plane indices (coordinate system agnostic)
        if motion.up_axis.lower() == "y":
            horizontal_indices = [0, 2]  # XZ plane
        elif motion.up_axis.lower() == "z":
            horizontal_indices = [0, 1]  # XY plane
        else:  # X-up
            horizontal_indices = [1, 2]  # YZ plane

        skating_list = []
        contact_frames = []
        skating_frames = []
        contact_heights = []  # Track contact foot heights for visualization

        # Convert to torch for compatibility with CLoSD implementation
        positions_torch = torch.from_numpy(positions).float()

        for t in range(num_frames - 1):
            # Get heights of FOOT joints at current frame (not all joints like original CLoSD)
            foot_heights = positions_torch[t, foot_indices, up_axis_idx]

            # Find the foot joint with minimum height
            contact_foot_idx = torch.argmin(foot_heights).item()
            contact_idx = foot_indices[contact_foot_idx]  # Map back to global joint index

            current_height = foot_heights[contact_foot_idx].item()
            next_height = positions_torch[t + 1, contact_idx, up_axis_idx].item()
            contact_heights.append(current_height)  # Store for visualization

            # Both frames must have lowest joint below margin for contact
            is_contact = current_height <= margin and next_height <= margin
            contact_frames.append(is_contact)

            if is_contact:
                # Calculate horizontal displacement (following CLoSD exactly)
                current_pos = positions_torch[t, contact_idx, horizontal_indices]
                next_pos = positions_torch[t + 1, contact_idx, horizontal_indices]
                offset = next_pos - current_pos
                skate_distance = torch.norm(offset).item() * 1000  # Convert to millimeters like CLoSD
                is_skating = skate_distance > skating_threshold_mm
                skating_frames.append(is_skating)
            else:
                skate_distance = 0.0
                skating_frames.append(False)

            skating_list.append(skate_distance)

        # Pad last frame
        skating_list.append(0.0)
        contact_frames.append(False)
        skating_frames.append(False)

        skating_per_frame = np.array(skating_list)
        contact_frames = np.array(contact_frames)
        skating_frames = np.array(skating_frames)

        # Calculate ratio (percentage of contact frames with skating)
        total_contact_frames = np.sum(contact_frames)
        skating_contact_frames = np.sum(contact_frames & skating_frames)
        skating_ratio = skating_contact_frames / total_contact_frames if total_contact_frames > 0 else 0.0

        # Aggregate results (CLoSD style returns mean across all frames)
        sequence_average = float(np.mean(skating_per_frame))

        # For per_joint, distribute evenly across foot joints (since CLoSD uses global minimum)
        per_joint = np.full(len(foot_indices), sequence_average / len(foot_indices))

        return MetricResult(
            per_frame=skating_per_frame,
            per_joint=per_joint,
            frame_average=skating_per_frame,
            sequence_average=sequence_average,
            metadata={
                "reference": "CLoSD/PhysDiff implementation",
                "source": "https://github.com/GuyTevet/CLoSD",
                "aggregation": "lowest_joint_per_frame",
                "measurement": "horizontal_displacement_mm",
                "total_contact_frames": int(total_contact_frames),
                "skating_contact_frames": int(skating_contact_frames),
                "foot_indices": foot_indices,
                "contact_frames_bool": contact_frames.tolist(),
                "skating_frames_bool": skating_frames.tolist(),
                "foot_heights": contact_heights,
                "margin": margin,
            },
            ratio=float(skating_ratio),
        )

    def _compute_gmd_smoothed_skating(self, positions: np.ndarray, motion: Motion) -> MetricResult:
        """Compute GMD-style foot skating with velocity smoothing.

        Based on GMD implementation with velocity-based skating detection:
        https://github.com/korrawe/guided-motion-diffusion/blob/2f6264a9b793333556ef911981983082a1113049/data_loaders/humanml/scripts/motion_process.py#L269

        GMD approach:
        - Uses foot joint heights for contact detection
        - Applies 5-frame uniform smoothing to velocities to reduce noise
        - Measures foot velocity when in contact with ground
        - Thresholds both height and velocity for skating detection

        Original GMD: thresh_height=0.05m (50mm), thresh_vel=0.5m/s (50cm/s), 5-frame window
        """
        num_frames, num_joints = positions.shape[:2]
        thresh_height = self.params["thresh_height"]  # Contact height threshold (meters)
        thresh_vel = self.params["thresh_vel"]  # Velocity threshold (m/s)
        avg_window = self.params["avg_window"]  # Smoothing window size

        # Get foot joint indices
        foot_joint_names = self.params.get("foot_joint_names", ["left_foot", "right_foot"])
        foot_indices = self._get_foot_indices(motion, foot_joint_names, num_joints)

        # Extract foot positions
        foot_positions = positions[:, foot_indices, :]  # Shape: (frames, 2, 3)

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]
        foot_heights = foot_positions[:, :, up_axis_idx]  # Shape: (frames, 2)

        # Horizontal plane indices (coordinate system agnostic)
        if motion.up_axis.lower() == "y":
            horizontal_indices = [0, 2]  # XZ plane
        elif motion.up_axis.lower() == "z":
            horizontal_indices = [0, 1]  # XY plane
        else:  # X-up
            horizontal_indices = [1, 2]  # YZ plane

        # Calculate velocities using central differences (more robust)
        foot_velocities = np.zeros_like(foot_positions)
        foot_velocities[1:-1] = (foot_positions[2:] - foot_positions[:-2]) / (2.0 / motion.fps)
        foot_velocities[0] = foot_velocities[1]  # Forward difference for first frame
        foot_velocities[-1] = foot_velocities[-2]  # Backward difference for last frame

        # Apply smoothing to velocities (GMD's key innovation)
        smoothed_velocities = np.zeros_like(foot_velocities)
        half_window = avg_window // 2

        for i in range(num_frames):
            start_idx = max(0, i - half_window)
            end_idx = min(num_frames, i + half_window + 1)
            smoothed_velocities[i] = np.mean(foot_velocities[start_idx:end_idx], axis=0)

        # Calculate horizontal velocity magnitudes for each foot
        horizontal_velocities = smoothed_velocities[:, :, horizontal_indices]  # Shape: (frames, 2, 2)
        velocity_magnitudes = np.linalg.norm(horizontal_velocities, axis=2)  # Shape: (frames, 2)

        skating_list = []
        contact_frames = []
        skating_frames = []
        contact_heights = []  # Track contact foot heights for visualization

        for t in range(num_frames):
            current_heights = foot_heights[t]  # Shape: (2,)
            current_velocities = velocity_magnitudes[t]  # Shape: (2,)

            # Find foot closest to ground
            contact_foot_idx = np.argmin(current_heights)
            contact_height = current_heights[contact_foot_idx]
            contact_velocity = current_velocities[contact_foot_idx]
            contact_heights.append(contact_height)  # Store for visualization

            # Contact detection: foot below height threshold
            is_contact = contact_height <= thresh_height
            contact_frames.append(is_contact)

            if is_contact:
                # Skating detection: velocity above threshold when in contact
                velocity_ms = contact_velocity  # Already in m/s
                is_skating = velocity_ms > thresh_vel
                skating_frames.append(is_skating)
                skating_list.append(velocity_ms * 1000)  # Convert to mm/s for consistency
            else:
                skating_frames.append(False)
                skating_list.append(0.0)

        # Calculate results
        skating_array = np.array(skating_list)
        contact_array = np.array(contact_frames)
        skating_bool_array = np.array(skating_frames)

        # Ratio: percentage of contact frames that are skating
        contact_count = np.sum(contact_array)
        if contact_count > 0:
            skating_ratio = np.sum(skating_bool_array) / contact_count
        else:
            skating_ratio = 0.0

        # Per-joint data (2 feet)
        per_joint = np.column_stack([velocity_magnitudes[:, 0] * 1000, velocity_magnitudes[:, 1] * 1000])

        # Frame average (average across feet per frame)
        frame_average = np.mean(per_joint, axis=1)

        # Sequence average
        sequence_average = np.mean(skating_array)

        result = MetricResult(
            per_frame=skating_array,
            per_joint=per_joint,
            frame_average=frame_average,
            sequence_average=sequence_average,
            ratio=skating_ratio,
            metadata={
                "reference": "GMD implementation with velocity smoothing",
                "source": "https://github.com/korrawe/guided-motion-diffusion",
                "measurement": "smoothed_velocity_mm_per_s",
                "contact_frames": contact_count,
                "skating_frames": np.sum(skating_bool_array),
                "total_frames": num_frames,
                "smoothing_window": avg_window,
                "height_threshold_mm": thresh_height * 1000,
                "velocity_threshold_mm_per_s": thresh_vel * 1000,
                "contact_frames_bool": contact_array.tolist(),
                "skating_frames_bool": skating_bool_array.tolist(),
                "foot_heights": contact_heights,
                "thresh_height": thresh_height,
            },
        )

        return result

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

    def _compute_hoidini_mesh_skating(self, motion: Motion) -> MetricResult:
        """Compute HOIDINI-style foot skating using mesh vertices instead of joints.

        HOIDINI approach adapted for mesh:
        - Find lowest foot vertex per frame (instead of lowest joint)
        - Contact detection based on vertex height threshold
        - Measures horizontal displacement when vertices are in contact

        Original HOIDINI implementation used contact_threshold=0.01m (1cm).
        Results are reported in millimeters for consistency with other metrics.
        """
        if motion.body_model_path is None:
            raise ValueError("Mesh-based skating detection requires body_model_path for SMPL forward kinematics")

        # Get SMPL vertices for all frames
        vertices = motion._get_smpl_vertices()
        num_frames = vertices.shape[0]

        # Get foot vertex indices from body model configuration
        body_model = self.params["body_model"]
        left_foot_vertices, right_foot_vertices = self._get_foot_vertex_indices(body_model, motion)
        all_foot_vertices = left_foot_vertices + right_foot_vertices

        contact_threshold = self.params["contact_threshold"]
        skating_threshold_mm = self.params["skating_threshold"] * 1000  # Convert to mm

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]

        # Get SMPL vertices for all frames
        vertices = motion._get_smpl_vertices()  # Shape: (num_frames, num_vertices, 3)

        # Horizontal plane indices (coordinate system agnostic)
        if motion.up_axis.lower() == "y":
            horizontal_indices = [0, 2]  # XZ plane
        elif motion.up_axis.lower() == "z":
            horizontal_indices = [0, 1]  # XY plane
        else:  # X-up
            horizontal_indices = [1, 2]  # YZ plane

        skating_list = []
        contact_frames = []
        skating_frames = []
        contact_heights = []  # Track contact vertex heights for visualization

        for t in range(num_frames - 1):
            # Get heights of all foot vertices at current frame
            current_foot_vertices = vertices[t, all_foot_vertices, :]
            current_heights = current_foot_vertices[:, up_axis_idx]

            # Find the vertex with minimum height (contact vertex)
            contact_vertex_idx = torch.argmin(torch.from_numpy(current_heights)).item()
            actual_vertex_idx = all_foot_vertices[contact_vertex_idx]

            current_height = current_heights[contact_vertex_idx]
            next_height = vertices[t + 1, actual_vertex_idx, up_axis_idx]
            contact_heights.append(current_height)  # Store for visualization

            # Both frames must have vertex below threshold
            is_contact = current_height <= contact_threshold and next_height <= contact_threshold
            contact_frames.append(is_contact)

            if is_contact:
                current_pos = vertices[t, actual_vertex_idx, horizontal_indices]
                next_pos = vertices[t + 1, actual_vertex_idx, horizontal_indices]
                offset = next_pos - current_pos
                skate_distance = np.linalg.norm(offset) * 1000  # Convert to millimeters
                # Consider skating if displacement > minimal threshold (e.g., 1mm)
                is_skating = skate_distance > skating_threshold_mm
                skating_frames.append(is_skating)
            else:
                skate_distance = 0.0
                skating_frames.append(False)

            skating_list.append(skate_distance)

        # Pad last frame
        skating_list.append(0.0)
        contact_frames.append(False)
        skating_frames.append(False)

        skating_per_frame = np.array(skating_list)
        contact_frames = np.array(contact_frames)
        skating_frames = np.array(skating_frames)

        # Calculate ratio (percentage of contact frames with skating)
        total_contact_frames = np.sum(contact_frames)
        skating_contact_frames = np.sum(contact_frames & skating_frames)
        skating_ratio = skating_contact_frames / total_contact_frames if total_contact_frames > 0 else 0.0

        # Aggregate results (HOIDINI style)
        sequence_average = float(np.mean(skating_per_frame))
        per_joint = np.array([sequence_average, sequence_average])  # Same for both feet

        return MetricResult(
            per_frame=skating_per_frame,
            per_joint=per_joint,
            frame_average=skating_per_frame,
            sequence_average=sequence_average,
            metadata={
                "reference": "HOIDINI paper implementation (mesh-based)",
                "aggregation": "lowest_foot_vertex_per_frame",
                "measurement": "vertex_position_displacement",
                "total_contact_frames": int(total_contact_frames),
                "skating_contact_frames": int(skating_contact_frames),
                "foot_vertex_count": len(all_foot_vertices),
                "contact_frames_bool": contact_frames.tolist(),
                "skating_frames_bool": skating_frames.tolist(),
                "foot_heights": contact_heights,
                "contact_threshold": contact_threshold,
            },
            ratio=float(skating_ratio),
        )

    def _compute_closd_mesh_sliding(self, motion: Motion) -> MetricResult:
        """Compute CLoSD-style foot sliding using mesh vertices instead of joints.

        CLoSD approach adapted for mesh:
        - Find lowest vertex across ALL mesh vertices per frame
        - Contact detection based on vertex height margin
        - Measures horizontal displacement in millimeters

        Original CLoSD implementation used margin=0.005m (5mm).
        """
        if motion.body_model_path is None:
            raise ValueError("Mesh-based sliding detection requires body_model_path for SMPL forward kinematics")

        # Get SMPL vertices for all frames
        vertices = motion._get_smpl_vertices()
        num_frames = vertices.shape[0]
        num_vertices = vertices.shape[1]

        margin = self.params["margin"]
        skating_threshold_mm = self.params["skating_threshold"] * 1000  # Convert to mm
        body_model = self.params["body_model"]  # Validate body model parameter

        # Use coordinate system agnostic height (up_axis)
        up_axis_idx = {"x": 0, "y": 1, "z": 2}[motion.up_axis.lower()]

        # Horizontal plane indices (coordinate system agnostic)
        if motion.up_axis.lower() == "y":
            horizontal_indices = [0, 2]  # XZ plane
        elif motion.up_axis.lower() == "z":
            horizontal_indices = [0, 1]  # XY plane
        else:  # X-up
            horizontal_indices = [1, 2]  # YZ plane

        skating_list = []
        contact_frames = []
        skating_frames = []
        contact_heights = []  # Track contact vertex heights for visualization

        # Convert to torch for compatibility with CLoSD implementation
        vertices_torch = torch.from_numpy(vertices).float()

        for t in range(num_frames - 1):
            # Get heights of ALL vertices at current frame (CLoSD uses global minimum)
            all_heights = vertices_torch[t, :, up_axis_idx]

            # Find the vertex with minimum height (contact_vertex from CLoSD)
            contact_vertex_idx = torch.argmin(all_heights).item()

            current_height = all_heights[contact_vertex_idx].item()
            next_height = vertices_torch[t + 1, contact_vertex_idx, up_axis_idx].item()
            contact_heights.append(current_height)  # Store for visualization

            # Both frames must have lowest vertex below margin for contact
            is_contact = current_height <= margin and next_height <= margin
            contact_frames.append(is_contact)

            if is_contact:
                # Calculate horizontal displacement (following CLoSD exactly)
                current_pos = vertices_torch[t, contact_vertex_idx, horizontal_indices]
                next_pos = vertices_torch[t + 1, contact_vertex_idx, horizontal_indices]
                offset = next_pos - current_pos
                skate_distance = torch.norm(offset).item() * 1000  # Convert to millimeters like CLoSD
                is_skating = skate_distance > skating_threshold_mm
                skating_frames.append(is_skating)
            else:
                skate_distance = 0.0
                skating_frames.append(False)

            skating_list.append(skate_distance)

        # Pad last frame
        skating_list.append(0.0)
        contact_frames.append(False)
        skating_frames.append(False)

        skating_per_frame = np.array(skating_list)
        contact_frames = np.array(contact_frames)
        skating_frames = np.array(skating_frames)

        # Calculate ratio (percentage of contact frames with skating)
        total_contact_frames = np.sum(contact_frames)
        skating_contact_frames = np.sum(contact_frames & skating_frames)
        skating_ratio = skating_contact_frames / total_contact_frames if total_contact_frames > 0 else 0.0

        # Aggregate results (CLoSD style returns mean across all frames)
        sequence_average = float(np.mean(skating_per_frame))

        # For per_joint, use foot vertex information (since CLoSD uses global minimum)
        left_foot_vertices, right_foot_vertices = self._get_foot_vertex_indices(body_model, motion)
        per_joint = np.array([sequence_average / 2, sequence_average / 2])  # Split between feet

        return MetricResult(
            per_frame=skating_per_frame,
            per_joint=per_joint,
            frame_average=skating_per_frame,
            sequence_average=sequence_average,
            metadata={
                "reference": "CLoSD/PhysDiff implementation (mesh-based)",
                "source": "https://github.com/GuyTevet/CLoSD",
                "aggregation": "lowest_vertex_per_frame",
                "measurement": "horizontal_displacement_mm",
                "total_contact_frames": int(total_contact_frames),
                "skating_contact_frames": int(skating_contact_frames),
                "total_vertices": num_vertices,
                "contact_frames_bool": contact_frames.tolist(),
                "skating_frames_bool": skating_frames.tolist(),
                "foot_heights": contact_heights,
                "margin": margin,
            },
            ratio=float(skating_ratio),
        )


# Register the metric
MetricRegistry.register("foot_skating", FootSkatingMetric)
