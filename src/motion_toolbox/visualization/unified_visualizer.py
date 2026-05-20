"""Unified visualizer for motion data across multiple formats and body models.

This module provides a single entry point for visualizing motion data regardless
of the input format or desired output visualization.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..core.motion import Motion
from .compatibility import CompatibilityMatrix, CompatibilityResult, ConversionRequirement
from .constants import ALIGN_DEFAULT_FPS
from .format_detector import DetectedFormat, FormatDetector
from .mesh_visualizer import MeshVisualizer
from .skeleton_visualizer import SkeletonVisualizer

logger = logging.getLogger(__name__)


class VisualizerError(Exception):
    """Base exception for visualizer errors."""

    pass


class IncompatibleFormatError(VisualizerError):
    """Raised when format is incompatible with requested visualization."""

    pass


class Visualizer:
    """Unified visualizer for motion data.

    This class provides a single interface for visualizing motion data in any format
    (HumanML3D 263D, 272D, STMC 205D, ALIGN 232D, Motion objects, etc.) with any
    visualization mode (skeleton or mesh) and any body model (SMPL, MHR).

    The visualizer automatically:
    - Detects the input format
    - Checks compatibility with requested visualization
    - Performs necessary conversions
    - Generates output visualization

    Attributes:
        body_model_path: Path to body model files (for SMPL/SMPL-X)
        default_mode: Default visualization mode ('skeleton', 'mesh', 'auto')
        default_body_model: Default body model for mesh visualization
        on_incompatible: How to handle incompatible combinations

    Examples:
        >>> from motion_toolbox.visualization import Visualizer
        >>>
        >>> # Simple usage - auto everything
        >>> vis = Visualizer()
        >>> vis.visualize(motion_data, output_path="output.html")
        >>>
        >>> # Explicit control
        >>> vis = Visualizer(
        ...     body_model_path="~/.motion_toolbox/body_models",
        ...     default_mode='mesh',
        ...     default_body_model='smpl',
        ... )
        >>> vis.visualize(
        ...     motion_data,
        ...     output_path="output.html",
        ...     format='263d',  # Override auto-detection
        ...     mode='mesh',
        ...     body_model='smpl',
        ...     on_incompatible='auto_skeleton',  # Fall back to skeleton
        ... )
    """

    def __init__(
        self,
        body_model_path: str | None = None,
        default_mode: str = "auto",
        default_body_model: str = "auto",
        on_incompatible: str = "auto_skeleton",
    ):
        """Initialize the unified visualizer.

        Args:
            body_model_path: Path to body model files (for SMPL/SMPL-X)
                            Defaults to ~/.motion_toolbox/datasets/body_models_clean_pkl
            default_mode: Default visualization mode ('skeleton', 'mesh', 'auto')
            default_body_model: Default body model for mesh ('smpl', 'mhr', 'auto')
            on_incompatible: How to handle incompatible format/mode combinations:
                            - 'error': Raise IncompatibleFormatError
                            - 'auto_skeleton': Fall back to skeleton visualization (default)
        """
        self.format_detector = FormatDetector()
        self.compatibility_matrix = CompatibilityMatrix()

        # Set body model path
        if body_model_path is None:
            body_model_path = str(Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl")
        self.body_model_path = body_model_path

        # Validate and set defaults
        valid_modes = {"skeleton", "mesh", "auto"}
        if default_mode not in valid_modes:
            raise ValueError(f"default_mode must be one of {valid_modes}, got '{default_mode}'")
        self.default_mode = default_mode

        valid_body_models = {"smpl", "mhr", "auto"}
        if default_body_model not in valid_body_models:
            raise ValueError(f"default_body_model must be one of {valid_body_models}, got '{default_body_model}'")
        self.default_body_model = default_body_model

        valid_on_incompatible = {"error", "auto_skeleton"}
        if on_incompatible not in valid_on_incompatible:
            raise ValueError(f"on_incompatible must be one of {valid_on_incompatible}, got '{on_incompatible}'")
        self.on_incompatible = on_incompatible

        logger.info(
            f"Initialized Visualizer(mode={default_mode}, body_model={default_body_model}, "
            f"on_incompatible={on_incompatible})"
        )

    def visualize(
        self,
        data: Any,
        output_path: str | Path,
        format: str | None = None,
        mode: str | None = None,
        body_model: str | None = None,
        on_incompatible: str | None = None,
        fps: int = 20,
        title: str | None = None,
        **kwargs,
    ) -> Path:
        """Visualize motion data.

        This is the main entry point for visualization. It will:
        1. Detect the format (or use provided format hint)
        2. Check compatibility with requested mode/body_model
        3. Perform necessary conversions
        4. Generate visualization

        Args:
            data: Motion data in any supported format
            output_path: Path to save output HTML file
            format: Optional format hint (overrides auto-detection)
            mode: Visualization mode ('skeleton', 'mesh', 'auto')
                 Defaults to self.default_mode
            body_model: Body model for mesh visualization ('smpl', 'mhr', 'auto')
                       Defaults to self.default_body_model
            on_incompatible: How to handle incompatible combinations
                            Overrides self.on_incompatible if provided
            fps: Frames per second for animation
            title: Title for visualization
            **kwargs: Additional arguments passed to visualization backend

        Returns:
            Path to generated visualization file

        Raises:
            IncompatibleFormatError: If format is incompatible and on_incompatible='error'

        Examples:
            >>> vis = Visualizer()
            >>>
            >>> # Visualize HumanML3D data as skeleton
            >>> vis.visualize(humanml3d_data, "output.html")
            >>>
            >>> # Visualize as SMPL mesh with fallback
            >>> vis.visualize(
            ...     humanml3d_data,
            ...     "output.html",
            ...     mode='mesh',
            ...     body_model='smpl',
            ...     on_incompatible='auto_skeleton',
            ... )
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use defaults if not specified
        if mode is None:
            mode = self.default_mode
        if body_model is None:
            body_model = self.default_body_model
        if on_incompatible is None:
            on_incompatible = self.on_incompatible

        # Step 1: Detect format
        logger.info("Detecting motion data format...")
        detected_format = self.format_detector.detect(data, format_hint=format)
        logger.info(
            f"Detected format: {detected_format.format_type} "
            f"(confidence: {detected_format.confidence:.2f}, "
            f"frames: {detected_format.num_frames})"
        )

        # Step 2: Resolve AUTO mode
        if mode == "auto":
            mode, body_model = self.compatibility_matrix.get_recommended_mode(
                detected_format.format_type,
                detected_format.skeleton_type,
            )
            logger.info(f"AUTO mode resolved to: mode={mode}, body_model={body_model}")

        # Step 3: Check compatibility
        logger.info(f"Checking compatibility (mode={mode}, body_model={body_model})...")
        compatibility = self.compatibility_matrix.check_compatibility(
            detected_format.format_type,
            mode,
            body_model,
            detected_format.skeleton_type,
        )

        # Step 4: Handle incompatibility
        if not compatibility.compatible:
            return self._handle_incompatible(
                data,
                output_path,
                detected_format,
                mode,
                body_model,
                on_incompatible,
                fps,
                title,
                compatibility,
                **kwargs,
            )

        # Step 5: Show warnings if any
        if compatibility.warnings:
            for warning in compatibility.warnings:
                logger.warning(warning)

        # Step 6: Show conversion info
        if compatibility.conversion_required != ConversionRequirement.NONE:
            logger.info(
                f"Conversion required: {compatibility.conversion_required.value} "
                f"(estimated time: {compatibility.estimated_time:.1f}s)"
            )
            if compatibility.conversion_required == ConversionRequirement.SMPLIFY:
                logger.warning(
                    "SMPLify optimization is slow. Consider using skeleton visualization "
                    "or set on_incompatible='auto_skeleton' for faster results."
                )

        # Step 7: Route to appropriate visualization backend
        return self._route_visualization(
            data,
            output_path,
            detected_format,
            mode,
            body_model,
            fps,
            title,
            compatibility,
            **kwargs,
        )

    def _handle_incompatible(
        self,
        data: Any,
        output_path: Path,
        detected_format: DetectedFormat,
        mode: str,
        body_model: str | None,
        on_incompatible: str,
        fps: int,
        title: str | None,
        compatibility: CompatibilityResult,
        **kwargs,
    ) -> Path:
        """Handle incompatible format/mode/body_model combination."""
        error_msg = f"Format '{detected_format.format_type}' is not compatible with mode='{mode}'"
        if body_model:
            error_msg += f", body_model='{body_model}'"

        if compatibility.warnings:
            error_msg += "\nWarnings:\n  - " + "\n  - ".join(compatibility.warnings)

        if compatibility.suggestions:
            error_msg += "\nSuggestions:\n  - " + "\n  - ".join(compatibility.suggestions)

        if on_incompatible == "error":
            raise IncompatibleFormatError(error_msg)

        elif on_incompatible == "auto_skeleton":
            logger.warning(error_msg)
            logger.info("Falling back to skeleton visualization...")

            # Try skeleton mode
            skeleton_compat = self.compatibility_matrix.check_compatibility(
                detected_format.format_type,
                "skeleton",
                None,
                detected_format.skeleton_type,
            )

            if skeleton_compat.compatible:
                return self._route_visualization(
                    data,
                    output_path,
                    detected_format,
                    "skeleton",
                    None,
                    fps,
                    title,
                    skeleton_compat,
                    **kwargs,
                )
            else:
                raise IncompatibleFormatError(
                    f"Cannot visualize format '{detected_format.format_type}' even with skeleton mode"
                )

        else:
            raise ValueError(f"Unknown on_incompatible value: {on_incompatible}")

    def _route_visualization(
        self,
        data: Any,
        output_path: Path,
        detected_format: DetectedFormat,
        mode: str,
        body_model: str | None,
        fps: int,
        title: str | None,
        compatibility: CompatibilityResult,
        **kwargs,
    ) -> Path:
        """Route to appropriate visualization backend."""
        logger.info(
            f"Routing visualization: format={detected_format.format_type}, mode={mode}, body_model={body_model}"
        )

        # Phase 2: Skeleton visualization routing
        if mode == "skeleton":
            return self._visualize_skeleton(
                data,
                output_path,
                detected_format,
                fps,
                title,
                **kwargs,
            )

        # Phase 3: Mesh visualization routing
        if mode == "mesh":
            return self._visualize_mesh(
                data,
                output_path,
                detected_format,
                body_model,
                fps,
                title,
                compatibility,
                **kwargs,
            )

        raise ValueError(f"Unknown visualization mode: {mode}")

    def _normalize_coordinates(
        self,
        positions: np.ndarray,
        detected_format: DetectedFormat,
        source_data: Any = None,
    ) -> np.ndarray:
        """Normalize joint positions to canonical coordinate system.

        Canonical system (matches HumanML3D and Three.js):
        - Y-up: Positive Y is up (head above pelvis)
        - Z-forward: Character faces +Z direction (toward camera at Z=4)
        - X-right: Positive X is character's left (viewer's right)
        - Floor at Y=0: Lowest point touches ground

        Different input formats have different conventions:
        - HumanML3D (263D, 272D): Already in canonical system, just needs floor alignment
        - SMPL: Y-down, Z-backward -> needs Y-flip, Z-flip, floor alignment
        - Motion objects: Depends on skeleton origin, may need floor alignment

        Args:
            positions: Joint positions, shape (T, N, 3)
            detected_format: Format detection result (for skeleton_type)
            source_data: Original input data (for additional context)

        Returns:
            Normalized positions in canonical coordinate system
        """
        positions = positions.copy()  # Don't modify original
        num_joints = positions.shape[1]
        skeleton_type = detected_format.skeleton_type
        format_type = detected_format.format_type

        # SMPL coordinate system correction
        # SMPL uses: Y-down, Z-backward (character faces -Z)
        # We need: Y-up, Z-forward (character faces +Z)
        # NOTE: MHR already uses Y-up, so exclude it
        is_smpl_origin = (
            skeleton_type == "smpl"
            or (
                num_joints >= 24 and skeleton_type != "mhr" and format_type != "mhr"
            )  # SMPL has 24+ joints, but not MHR
            or (
                format_type == "motion"
                and isinstance(source_data, Motion)
                and hasattr(source_data, "skeleton")
                and getattr(source_data.skeleton, "name", "").lower() == "smpl"
            )
        )

        if is_smpl_origin and num_joints >= 24:
            logger.info("Applying SMPL coordinate normalization (Y-flip, Z-flip)")
            # Flip Y (Y-down -> Y-up)
            positions[:, :, 1] = -positions[:, :, 1]
            # Flip Z (Z-backward -> Z-forward)
            positions[:, :, 2] = -positions[:, :, 2]

        # Floor alignment: shift so lowest point touches Y=0
        min_y = positions[:, :, 1].min()
        if min_y != 0:
            positions[:, :, 1] -= min_y
            logger.debug(f"Floor alignment: shifted Y by {-min_y:.3f}")

        return positions

    def _visualize_skeleton(
        self,
        data: Any,
        output_path: Path,
        detected_format: DetectedFormat,
        fps: int,
        title: str | None,
        **kwargs,
    ) -> Path:
        """Visualize motion as animated skeleton.

        Extracts joint positions from the data and renders with SkeletonVisualizer.
        """
        logger.info(f"Extracting joint positions from {detected_format.format_type}...")

        # Extract joint positions based on format
        joint_positions = self._extract_joint_positions(data, detected_format)

        # Normalize to canonical coordinate system (Y-up, Z-forward, floor at Y=0)
        joint_positions = self._normalize_coordinates(joint_positions, detected_format, data)

        # Determine skeleton type and get appropriate kinematic chains
        skeleton_type = detected_format.skeleton_type
        num_joints = joint_positions.shape[1]

        # For Motion objects, try to use the skeleton's kinematic chains
        kinematic_chains = None
        chain_colors = None
        if detected_format.format_type == "motion" and isinstance(data, Motion):
            try:
                # Get chains from Motion's skeleton object
                chains = data.skeleton.get_kinematic_chains()
                kinematic_chains = [list(chain) for chain in chains]
                logger.info(f"Using kinematic chains from Motion skeleton ({len(kinematic_chains)} chains)")
            except Exception as e:
                logger.warning(f"Could not extract chains from Motion skeleton: {e}")

        # Otherwise, infer chains from skeleton type and joint count
        if kinematic_chains is None:
            kinematic_chains, chain_colors = self._get_kinematic_chains_for_skeleton(skeleton_type, num_joints)

        # Generate title if not provided
        if title is None:
            title = f"Skeleton Animation ({detected_format.format_type})"

        # Create visualizer with appropriate chains and render
        vis = SkeletonVisualizer(
            kinematic_chains=kinematic_chains,
            chain_colors=chain_colors,
        )
        return vis.visualize_positions(
            joint_positions,
            output_path,
            fps=fps,
            title=title,
            label=f"{detected_format.skeleton_type or 'skeleton'}",
            **kwargs,
        )

    def _get_kinematic_chains_for_skeleton(
        self,
        skeleton_type: str | None,
        num_joints: int,
    ) -> tuple[list[list[int]], list[str] | None]:
        """Get kinematic chains and colors for a skeleton type.

        Args:
            skeleton_type: Type of skeleton (humanml3d, smpl, mhr)
            num_joints: Number of joints in the skeleton

        Returns:
            Tuple of (kinematic_chains, chain_colors)
        """
        from .skeleton_visualizer import (
            DEFAULT_CHAIN_COLORS,
            DEFAULT_KINEMATIC_CHAINS,
            SMPL_KINEMATIC_CHAINS,
        )

        # HumanML3D (22 joints)
        if skeleton_type == "humanml3d" or num_joints == 22:
            logger.info("Using HumanML3D kinematic chains (22 joints)")
            return DEFAULT_KINEMATIC_CHAINS, DEFAULT_CHAIN_COLORS

        # SMPL (24 joints) - use explicit visual chains
        elif skeleton_type == "smpl" or num_joints == 24:
            logger.info("Using SMPL kinematic chains (24 joints)")
            return SMPL_KINEMATIC_CHAINS, DEFAULT_CHAIN_COLORS

        # MHR (127 joints) - use simplified main chains (excludes procedural/twist joints)
        elif skeleton_type == "mhr" or num_joints >= 100:
            logger.info(f"Using MHR simplified kinematic chains ({num_joints} joints)")
            # MHR has 127 joints total, but many are procedural/twist joints (l_lowleg_twist1_proc, etc.)
            # Joint 0 (body_world) is just a coordinate frame origin - skip it
            # These are the main structural chains for visualization:
            kinematic_chains = [
                [1, 2, 3, 4, 8],  # Left leg: root -> l_upleg -> l_lowleg -> l_foot -> l_ball
                [1, 18, 19, 20, 24],  # Right leg: root -> r_upleg -> r_lowleg -> r_foot -> r_ball
                [1, 34, 35, 36, 37, 110, 113],  # Spine: root -> c_spine0-3 -> c_neck -> c_head
                [
                    37,
                    74,
                    75,
                    76,
                    78,
                    88,
                ],  # Left arm: c_spine3 -> l_clavicle -> l_uparm -> l_lowarm -> l_wrist -> l_middle1
                [
                    37,
                    38,
                    39,
                    40,
                    42,
                    52,
                ],  # Right arm: c_spine3 -> r_clavicle -> r_uparm -> r_lowarm -> r_wrist -> r_middle1
            ]
            return kinematic_chains, DEFAULT_CHAIN_COLORS

        # Default: use HumanML3D chains with warning
        else:
            logger.warning(
                f"Unknown skeleton type '{skeleton_type}' with {num_joints} joints. "
                f"Using HumanML3D chains as fallback - visualization may be incorrect."
            )
            return DEFAULT_KINEMATIC_CHAINS, DEFAULT_CHAIN_COLORS

    def _generate_chain_colors(self, num_chains: int) -> list[str]:
        """Generate colors for kinematic chains.

        Args:
            num_chains: Number of chains to generate colors for

        Returns:
            List of color hex strings
        """
        # Base color palette
        base_colors = [
            "#ff6b6b",  # Red
            "#4ecdc4",  # Cyan
            "#45b7d1",  # Blue
            "#f9ca24",  # Yellow
            "#6c5ce7",  # Purple
            "#ff9ff3",  # Pink
            "#54a0ff",  # Light blue
            "#48dbfb",  # Aqua
            "#1dd1a1",  # Green
            "#feca57",  # Orange
        ]

        # Repeat colors if we need more
        colors = []
        for i in range(num_chains):
            colors.append(base_colors[i % len(base_colors)])

        return colors

    def _extract_joint_positions(
        self,
        data: Any,
        detected_format: DetectedFormat,
    ) -> np.ndarray:
        """Extract joint positions from motion data.

        Args:
            data: Motion data in any format
            detected_format: Format detection result

        Returns:
            Joint positions, shape (T, N, 3)

        Raises:
            ValueError: If positions cannot be extracted
        """
        format_type = detected_format.format_type

        # Motion object - use .positions property
        if format_type == "motion":
            if not isinstance(data, Motion):
                raise ValueError("Expected Motion object for format 'motion'")
            return data.positions

        # Joint positions - already in correct format
        if format_type == "joint_positions":
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if data.ndim != 3 or data.shape[2] != 3:
                raise ValueError(f"Joint positions must have shape (T, N, 3), got {data.shape}")
            return data

        # HumanML3D 263D - recover from RIC encoding
        if format_type == "263d":
            from ..converters.humanml3d_to_smpl import recover_from_ric

            if not isinstance(data, np.ndarray):
                data = np.array(data)
            return recover_from_ric(data, joints_num=22)

        # 272D - extract global positions directly from format (preserves floor alignment)
        if format_type == "272d":
            # 272D format: [root_vel_xz (2), heading_6d (6), local_pos (66), joint_vel (66), rotations_6d (132)]
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            num_frames = data.shape[0]
            local_positions = data[:, 8:74].reshape(num_frames, 22, 3)
            heading_6d = data[:, 2:8]
            root_velocities_xz = data[:, 0:2]

            # Import utilities
            from ..utils.humanml_utils import rot_yaw_np

            # Robust heading decode: heading is always a pure Y-axis rotation.
            # Only channels 0 (cosθ) and 2 (sinθ) are live; the other four are
            # structural constants that carry noise after model inference. Using
            # Gram-Schmidt on the full 6D lets that noise tilt the heading out of
            # the XZ plane, causing accumulated character tilt. Instead, extract
            # the yaw angle from the two live channels only.
            yaw_deltas = np.arctan2(heading_6d[:, 2], heading_6d[:, 0])
            heading_diff = rot_yaw_np(yaw_deltas)

            # Reconstruct absolute heading rotation matrices
            absolute_heading = np.zeros((num_frames, 3, 3), dtype=np.float32)
            absolute_heading[0] = np.eye(3, dtype=np.float32)
            for t in range(1, num_frames):
                # heading_diff[t] = heading_rot[t] @ heading_rot[t-1].T
                # So heading_rot[t] = heading_diff[t] @ heading_rot[t-1]
                absolute_heading[t] = heading_diff[t] @ absolute_heading[t - 1]

            # Compute root positions from velocities
            root_positions = np.zeros((num_frames, 3), dtype=np.float32)
            for t in range(1, num_frames):
                vel_3d = np.array([root_velocities_xz[t, 0], 0, root_velocities_xz[t, 1]], dtype=np.float32)
                global_vel = absolute_heading[t - 1] @ vel_3d
                root_positions[t] = root_positions[t - 1] + global_vel

            # Transform local positions to global
            # local_positions were stored as: local = heading_rot @ (global - root)
            # So: global = heading_rot.T @ local + root
            global_positions = np.zeros((num_frames, 22, 3), dtype=np.float32)
            for t in range(num_frames):
                for joint_idx in range(22):
                    local_pos = local_positions[t, joint_idx]
                    # Apply INVERSE rotation (transpose) to recover global
                    global_pos = absolute_heading[t].T @ local_pos
                    global_positions[t, joint_idx] = global_pos + root_positions[t]

            # Fix root Y position: 272D format loses absolute root height
            # The root is at y=0 in global coords, but should be at proper height
            # Calculate correct root height: root should be such that feet touch ground (y=0)
            # So root_y = -foot_y_local (where foot_y_local is foot position relative to root)
            foot_indices = [10, 11]  # left_foot, right_foot in HumanML3D

            # Get foot Y positions relative to root (from local positions)
            # These are NEGATIVE values (feet are below root)
            foot_y_relative = local_positions[:, foot_indices, 1].min(axis=1)  # Min across both feet per frame

            # The root should be at height such that lowest foot is at y=0
            # root_y = -foot_y_relative (if foot is at -0.9 relative to root, root should be at 0.9)
            for t in range(num_frames):
                # Set root Y to proper height (feet relative to root is negative, so negate)
                root_y_offset = -foot_y_relative[t]
                global_positions[t, :, 1] += root_y_offset

            # Now apply floor alignment (move feet to y=0)
            foot_y_min = global_positions[:, foot_indices, 1].min()
            global_positions[:, :, 1] -= foot_y_min

            return global_positions

        # 205D (STMC) - convert to Motion then extract positions
        if format_type == "205d":
            import torch

            from ..converters.stmc_converter import STMC205Converter

            converter = STMC205Converter()
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            motion = converter.inverse(data)
            positions = motion.positions.copy()

            # Apply coordinate normalization for skeleton visualization
            # STMC reconstruction appears to have Y-down, so flip Y
            positions[:, :, 1] = -positions[:, :, 1]  # Flip Y (up/down)

            # Floor alignment - move feet to y=0
            foot_indices = [7, 8, 10, 11]  # ankle and toe indices
            valid_foot_indices = [i for i in foot_indices if i < positions.shape[1]]
            if valid_foot_indices:
                foot_y_min = positions[:, valid_foot_indices, 1].min()
                positions[:, :, 1] -= foot_y_min

            return positions

        # ALIGN 232D - convert to Motion then extract positions
        if format_type == "align":
            from ..converters.align_converter import ALIGNConverter

            converter = ALIGNConverter(body_model_path=self.body_model_path)
            motion = converter.from_align(data, fps=ALIGN_DEFAULT_FPS)
            positions = motion.positions.copy()

            # ALIGN already handles floor alignment
            return positions

        # SMPL parameters - convert to Motion then extract positions
        if format_type == "smpl_params":
            logger.warning(
                "SMPL parameters detected - converting to joint positions. This may not preserve all information."
            )
            # For now, just return a placeholder - full SMPL FK would be needed
            # This will be properly implemented when we integrate body models
            raise NotImplementedError(
                "SMPL parameter visualization will be fully implemented in Phase 3. "
                "For now, convert to Motion object first."
            )

        # SMPL poses - need translation too
        if format_type == "smpl_poses":
            raise NotImplementedError(
                "SMPL pose visualization requires translation data. Use full SMPL parameters or Motion object instead."
            )

        # MHR format - extract joint positions from MHR forward pass
        if format_type == "mhr":
            from pathlib import Path

            import torch

            # Load MHR model
            try:
                from mhr.mhr import MHR
            except ImportError:
                raise RuntimeError("MHR package not available. Run 'bash scripts/setup_mhr.sh' to install.")

            # Get assets path
            assets_path = (
                Path(self.body_model_path)
                if self.body_model_path
                else Path.home() / ".motion_toolbox" / "datasets" / "mhr" / "assets"
            )

            if not assets_path.exists():
                raise RuntimeError(f"MHR assets not found at {assets_path}")

            device = torch.device("cpu")
            mhr_model = MHR.from_files(folder=assets_path, lod=1, device=device)

            # Extract MHR parameters
            model_params = torch.as_tensor(data["model_parameters"], device=device).float()
            num_frames = model_params.shape[0]

            identity = data.get("identity_coeffs")
            if identity is None:
                identity = torch.zeros((num_frames, 45), device=device)
            else:
                identity = torch.as_tensor(identity, device=device).float()
                if identity.shape[0] == 1:
                    identity = identity.repeat(num_frames, 1)

            face_expr = data.get("face_expr_coeffs")
            if face_expr is None:
                face_expr = torch.zeros((num_frames, 72), device=device)
            else:
                face_expr = torch.as_tensor(face_expr, device=device).float()
                if face_expr.shape[0] == 1:
                    face_expr = face_expr.repeat(num_frames, 1)

            # Run forward pass
            with torch.no_grad():
                vertices, skeleton_state = mhr_model(
                    identity_coeffs=identity,
                    model_parameters=model_params,
                    face_expr_coeffs=face_expr,
                )

            # skeleton_state format: (frames, 127 joints, 8 values)
            # Values 0-2: GLOBAL/WORLD position in centimeters (X, Y, Z)
            # Values 3-6: Rotation quaternion (x, y, z, w)
            # Value 7: Scale (always 1.0)
            # NOTE: These are already global positions, not local offsets!
            positions = skeleton_state[..., :3].cpu().numpy()

            # MHR uses centimeters, convert to meters
            positions = positions * 0.01

            # MHR's default facing is backward (-Z), flip to face camera (+Z)
            # This rotates 180° around Y-axis
            positions[:, :, 2] *= -1

            # Floor alignment
            min_y = positions[:, :, 1].min()
            positions[:, :, 1] -= min_y

            logger.info(f"Extracted MHR joint positions: {positions.shape}")
            return positions

        raise ValueError(
            f"Cannot extract joint positions from format '{format_type}'. "
            f"Supported formats: motion, joint_positions, 263d, 272d, 205d, mhr"
        )

    def _visualize_mesh(
        self,
        data: Any,
        output_path: Path,
        detected_format: DetectedFormat,
        body_model: str,
        fps: int,
        title: str | None,
        compatibility: CompatibilityResult,
        **kwargs,
    ) -> Path:
        """Visualize motion as animated mesh.

        Converts data to Motion object if needed, then renders with MeshVisualizer.
        """
        # If format is MHR, use MHR body model regardless of requested body_model
        if detected_format.format_type == "mhr":
            body_model = "mhr"
            logger.info("MHR format detected - using MHR body model")

        logger.info(f"Converting {detected_format.format_type} to Motion for {body_model} mesh visualization...")

        # Convert to Motion object
        motion = self._convert_to_motion(data, detected_format, compatibility)

        # Determine skip_rotation based on Motion's coordinate system
        # Y-up Motion objects need skip_rotation=true (no 180° flip)
        # Z-up or unknown need skip_rotation=false (apply 180° flip)
        if hasattr(motion, "up_axis"):
            skip_rotation = motion.up_axis == "y"
        else:
            # Fallback for formats that produce Y-up output
            skip_rotation = detected_format.format_type in ["205d"]

        # Create mesh visualizer and render
        # Disable floor in GLB since unified HTML viewer adds its own checkerboard
        vis = MeshVisualizer(
            body_model=body_model,
            body_model_path=self.body_model_path,
            fps=fps,
            include_floor=False,
            **kwargs,
        )

        return vis.visualize_motion(motion, output_path, skip_rotation=skip_rotation)

    def _convert_to_motion(
        self,
        data: Any,
        detected_format: DetectedFormat,
        compatibility: CompatibilityResult,
    ) -> Motion:
        """Convert data to Motion object for mesh visualization.

        Args:
            data: Input data in any format
            detected_format: Format detection result
            compatibility: Compatibility check result

        Returns:
            Motion object ready for mesh visualization

        Raises:
            NotImplementedError: If conversion requires SMPLify optimization
        """
        format_type = detected_format.format_type

        # Motion object - already in correct format
        if format_type == "motion":
            if not isinstance(data, Motion):
                raise ValueError("Expected Motion object for format 'motion'")
            return data

        # SMPL parameters - convert to Motion
        if format_type == "smpl_params":
            # This would require SMPL forward kinematics
            # For now, indicate it needs proper implementation
            raise NotImplementedError(
                "Direct SMPL parameter visualization not yet fully implemented. "
                "Convert to Motion object first using SMPL body model."
            )

        # 272D - convert to Motion
        if format_type == "272d":
            from ..converters.motion_converters import Format272ToMotionConverter

            converter = Format272ToMotionConverter()
            motion = converter.convert(data)
            logger.info(f"Converted 272D to Motion ({motion.num_frames} frames)")
            return motion

        # 205D (STMC) - convert to Motion
        if format_type == "205d":
            import torch

            from ..converters.stmc_converter import STMC205Converter
            from ..core.motion import Motion as MotionClass

            converter = STMC205Converter()
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            motion = converter.inverse(data)

            # For 205D mesh: we skip the viewer rotation (skipRotation=true)
            # so we keep the motion as-is (Y-up coordinate system)
            # Just apply floor alignment based on foot positions
            root_translation = motion.root_translation.copy()
            positions = motion.positions  # Already Y-up from STMC

            # Floor alignment - find lowest foot and shift up
            foot_indices = [7, 8, 10, 11]
            valid_foot_indices = [i for i in foot_indices if i < positions.shape[1]]
            if valid_foot_indices:
                foot_y_min = positions[:, valid_foot_indices, 1].min()
                root_translation[:, 1] -= foot_y_min

            motion_fixed = MotionClass(
                joint_rotations=motion.joint_rotations,
                root_translation=root_translation,
                skeleton=motion.skeleton,
                framerate=motion.framerate,
                floor_alignment="none",
            )

            logger.info(f"Converted 205D to Motion ({motion_fixed.num_frames} frames)")
            return motion_fixed

        # 232D (ALIGN) - convert to Motion
        if format_type == "align":
            from ..converters.align_converter import ALIGNConverter

            converter = ALIGNConverter(body_model_path=self.body_model_path)
            motion = converter.from_align(data, fps=ALIGN_DEFAULT_FPS)
            logger.info(f"Converted ALIGN to Motion ({motion.num_frames} frames)")
            return motion

        # 263D (HumanML3D) - requires SMPLify optimization
        if format_type == "263d":
            if compatibility.conversion_required == ConversionRequirement.SMPLIFY:
                raise NotImplementedError(
                    "HumanML3D 263D format requires SMPLify optimization for mesh visualization. "
                    "This is a slow process. Options:\n"
                    "  1. Use skeleton visualization instead (fast)\n"
                    "  2. Use the HumanML3DToSMPLConverter directly with SMPLify\n"
                    "  3. Convert to 272D format first"
                )
            # Simple conversion if SMPLify not required
            from ..converters.humanml3d_direct import HumanML3DDirectConverter

            try:
                converter = HumanML3DDirectConverter()
                motion = converter.convert(data)
                logger.info(f"Converted 263D to Motion ({motion.num_frames} frames)")
                return motion
            except Exception as e:
                logger.error(f"Failed to convert 263D: {e}")
                raise NotImplementedError(
                    "263D conversion failed. Use skeleton visualization or convert to 272D format first."
                )

        # Joint positions - convert to Motion with identity rotations
        if format_type == "joint_positions":
            # This is a lossy conversion - joint positions don't contain rotation info
            # For proper mesh visualization, we'd need to estimate rotations (inverse kinematics)
            raise NotImplementedError(
                "Joint positions alone cannot be converted to mesh animation "
                "(missing rotation information). Options:\n"
                "  1. Use skeleton visualization instead\n"
                "  2. Provide full motion data (263D, 272D, or Motion object)"
            )

        # MHR format - convert to Motion using MHRToMotionConverter
        if format_type == "mhr":
            from ..converters.motion_converters import MHRToMotionConverter

            converter = MHRToMotionConverter()
            motion = converter.convert(data)
            logger.info(f"Converted MHR to Motion ({motion.num_frames} frames, preserving model_data)")
            return motion

        raise ValueError(
            f"Cannot convert format '{format_type}' to Motion for mesh visualization. "
            f"Supported formats for mesh: motion, 272d, 205d, mhr"
        )

    def visualize_batch(
        self,
        source: Any,
        output_dir: str | Path,
        *,
        limit: int | None = None,
        mode: str | None = "skeleton",
        body_model: str | None = None,
        fps: int = 20,
        prefix: str = "",
        **kwargs,
    ) -> list[Path]:
        """Visualize many motions from a dataset (or any iterable of samples).

        Each item produced by ``source`` must be one of:

        * a :class:`~motion_toolbox.core.motion.Motion` instance,
        * a dict with a ``"motion"`` key (matches the dataset classes in
          :mod:`motion_toolbox.data`), or
        * a raw array / Motion-compatible payload that
          :meth:`visualize` can route on its own.

        Parameters
        ----------
        source:
            Any sized or iterable producer of samples. Tested against
            :class:`MotionFolderDataset`, :class:`Format272Dataset`,
            :class:`HumanML3DDataset`, and a plain list of Motions.
        output_dir:
            Directory where the per-sample HTML files are written. Created
            if missing.
        limit:
            Optional cap on the number of samples to render.
        mode:
            Visualization mode. Defaults to ``"skeleton"`` because it
            works for every format without requiring SMPL weights on
            disk. Pass ``"auto"`` to delegate to the unified routing
            logic, or ``"mesh"`` to force a mesh viewer.
        body_model, fps, **kwargs:
            Forwarded to :meth:`visualize`.
        prefix:
            Prepended to each output filename.

        Returns
        -------
        list[pathlib.Path]
            Paths of the generated HTML files, in iteration order.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            iterator = (source[i] for i in range(len(source)))  # type: ignore[arg-type]
            has_len = True
        except TypeError:
            iterator = iter(source)
            has_len = False

        outputs: list[Path] = []
        for idx, sample in enumerate(iterator):
            if limit is not None and idx >= limit:
                break

            if isinstance(sample, Motion):
                data: Any = sample
                name = f"motion_{idx:04d}"
            elif isinstance(sample, dict) and "motion" in sample:
                data = sample["motion"]
                name = str(sample.get("name") or f"motion_{idx:04d}")
            else:
                data = sample
                name = f"motion_{idx:04d}"

            html_path = out_dir / f"{prefix}{name}.html"
            self.visualize(
                data,
                html_path,
                mode=mode,
                body_model=body_model,
                fps=fps,
                title=name,
                **kwargs,
            )
            outputs.append(html_path)
            if not has_len:
                logger.info(f"Rendered {idx + 1} samples to {out_dir}")

        return outputs

    def to_scene_data(
        self,
        data: Any,
        format: str | None = None,
        mode: str | None = None,
        body_model: str | None = None,
        **kwargs,
    ) -> dict:
        """Export motion data as Three.js scene data (JSON-serializable).

        This method exports visualization data in a format that can be:
        - Sent to frontend via API (Flask, FastAPI, Django)
        - Embedded in React/Vue/Angular components
        - Used for custom Three.js visualizations

        Args:
            data: Motion data in any supported format
            format: Optional format hint
            mode: Visualization mode ('skeleton', 'mesh', 'auto')
            body_model: Body model for mesh visualization
            **kwargs: Additional arguments (fps, title, etc.)

        Returns:
            Dictionary with Three.js-compatible scene data that can be
            JSON-serialized and sent to frontend

        Examples:
            >>> vis = Visualizer()
            >>> data = np.random.randn(120, 263)
            >>> scene = vis.to_scene_data(data, mode='skeleton')
            >>> import json
            >>> json_str = json.dumps(scene)  # Send to frontend
        """
        from .scene_exporter import SceneExporter

        # Use defaults if not specified
        if mode is None:
            mode = "skeleton"  # Default to skeleton for scene data export

        # Detect format
        detected_format = self.format_detector.detect(data, format_hint=format)

        # Export based on mode
        exporter = SceneExporter()

        if mode == "skeleton" or mode == "auto":
            # Extract joint positions
            positions = self._extract_joint_positions(data, detected_format)

            # Pop forwarded kwargs so they don't collide with explicit args below.
            fps_arg = kwargs.pop("fps", 20)
            title_arg = kwargs.pop("title", f"Motion ({detected_format.format_type})")

            return exporter.export_skeleton(
                positions,
                fps=fps_arg,
                title=title_arg,
                **kwargs,
            )

        elif mode == "mesh":
            # For mesh mode, we'd need to generate GLB first
            # This is a more complex workflow - document it
            logger.warning(
                "Mesh mode scene data export requires GLB generation. "
                "Consider using visualize() method to generate GLB + HTML, "
                "then reference the GLB path in your frontend."
            )
            return {
                "type": "mesh",
                "message": "Mesh visualization requires GLB file generation",
                "recommendation": "Use Visualizer.visualize() to generate GLB, then reference the path",
                "metadata": {
                    "format": detected_format.format_type,
                    "body_model": body_model or "smpl",
                },
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __repr__(self) -> str:
        """String representation of visualizer."""
        return (
            f"Visualizer(mode={self.default_mode}, body_model={self.default_body_model}, "
            f"on_incompatible={self.on_incompatible})"
        )


# Convenience functions for backward compatibility


def visualize(
    data: Any,
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Convenience function to visualize motion data.

    Creates a Visualizer instance and calls visualize().

    Args:
        data: Motion data in any supported format
        output_path: Path to save output HTML file
        **kwargs: Additional arguments passed to Visualizer.visualize()

    Returns:
        Path to generated visualization file

    Examples:
        >>> from motion_toolbox.visualization import visualize
        >>> visualize(motion_data, "output.html", mode='skeleton')
    """
    vis = Visualizer()
    return vis.visualize(data, output_path, **kwargs)
