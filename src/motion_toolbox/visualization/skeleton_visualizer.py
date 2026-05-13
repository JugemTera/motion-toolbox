"""Skeleton visualization backend for unified visualizer.

This module provides skeleton-based visualization for all motion formats,
extracting joint positions and rendering them with Three.js.
"""

import json
import logging
from pathlib import Path

import numpy as np

from ..core.motion import Motion
from .constants import (
    DEFAULT_CHAIN_COLORS,
    DEFAULT_FPS,
    KINEMATIC_CHAINS,
    SKELETON_COMPARISON_OFFSET_X,
)
from .template_renderer import TemplateRenderer

logger = logging.getLogger(__name__)


# Aliases for backward compatibility
DEFAULT_KINEMATIC_CHAINS = KINEMATIC_CHAINS["humanml3d"]
SMPL_KINEMATIC_CHAINS = KINEMATIC_CHAINS["smpl"]


class SkeletonVisualizer:
    """Backend for skeleton visualization.

    Handles extraction of joint positions from various formats and
    generates Three.js-based HTML visualizations.

    Attributes:
        kinematic_chains: List of joint chains for bones
        chain_colors: Colors for each kinematic chain
        joint_radius: Sphere radius for joint markers (metres)
        bone_radius: Cylinder radius for bone segments (metres)
        auto_flip: Whether to apply the body-specific upside-down heuristic

    Examples:
        >>> from motion_toolbox.visualization.constants import JOINT_COUNTS
        >>> vis = SkeletonVisualizer()
        >>>
        >>> # Visualize joint positions (HumanML3D format)
        >>> positions = np.random.randn(120, JOINT_COUNTS['humanml3d'], 3)
        >>> vis.visualize_positions(positions, "output.html")
        >>>
        >>> # Visualize Motion object
        >>> vis.visualize_motion(motion_obj, "output.html")
    """

    def __init__(
        self,
        kinematic_chains: list[list[int]] | None = None,
        chain_colors: list[str] | None = None,
        joint_radius: float = 0.035,
        bone_radius: float = 0.018,
        auto_flip: bool = True,
    ):
        """Initialize skeleton visualizer.

        Args:
            kinematic_chains: Custom kinematic chains (bone connections)
            chain_colors: Custom colors for each chain
            joint_radius: Radius of joint spheres in metres (default body-scale 3.5cm)
            bone_radius: Radius of bone cylinders in metres (default body-scale 1.8cm)
            auto_flip: Apply body-specific upside-down detection heuristic. Set False
                for non-body skeletons (e.g. hands) where joint indices 7/8/15 have
                no "feet/head" meaning.
        """
        self.kinematic_chains = kinematic_chains or DEFAULT_KINEMATIC_CHAINS
        self.chain_colors = chain_colors or DEFAULT_CHAIN_COLORS
        self.joint_radius = joint_radius
        self.bone_radius = bone_radius
        self.auto_flip = auto_flip

    def visualize_positions(
        self,
        positions: np.ndarray,
        output_path: str | Path,
        fps: int = DEFAULT_FPS,
        title: str = "Skeleton Animation",
        label: str = "Skeleton",
    ) -> Path:
        """Visualize joint positions as animated skeleton.

        Args:
            positions: Joint positions, shape (T, N, 3)
            output_path: Path to save HTML file
            fps: Frames per second
            title: Title for visualization
            label: Label for skeleton

        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_frames, num_joints = positions.shape[:2]
        logger.info(f"Creating skeleton visualization: {num_frames} frames, {num_joints} joints")

        positions = positions.copy()  # Don't modify original

        if self.auto_flip:
            # Body-specific heuristic: detect and correct inverted skeletons
            # (head below feet). Uses body joint indices 15=head, 7/8=feet.
            # Must NOT be applied to non-body skeletons (hands, objects, etc.)
            head_idx = 15 if num_joints > 15 else (12 if num_joints > 12 else 0)
            feet_indices = [j for j in [7, 8] if j < num_joints]

            if feet_indices and head_idx > 0:
                head_y = positions[0, head_idx, 1]
                feet_y = positions[0, feet_indices, 1].mean()

                if head_y < feet_y:
                    logger.info("Auto-flipping inverted skeleton")
                    positions[:, :, 1] = -positions[:, :, 1]
                    positions[:, :, 2] = -positions[:, :, 2]

                    y_min = positions[:, :, 1].min()
                    if y_min < 0:
                        positions[:, :, 1] -= y_min

        # Generate HTML
        html_content = self._generate_single_skeleton_html(
            positions.tolist(),
            title,
            label,
            fps,
            num_frames,
        )

        output_path.write_text(html_content)
        logger.info(f"Skeleton visualization saved to: {output_path}")

        return output_path

    def visualize_motion(
        self,
        motion: Motion,
        output_path: str | Path,
        fps: int | None = None,
        title: str = "Motion Animation",
        label: str | None = None,
    ) -> Path:
        """Visualize Motion object as animated skeleton.

        Args:
            motion: Motion object
            output_path: Path to save HTML file
            fps: Frames per second (uses motion.framerate if None)
            title: Title for visualization
            label: Label for skeleton (auto-generated if None)

        Returns:
            Path to generated HTML file
        """
        if fps is None:
            fps = int(motion.framerate)

        if label is None:
            skeleton_name = motion.body_model_type or f"{motion.num_joints} joints"
            label = f"Motion ({skeleton_name})"

        # Get joint positions from motion
        positions = motion.positions

        return self.visualize_positions(
            positions,
            output_path,
            fps=fps,
            title=title,
            label=label,
        )

    def visualize_comparison(
        self,
        positions_left: np.ndarray,
        positions_right: np.ndarray,
        output_path: str | Path,
        fps: int = DEFAULT_FPS,
        title: str = "Skeleton Comparison",
        left_label: str = "Left",
        right_label: str = "Right",
    ) -> Path:
        """Visualize two skeletons side-by-side for comparison.

        Args:
            positions_left: Left skeleton positions, shape (T, N, 3)
            positions_right: Right skeleton positions, shape (T, N, 3)
            output_path: Path to save HTML file
            fps: Frames per second
            title: Title for visualization
            left_label: Label for left skeleton
            right_label: Label for right skeleton

        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure same number of frames
        min_frames = min(len(positions_left), len(positions_right))
        positions_left = positions_left[:min_frames]
        positions_right = positions_right[:min_frames]

        # Offset right skeleton to be side by side
        offset = np.array([SKELETON_COMPARISON_OFFSET_X, 0.0, 0.0])
        positions_right_offset = positions_right + offset

        logger.info(f"Creating comparison visualization: {min_frames} frames, {positions_left.shape[1]} joints")

        # Generate HTML
        html_content = self._generate_comparison_html(
            positions_left.tolist(),
            positions_right_offset.tolist(),
            title,
            left_label,
            right_label,
            fps,
            min_frames,
        )

        output_path.write_text(html_content)
        logger.info(f"Comparison visualization saved to: {output_path}")

        return output_path

    def _generate_single_skeleton_html(
        self,
        positions: list,
        title: str,
        label: str,
        fps: int,
        num_frames: int,
        joint_radius: float | None = None,
        bone_radius: float | None = None,
    ) -> str:
        """Generate HTML for single skeleton visualization using the skeleton_viewer template."""
        joint_radius = joint_radius if joint_radius is not None else self.joint_radius
        bone_radius = bone_radius if bone_radius is not None else self.bone_radius
        renderer = TemplateRenderer()
        return renderer.render(
            "skeleton_viewer.html",
            {
                "title": title,
                "num_frames_max": str(num_frames - 1),
                "num_frames": str(num_frames),
                "fps": str(fps),
                "joint_radius": str(joint_radius),
                "bone_radius": str(bone_radius),
                "positions_json": json.dumps(positions),
                "chains_json": json.dumps(self.kinematic_chains),
                "colors_json": json.dumps(self.chain_colors),
            },
        )

    def _generate_comparison_html(
        self,
        pos_left: list,
        pos_right: list,
        title: str,
        left_label: str,
        right_label: str,
        fps: int,
        num_frames: int,
    ) -> str:
        """Generate HTML for side-by-side comparison (implementation similar to single but with two skeletons)."""
        chains_json = json.dumps(self.kinematic_chains)
        colors_json = json.dumps(self.chain_colors)

        # Simplified version - full implementation would have two skeleton groups
        # For now, reuse single skeleton logic (can be enhanced later)
        return self._generate_single_skeleton_html(
            pos_left,
            title,
            left_label,
            fps,
            num_frames,
        )


def visualize_skeleton(
    data: np.ndarray | Motion,
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Convenience function to visualize skeleton data.

    Args:
        data: Joint positions (T, N, 3) or Motion object
        output_path: Path to save HTML file
        **kwargs: Additional arguments passed to SkeletonVisualizer

    Returns:
        Path to generated HTML file

    Examples:
        >>> from motion_toolbox.visualization.constants import JOINT_COUNTS
        >>> positions = np.random.randn(120, JOINT_COUNTS['humanml3d'], 3)
        >>> visualize_skeleton(positions, "output.html")
    """
    vis = SkeletonVisualizer()

    if isinstance(data, Motion):
        return vis.visualize_motion(data, output_path, **kwargs)
    elif isinstance(data, np.ndarray):
        return vis.visualize_positions(data, output_path, **kwargs)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
