"""Scene data export for frontend integration.

This module provides functionality to export motion visualizations as
JSON-serializable scene data for use in web frameworks.
"""

from typing import Any

import numpy as np

from ..core.motion import Motion


class SceneExporter:
    """Exports motion data as JSON-serializable scene data.

    This allows visualization data to be embedded in web frameworks
    (Flask, FastAPI, Django, React, Vue, etc.) without generating
    standalone HTML files.

    Examples:
        >>> exporter = SceneExporter()
        >>> scene_data = exporter.export_skeleton(motion_obj)
        >>> json.dumps(scene_data)  # Can be sent to frontend
    """

    def export_skeleton(
        self,
        positions: np.ndarray,
        fps: int = 20,
        kinematic_chains: list[list[int]] | None = None,
        chain_colors: list[str] | None = None,
        title: str = "Skeleton Animation",
    ) -> dict[str, Any]:
        """Export skeleton as JSON-serializable scene data.

        Args:
            positions: Joint positions, shape (T, N, 3)
            fps: Frames per second
            kinematic_chains: Bone connections
            chain_colors: Colors for each chain
            title: Title for the visualization

        Returns:
            Dictionary with scene data that can be JSON-serialized
        """
        # Default kinematic chains for HumanML3D
        if kinematic_chains is None:
            kinematic_chains = [
                [0, 2, 5, 8, 11],  # Right leg
                [0, 1, 4, 7, 10],  # Left leg
                [0, 3, 6, 9, 12, 15],  # Spine
                [9, 14, 17, 19, 21],  # Right arm
                [9, 13, 16, 18, 20],  # Left arm
            ]

        if chain_colors is None:
            chain_colors = [
                "#ff6b6b",  # Right leg - red
                "#4ecdc4",  # Left leg - cyan
                "#45b7d1",  # Spine - blue
                "#f9ca24",  # Right arm - yellow
                "#6c5ce7",  # Left arm - purple
            ]

        return {
            "type": "skeleton",
            "title": title,
            "fps": fps,
            "num_frames": int(positions.shape[0]),
            "num_joints": int(positions.shape[1]),
            "positions": positions.tolist(),
            "kinematic_chains": kinematic_chains,
            "chain_colors": chain_colors,
            "metadata": {
                "format": "motion_toolbox_scene_v1",
                "three_js_compatible": True,
            },
        }

    def export_mesh(
        self,
        glb_path: str,
        fps: int = 30,
        title: str = "Mesh Animation",
    ) -> dict[str, Any]:
        """Export mesh as JSON-serializable scene data with GLB reference.

        Args:
            glb_path: Path to GLB file
            fps: Frames per second
            title: Title for the visualization

        Returns:
            Dictionary with scene data including GLB reference
        """
        return {
            "type": "mesh",
            "title": title,
            "fps": fps,
            "glb_path": glb_path,
            "metadata": {
                "format": "motion_toolbox_scene_v1",
                "viewer": "model-viewer",
            },
        }


def export_scene_data(
    data: Any,
    mode: str = "skeleton",
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to export scene data.

    Args:
        data: Motion data (positions array or Motion object)
        mode: Visualization mode ('skeleton' or 'mesh')
        **kwargs: Additional arguments for exporter

    Returns:
        JSON-serializable scene data dictionary

    Examples:
        >>> from motion_toolbox.visualization.constants import JOINT_COUNTS
        >>> positions = np.random.randn(120, JOINT_COUNTS['humanml3d'], 3)
        >>> scene = export_scene_data(positions, mode='skeleton')
        >>> import json
        >>> json_str = json.dumps(scene)
    """
    exporter = SceneExporter()

    if mode == "skeleton":
        if isinstance(data, Motion):
            positions = data.positions
            fps = int(data.framerate)
        elif isinstance(data, np.ndarray):
            positions = data
            fps = kwargs.get("fps", 20)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return exporter.export_skeleton(positions, fps=fps, **kwargs)

    elif mode == "mesh":
        # For mesh mode, data should be a path to GLB file
        if not isinstance(data, str):
            raise TypeError("For mesh mode, data should be GLB file path (str)")
        return exporter.export_mesh(data, **kwargs)

    else:
        raise ValueError(f"Unknown mode: {mode}")
