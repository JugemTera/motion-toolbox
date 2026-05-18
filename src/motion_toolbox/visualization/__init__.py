"""Motion Toolbox Visualization Module.

This module provides comprehensive 3D motion visualization capabilities.
Supports matplotlib-based rendering and GLB export for 3D visualization.
Includes headless operation for server environments.

The new unified visualization API is available through the Visualizer class,
which provides automatic format detection and intelligent routing to appropriate
visualization backends.
"""

# Configure headless environment first
from .compatibility import (
    BodyModel,
    CompatibilityMatrix,
    CompatibilityResult,
    ConversionRequirement,
    VisualizationMode,
    check_compatibility,
)
from .constants import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_GOOD,
    CONFIDENCE_HIGH,
    CONFIDENCE_LEVELS,
    CONFIDENCE_LOW,
    CONFIDENCE_MODERATE,
    DEFAULT_CHAIN_COLORS,
    # Visualization constants
    DEFAULT_FPS,
    FOOT_INDICES,
    FORMAT_272D_DIM,
    # New dictionary-based constants (recommended)
    FORMAT_DIMS,
    # Legacy individual constants (for backward compatibility)
    HUMANML3D_263D_DIM,
    JOINT_COUNTS,
    KINEMATIC_CHAINS,
    SKELETON_COMPARISON_OFFSET_X,
    SMPL_POSES_72D_DIM,
    STMC_205D_DIM,
    TIME_ESTIMATES,
)
from .format_detector import DetectedFormat, FormatDetector, detect_format
from .grid_config import GridConfig
from .headless_config import configure_headless_environment
from .html_viewer import MotionVisualizer, create_motion_visualization
from .keyframe_overlay_viewer import (
    KeyframeOverlayViewer,
    create_keyframe_overlay,
)
from .mesh_visualizer import MeshVisualizer, visualize_mesh
from .mpl_visualizer import MplVisualizer
from .scene_exporter import SceneExporter, export_scene_data
from .separate_viewers_grid import SeparateViewersGrid, create_separate_viewers_grid
from .skeleton_comparison import (
    create_single_skeleton_html,
    create_skeleton_comparison_html,
)
from .skeleton_visualizer import SkeletonVisualizer, visualize_skeleton
from .spatial_animation_grid import (
    SpatialAnimationGrid,
    create_spatial_animation_grid,
    create_spatial_animation_grid_from_config,
)

# New unified visualization API (Phase 1, 2, 3 & 4)
from .unified_visualizer import IncompatibleFormatError, Visualizer, VisualizerError, visualize

_headless_configured = configure_headless_environment()

try:
    from .glb_exporter import GLBExporter

    GLB_AVAILABLE = True
except ImportError:
    GLB_AVAILABLE = False
    GLBExporter = None

__all__ = [
    # Unified visualization API (NEW - Phase 1, 2, 3 & 4)
    "Visualizer",
    "visualize",
    "VisualizerError",
    "IncompatibleFormatError",
    "FormatDetector",
    "DetectedFormat",
    "detect_format",
    "CompatibilityMatrix",
    "CompatibilityResult",
    "ConversionRequirement",
    "VisualizationMode",
    "BodyModel",
    "check_compatibility",
    "SkeletonVisualizer",
    "visualize_skeleton",
    "MeshVisualizer",
    "visualize_mesh",
    "SceneExporter",
    "export_scene_data",
    # Constants (dictionary-based - recommended)
    "FORMAT_DIMS",
    "JOINT_COUNTS",
    "CONFIDENCE_LEVELS",
    "TIME_ESTIMATES",
    "KINEMATIC_CHAINS",
    "FOOT_INDICES",
    # Constants (legacy individual - for backward compatibility)
    "HUMANML3D_263D_DIM",
    "FORMAT_272D_DIM",
    "STMC_205D_DIM",
    "SMPL_POSES_72D_DIM",
    "CONFIDENCE_CERTAIN",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_GOOD",
    "CONFIDENCE_MODERATE",
    "CONFIDENCE_LOW",
    "DEFAULT_FPS",
    "SKELETON_COMPARISON_OFFSET_X",
    "DEFAULT_CHAIN_COLORS",
    # Existing API (backward compatible)
    "MplVisualizer",
    "get_visualizer",
    "GLBExporter",
    "get_glb_exporter",
    "MotionVisualizer",
    "create_motion_visualization",
    "SeparateViewersGrid",
    "create_separate_viewers_grid",
    "SpatialAnimationGrid",
    "create_spatial_animation_grid",
    "create_spatial_animation_grid_from_config",
    "KeyframeOverlayViewer",
    "create_keyframe_overlay",
    "GridConfig",
    "GLB_AVAILABLE",
    "create_skeleton_comparison_html",
    "create_single_skeleton_html",
]


def get_visualizer(visualization_type: str = "skeleton", **kwargs):
    """Get appropriate visualizer based on requirements.

    Args:
        visualization_type: "skeleton" (mesh and SMPL will be implemented via GLB export)
        **kwargs: Additional arguments for visualizer

    Returns:
        Configured visualizer instance
    """
    if visualization_type == "skeleton":
        # Use matplotlib visualizer
        mpl_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["figsize", "dpi", "joint_color", "bone_color", "joint_size", "bone_width"]
        }
        return MplVisualizer(**mpl_kwargs)

    elif visualization_type == "mesh":
        raise NotImplementedError("Mesh visualization will be implemented via GLB export")

    elif visualization_type == "both":
        raise NotImplementedError("Combined visualization will be implemented via GLB export")

    else:
        raise ValueError(f"Unknown visualization_type: {visualization_type}. Use 'skeleton'.")


def get_glb_exporter(model_path, **kwargs):
    """Factory function to create GLB exporter.

    Args:
        model_path (str): Path to SMPL model files
        **kwargs: Additional arguments for MEExporter

    Returns:
        MEExporter instance
    """
    if not GLB_AVAILABLE:
        raise ImportError("GLB export requires smplx and pygltflib. Install with: pip install smplx pygltflib torch")

    # Import here to avoid issues when not available
    from .glb_exporter import GLBExporter

    return GLBExporter(model_path, **kwargs)


