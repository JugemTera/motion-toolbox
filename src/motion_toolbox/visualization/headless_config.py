"""Headless visualization configuration for server environments.

This module configures matplotlib and other visualization libraries for headless operation,
ensuring videos and images can be generated without display requirements.
"""

import os
import warnings


def configure_headless_backend():
    """Configure matplotlib for headless operation."""
    try:
        import matplotlib

        # Set headless backend before importing pyplot
        if not hasattr(matplotlib, "_called_configure_headless"):
            matplotlib.use("Agg")  # Non-interactive backend
            matplotlib._called_configure_headless = True

        # Suppress GUI-related warnings
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Gui is implemented.*")

        # Set environment variables for headless operation
        os.environ["MPLBACKEND"] = "Agg"

        print("✅ Configured matplotlib for headless operation")
        return True

    except ImportError:
        warnings.warn("Matplotlib not available for headless configuration")
        return False


def configure_headless_environment():
    """Configure full headless environment for visualization."""
    # Configure display environment
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":99"  # Virtual display

    # Configure Qt for headless (if using Qt-based backends)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # Configure matplotlib
    matplotlib_configured = configure_headless_backend()

    # Configure OpenCV for headless operation
    opencv_configured = False
    try:
        import importlib.util

        opencv_configured = importlib.util.find_spec("cv2") is not None
        if opencv_configured:
            print("✅ OpenCV available for headless operation")
    except Exception:
        warnings.warn("OpenCV not available")

    return {"matplotlib": matplotlib_configured, "opencv": opencv_configured, "display_configured": True}


def ensure_output_directory(path: str) -> str:
    """Ensure output directory exists for headless rendering.

    Args:
        path: Output file path

    Returns:
        Validated output path
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)


# Automatically configure headless environment when module is imported
if __name__ != "__main__":
    _headless_config = configure_headless_environment()
else:
    print("Headless configuration module")
    print("Import this module to automatically configure headless visualization")
    config = configure_headless_environment()
    print(f"Configuration result: {config}")
