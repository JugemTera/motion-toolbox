"""Unified motion file loader with automatic format detection.

The module separates *reading from disk* (I/O) from *format conversion*
(converters).  ``load_motion`` is the single public entry-point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.motion import Motion

# ---------------------------------------------------------------------------
# Format identifiers used throughout (kept in sync with FormatDetector)
# ---------------------------------------------------------------------------

_SMPL_REQUIRED_KEYS = {"pose_body", "root_orient", "trans"}
_GVHMR_KEY = "smpl_params_global"

# NPZ key conventions for 272D archives
_272D_NPZ_KEYS = ("motion_data", "data")

# Dimension → format mapping for .npy arrays
_DIM_TO_FORMAT: dict[int, str] = {
    263: "263d",
    272: "272d",
    205: "205d",
    232: "align",
}


# ---------------------------------------------------------------------------
# File readers  (pure I/O – no conversion)
# ---------------------------------------------------------------------------


def _read_npz(path: Path) -> dict[str, Any]:
    return dict(np.load(path, allow_pickle=True))


def _read_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def _read_pt(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError("PyTorch is required to load .pt files.  Install it with:  pip install torch") from exc
    return torch.load(path, map_location="cpu", weights_only=False)


_READERS: dict[str, Any] = {
    ".npz": _read_npz,
    ".npy": _read_npy,
    ".pt": _read_pt,
}


# ---------------------------------------------------------------------------
# Format detection from loaded data
# ---------------------------------------------------------------------------


def _detect_npz_format(data: dict[str, Any]) -> str:
    """Classify an NPZ archive as ``smpl`` or ``272d``."""
    keys = set(data.keys())
    if _SMPL_REQUIRED_KEYS.issubset(keys):
        return "smpl"
    for k in _272D_NPZ_KEYS:
        if k in keys:
            return "272d"
    if len(keys) == 1:
        arr = data[next(iter(keys))]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 272:
            return "272d"
    raise ValueError(
        f"Cannot auto-detect NPZ format.  Keys present: {sorted(keys)}.  "
        "Pass format='smpl' or format='272d' explicitly."
    )


def _detect_npy_format(data: np.ndarray) -> str:
    """Classify a plain ``.npy`` array by its feature dimension."""
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D array (T, D) from .npy file, got shape {data.shape}.")
    dim = data.shape[1]
    fmt = _DIM_TO_FORMAT.get(dim)
    if fmt is None:
        raise ValueError(
            f"Cannot auto-detect format for .npy with {dim} features.  Known dimensions: {sorted(_DIM_TO_FORMAT)}."
        )
    return fmt


def _detect_pt_format(data: Any) -> str:
    """Classify a ``.pt`` archive (currently only GVHMR)."""
    if isinstance(data, dict) and _GVHMR_KEY in data:
        return "gvhmr"
    raise ValueError(f"Cannot auto-detect .pt format.  Expected key '{_GVHMR_KEY}'.  Pass format='gvhmr' explicitly.")


# ---------------------------------------------------------------------------
# Converter dispatch
# ---------------------------------------------------------------------------


def _convert(
    fmt: str,
    data: Any,
    *,
    body_model_path: str | None,
    framerate: float,
) -> Motion:
    """Route ``data`` through the correct converter and return a Motion."""
    if fmt == "smpl":
        from ..converters.motion_converters import SMPLToMotionConverter

        converter = SMPLToMotionConverter(body_model_path=body_model_path, framerate=framerate)
        return converter.convert(data)

    if fmt == "gvhmr":
        from ..converters.motion_converters import GVHMRToMotionConverter

        converter = GVHMRToMotionConverter(body_model_path=body_model_path)
        return converter.convert(data)

    if fmt == "272d":
        from ..converters.motion_converters import Format272ToMotionConverter

        arr = _extract_272d_array(data)
        converter = Format272ToMotionConverter(framerate=framerate, body_model_path=body_model_path)
        return converter.convert(arr)

    if fmt == "263d":
        from ..converters.motion_converters import HumanML3DToMotionConverter

        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        converter = HumanML3DToMotionConverter()
        return converter.convert(arr)

    if fmt == "205d":
        from ..converters.stmc_converter import STMC205Converter

        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        converter = STMC205Converter()
        return converter.convert(arr)

    if fmt == "align":
        from ..converters.align_converter import ALIGNConverter

        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        converter = ALIGNConverter()
        return converter.convert(arr)

    raise ValueError(f"Unsupported format '{fmt}'.  Supported: smpl, gvhmr, 272d, 263d, 205d, align.")


def _extract_272d_array(data: Any) -> np.ndarray:
    """Pull the 272-D array out of a dict (NPZ) or pass through an ndarray."""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, dict):
        for key in _272D_NPZ_KEYS:
            if key in data:
                return np.asarray(data[key])
        if len(data) == 1:
            return np.asarray(next(iter(data.values())))
    raise ValueError("Could not extract 272D array from data.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_motion(
    path: str | Path,
    *,
    format: str | None = None,
    body_model_path: str | None = None,
    framerate: float = 30.0,
) -> Motion:
    """Load a motion file and return a :class:`Motion` object.

    Parameters
    ----------
    path:
        Path to a motion file (``.npz``, ``.npy``, or ``.pt``).
    format:
        Explicit format hint.  When ``None`` the format is auto-detected from
        the file extension and contents.  Accepted values:
        ``smpl``, ``gvhmr``, ``272d``, ``263d``, ``205d``, ``align``.
    body_model_path:
        Path to SMPL body-model weights.  Required for mesh-based rendering
        of SMPL / GVHMR data; ignored for dimensionality-based formats.
    framerate:
        Output framerate in Hz.  Defaults to 30.

    Returns
    -------
    Motion
        A fully constructed Motion object.

    Examples
    --------
    >>> from motion_toolbox.io import load_motion
    >>> motion = load_motion("walk.npz", body_model_path="/models/smpl")
    >>> motion = load_motion("dance.npy")            # auto-detects 263d/272d/205d/align
    >>> motion = load_motion("gvhmr_out.pt")          # auto-detects gvhmr
    >>> motion = load_motion("data.npz", format="272d")  # explicit hint
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {path}")

    ext = path.suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported file extension '{ext}'.  Supported: {', '.join(sorted(_READERS))}.")

    data = reader(path)

    if format is None:
        if ext == ".npz":
            format = _detect_npz_format(data)
        elif ext == ".npy":
            format = _detect_npy_format(data)
        elif ext == ".pt":
            format = _detect_pt_format(data)

    return _convert(
        format,
        data,
        body_model_path=body_model_path,
        framerate=framerate,
    )
