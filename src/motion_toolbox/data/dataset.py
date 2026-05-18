"""Generic folder-backed motion dataset.

Wraps :func:`motion_toolbox.io.load_motion` for any directory of motion files
that ``load_motion`` can auto-detect (SMPL ``.npz``, GVHMR ``.pt``, and the
fixed-dimension ``.npy`` family: 263D HumanML3D, 272D, 205D STMC, 232D ALIGN).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..core.motion import Motion
from ..io import load_motion

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    from torch.utils.data import Dataset as _TorchDataset
else:  # pragma: no cover - exercised only on torch-less installs
    _TorchDataset = object  # type: ignore[assignment,misc]


class MotionFolderDataset(_TorchDataset):  # type: ignore[misc,valid-type]
    """Lazy ``torch.utils.data.Dataset`` over a folder of motion files.

    Each ``__getitem__`` call loads one file via :func:`load_motion` and
    returns a dict ``{"motion": Motion, "name": str, "path": str}``. Files
    are discovered with a glob pattern at construction time; loading is
    deferred so the dataset is cheap to instantiate even on very large
    corpora.

    Parameters
    ----------
    root:
        Directory containing motion files.
    pattern:
        Glob pattern (default ``"*.npy"``). Use ``"**/*.npz"`` for recursive
        ``.npz`` search, etc.
    format:
        Optional explicit format hint forwarded to :func:`load_motion`
        (``"smpl"``, ``"gvhmr"``, ``"272d"``, ``"263d"``, ``"205d"``,
        ``"align"``). When ``None`` the format is auto-detected per file.
    framerate:
        Target framerate forwarded to :func:`load_motion`.
    body_model_path:
        Path to SMPL body-model weights, forwarded to :func:`load_motion`
        when needed (SMPL / GVHMR).
    transform:
        Optional callable applied to the loaded :class:`Motion` before it is
        returned. Useful for slicing, downsampling, normalisation, etc.
    exclude:
        Filenames (basename only) to skip — handy for HumanML3D's
        ``Mean.npy`` / ``Std.npy`` statistics files.
    return_motion_only:
        When ``True`` return the bare :class:`Motion` instead of a dict.

    Examples
    --------
    >>> dataset = MotionFolderDataset(
    ...     "sample_data/format_272",
    ...     pattern="*.npy",
    ...     format="272d",
    ...     exclude=("Mean.npy", "Std.npy"),
    ... )
    >>> len(dataset)
    3
    >>> sample = dataset[0]
    >>> sample["motion"].num_frames
    196
    """

    def __init__(
        self,
        root: str | Path,
        pattern: str = "*.npy",
        *,
        format: str | None = None,
        framerate: float = 30.0,
        body_model_path: str | None = None,
        transform: Callable[[Motion], Motion] | None = None,
        exclude: Sequence[str] = ("Mean.npy", "Std.npy"),
        return_motion_only: bool = False,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "MotionFolderDataset requires PyTorch. Install with: "
                "pip install 'motion-toolbox[ml]'"
            )
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.pattern = pattern
        self.format = format
        self.framerate = framerate
        self.body_model_path = body_model_path
        self.transform = transform
        self.exclude = set(exclude)
        self.return_motion_only = return_motion_only

        files = sorted(p for p in self.root.glob(pattern) if p.is_file())
        files = [p for p in files if p.name not in self.exclude]
        if not files:
            raise FileNotFoundError(
                f"No files matched pattern '{pattern}' under {self.root} "
                f"(after excluding {sorted(self.exclude) or 'nothing'})."
            )
        self.files: list[Path] = files

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Any:
        path = self.files[index]
        motion = load_motion(
            path,
            format=self.format,
            body_model_path=self.body_model_path,
            framerate=self.framerate,
        )
        if self.transform is not None:
            motion = self.transform(motion)

        if self.return_motion_only:
            return motion
        return {"motion": motion, "name": path.stem, "path": str(path)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        """Return basenames (without extension) in dataset order."""
        return [p.stem for p in self.files]

    def __repr__(self) -> str:
        return (
            f"MotionFolderDataset(root={self.root!s}, pattern={self.pattern!r}, "
            f"format={self.format!r}, n={len(self)})"
        )
