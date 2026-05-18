"""272-D MotionMillion-style dataset.

The 272-dimensional representation stores per-frame motion features in a
fixed-width tensor (see ``docs/272_FORMAT_REFERENCE.md``). A dataset is a
directory of ``.npy`` files with optional ``Mean.npy`` / ``Std.npy``
statistics files alongside them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..core.motion import Motion
from .dataset import MotionFolderDataset


class Format272Dataset(MotionFolderDataset):
    """Dataset over a folder of 272-D ``.npy`` motion files.

    Skips ``Mean.npy`` / ``Std.npy`` automatically and exposes them as
    :attr:`mean` / :attr:`std` numpy arrays when present so normalisation
    can be reproduced.

    Examples
    --------
    >>> dataset = Format272Dataset("sample_data/format_272/")
    >>> sample = dataset[0]
    >>> sample["motion"].num_joints  # HumanML3D layout (22 body joints)
    22
    """

    def __init__(
        self,
        root: str | Path,
        *,
        framerate: float = 30.0,
        body_model_path: str | None = None,
        transform: Callable[[Motion], Motion] | None = None,
        return_motion_only: bool = False,
    ) -> None:
        super().__init__(
            root,
            pattern="*.npy",
            format="272d",
            framerate=framerate,
            body_model_path=body_model_path,
            transform=transform,
            exclude=("Mean.npy", "Std.npy"),
            return_motion_only=return_motion_only,
        )

        mean_path = self.root / "Mean.npy"
        std_path = self.root / "Std.npy"
        self.mean: np.ndarray | None = np.load(mean_path) if mean_path.exists() else None
        self.std: np.ndarray | None = np.load(std_path) if std_path.exists() else None

    def __repr__(self) -> str:
        stats = "with mean/std" if (self.mean is not None and self.std is not None) else "no stats"
        return f"Format272Dataset(root={self.root!s}, n={len(self)}, {stats})"
