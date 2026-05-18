"""HumanML3D dataset wrapper.

HumanML3D ships per-sample as:

* ``<id>.npy``         — 263-D RIC features ``(T, 263)``
* ``<id>.txt``         — one or more text captions (one annotation per line)
* ``<id>_joints.npy``  — 3-D joint positions ``(T, 22, 3)`` (optional)

This dataset returns one entry per ``<id>`` with the raw 263-D features,
the text captions, and (if available) joint positions. The features are
suitable for model training; the visualizer accepts them directly via
:meth:`motion_toolbox.visualization.Visualizer.visualize_batch` because
``Visualizer.visualize`` auto-detects the 263-D format.

A first-class :class:`~motion_toolbox.core.motion.Motion` reconstruction
is intentionally *not* attempted here — the 263-D → SMPL path requires
SMPLify optimisation (see :class:`HumanML3DToSMPLConverter`).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    from torch.utils.data import Dataset as _TorchDataset
else:  # pragma: no cover
    _TorchDataset = object  # type: ignore[assignment,misc]


class HumanML3DDataset(_TorchDataset):  # type: ignore[misc,valid-type]
    """PyTorch ``Dataset`` over a HumanML3D-style folder.

    Each ``__getitem__`` returns a dict with these keys:

    * ``"features"`` — ``(T, 263)`` numpy array of 263-D RIC features.
    * ``"motion"``   — same numpy array, exposed under the conventional
      key so :meth:`Visualizer.visualize_batch` and similar helpers
      "just work".
    * ``"text"``     — list of caption strings (one per annotation).
    * ``"joints"``   — ``(T, 22, 3)`` numpy array of joint positions, or
      ``None`` when the ``_joints.npy`` sidecar is missing.
    * ``"name"`` / ``"path"`` — sample id and source path.

    Parameters
    ----------
    root:
        Folder containing ``<id>.npy`` / ``<id>.txt`` / ``<id>_joints.npy``.
    split_ids:
        Optional iterable of sample ids to restrict to (train / val / test).
    feature_dim:
        Expected feature width (default 263). Files with a different last-axis
        size are dropped.
    transform:
        Optional callable applied to the 263-D feature array.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split_ids: Iterable[str] | None = None,
        feature_dim: int = 263,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("HumanML3DDataset requires PyTorch. Install with: pip install 'motion-toolbox[ml]'")

        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.feature_dim = feature_dim
        self.transform = transform

        candidates = sorted(self.root.glob("*.npy"))
        skip_names = {"Mean.npy", "Std.npy"}
        primary: list[Path] = []
        for path in candidates:
            if path.name in skip_names:
                continue
            if path.stem.endswith("_joints") or path.stem.endswith("_263d"):
                continue
            primary.append(path)

        if split_ids is not None:
            wanted = set(split_ids)
            primary = [p for p in primary if p.stem in wanted]

        if not primary:
            raise FileNotFoundError(f"No HumanML3D feature files (<id>.npy) found under {self.root}")
        self.files: list[Path] = primary

        mean_path = self.root / "Mean.npy"
        std_path = self.root / "Std.npy"
        self.mean: np.ndarray | None = np.load(mean_path) if mean_path.exists() else None
        self.std: np.ndarray | None = np.load(std_path) if std_path.exists() else None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        sample_id = path.stem

        features = np.load(path)
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature array of shape (T, {self.feature_dim}) for {path}, got {features.shape}"
            )
        if self.transform is not None:
            features = self.transform(features)

        text_path = path.with_suffix(".txt")
        captions: list[str] = []
        if text_path.exists():
            captions = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        joints_path = self.root / f"{sample_id}_joints.npy"
        joints: np.ndarray | None = np.load(joints_path) if joints_path.exists() else None

        return {
            "features": features,
            # Expose under "motion" so dict-aware helpers (Visualizer.visualize_batch,
            # motion_collate-style routing) can pass the features straight through to
            # ``Visualizer.visualize`` which auto-detects the 263-D layout.
            "motion": features,
            "text": captions,
            "joints": joints,
            "name": sample_id,
            "path": str(path),
        }

    def names(self) -> list[str]:
        return [p.stem for p in self.files]

    def __repr__(self) -> str:
        return f"HumanML3DDataset(root={self.root!s}, n={len(self)}, feature_dim={self.feature_dim})"
