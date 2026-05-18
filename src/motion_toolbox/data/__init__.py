"""PyTorch datasets and collate helpers for motion data.

This module exposes :class:`torch.utils.data.Dataset` subclasses that wrap
:func:`motion_toolbox.io.load_motion` so users can train and evaluate models
directly against directories of motion files (HumanML3D, the ROMO 272-D
corpus, AMASS-SMPL, GVHMR, STMC, ALIGN, MHR — anything ``load_motion`` can
parse).

The :func:`motion_collate` helper pads variable-length motions into a batched
tensor dict ready to feed into a model.

Examples
--------
>>> from torch.utils.data import DataLoader
>>> from motion_toolbox.data import MotionFolderDataset, motion_collate
>>>
>>> dataset = MotionFolderDataset("path/to/motions/", pattern="*.npy")
>>> loader = DataLoader(
...     dataset,
...     batch_size=8,
...     shuffle=True,
...     collate_fn=motion_collate,
... )
>>> for batch in loader:
...     rotations = batch["joint_rotations"]        # (B, T_max, J, 3)
...     translation = batch["root_translation"]     # (B, T_max, 3)
...     lengths = batch["lengths"]                  # (B,)
...     mask = batch["mask"]                        # (B, T_max) bool

The dataset classes return :class:`~motion_toolbox.core.motion.Motion`
objects (optionally wrapped in a metadata dict) so they integrate with
:class:`~motion_toolbox.visualization.Visualizer` without any conversion:

>>> from motion_toolbox.visualization import Visualizer
>>> vis = Visualizer()
>>> vis.visualize_batch(dataset, output_dir="outputs/dataset_preview", limit=8)
"""

from .collate import motion_collate
from .dataset import MotionFolderDataset
from .format_272 import Format272Dataset
from .humanml3d import HumanML3DDataset

__all__ = [
    "MotionFolderDataset",
    "Format272Dataset",
    "HumanML3DDataset",
    "motion_collate",
]
