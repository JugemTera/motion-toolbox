# Dataloaders

The ``motion_toolbox.data`` module ships PyTorch ``Dataset`` subclasses
that wrap :func:`motion_toolbox.io.load_motion`. They cover the three
shapes of dataset we use day-to-day:

| Class                  | What it expects on disk                                     | Returns per item                                                                       |
|------------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| ``MotionFolderDataset``| Any folder of ``.npy``/``.npz``/``.pt`` motion files        | ``{"motion": Motion, "name": str, "path": str}``                                       |
| ``Format272Dataset``   | A ``.npy`` folder using the 272-D representation            | Same as above; ``Mean.npy`` / ``Std.npy`` exposed as attributes                        |
| ``HumanML3DDataset``   | HumanML3D layout (``<id>.npy`` + ``<id>.txt`` + ``_joints``)| ``{"features", "motion", "text", "joints", "name", "path"}`` — raw 263-D features      |

All datasets defer file I/O to ``__getitem__`` so they are cheap to
construct even on large corpora. ``MotionFolderDataset`` and
``Format272Dataset`` return :class:`~motion_toolbox.core.motion.Motion`
objects so the rest of the toolbox (visualizers, converters,
evaluation) works on dataset samples without any glue code.
``HumanML3DDataset`` returns the raw 263-D feature array under both
``"features"`` and ``"motion"`` so it can be fed straight to a model and
also visualised via :meth:`Visualizer.visualize_batch` (which
auto-detects the 263-D layout).

## Installation

The data module is part of the core package, but it imports
``torch.utils.data.Dataset``, so you need an install that includes
PyTorch:

```bash
pip install -e ".[ml]"
# or
uv sync --extra ml
```

## Quick start

```python
from motion_toolbox.data import MotionFolderDataset, motion_collate
from torch.utils.data import DataLoader

dataset = MotionFolderDataset(
    "sample_data/format_272",
    pattern="*.npy",
    format="272d",            # optional, overrides auto-detection
)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=motion_collate)

for batch in loader:
    rotations   = batch["joint_rotations"]   # (B, T_max, J, 3)
    translation = batch["root_translation"]  # (B, T_max, 3)
    mask        = batch["mask"]              # (B, T_max) bool
    lengths     = batch["lengths"]           # (B,) int64
    names       = batch["names"]             # list[str]
    raw_motions = batch["motions"]           # list[Motion]  (handy for visualizer)
```

The collate function pads to the longest sequence in the batch and
produces a boolean ``mask`` you can feed into transformer attention or
loss masking.

## 272-D dataset (MotionMillion / RoMo format)

```python
from motion_toolbox.data import Format272Dataset

dataset = Format272Dataset("sample_data/format_272")
print(dataset)                                 # n=3, no stats / with mean/std
print(dataset[0]["motion"].num_joints)          # 22 (HumanML3D layout)

if dataset.mean is not None:
    feats = (np.load(dataset.files[0]) - dataset.mean) / dataset.std
```

``Format272Dataset`` automatically skips ``Mean.npy`` / ``Std.npy`` when
iterating and exposes them through the ``mean`` / ``std`` attributes so
you can reproduce HumanML3D-style normalisation.

## HumanML3D dataset (motion + text + joints)

```python
from motion_toolbox.data import HumanML3DDataset

dataset = HumanML3DDataset(
    "path/to/humanml3d/train",
    split_ids=["000000", "000001"],   # optional split filter
)
sample = dataset[0]
print(sample["features"].shape)  # (T, 263) — RIC features
print(sample["text"])            # list[str], one entry per caption
print(sample["joints"])          # (T, 22, 3) np.ndarray or None
```

Note: a true ``Motion`` reconstruction from 263-D RIC features needs
SMPLify optimisation (see ``HumanML3DToSMPLConverter``), so this
dataset deliberately keeps the features as numpy. Pre-normalise with
``dataset.mean`` / ``dataset.std`` (when available) before training.

## Connecting datasets to the visualizer

The unified :class:`~motion_toolbox.visualization.Visualizer` understands
any of these datasets directly:

```python
from motion_toolbox.visualization import Visualizer

Visualizer().visualize_batch(
    dataset,
    output_dir="outputs/dataset_preview",
    limit=8,
    fps=20,
)
```

This writes one HTML viewer per sample under
``outputs/dataset_preview/`` using the dataset's ``"name"`` field as the
filename.

## Adding transforms

All datasets accept a ``transform`` callable that receives the loaded
:class:`Motion` and returns a (possibly modified) :class:`Motion`:

```python
def crop_to_120_frames(motion):
    if motion.num_frames <= 120:
        return motion
    start = (motion.num_frames - 120) // 2
    return motion.slice_frames(start, start + 120)

dataset = MotionFolderDataset("path/to/folder", transform=crop_to_120_frames)
```

Because the transform sees a real :class:`Motion`, you have its full API
to slice, downsample, normalise quaternions, change framerate, etc.

## Notes on batching

* All motions in a batch must share the same number of joints (the same
  skeleton). Either filter your dataset accordingly, or build separate
  loaders per skeleton.
* ``motion_collate`` pads with zeros and reports the true sequence
  length via ``mask`` / ``lengths``; downstream models should respect
  them.
* The collated dict also keeps the original ``Motion`` objects in
  ``batch["motions"]`` so you can call ``Visualizer().visualize(motion,
  …)`` on individual samples without re-loading from disk.
