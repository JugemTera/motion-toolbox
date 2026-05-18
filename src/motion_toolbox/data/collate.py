"""Batch collate function for variable-length motion samples.

Pads sequences with zeros to the longest motion in the batch and returns a
flat dict of tensors plus a boolean ``mask`` indicating real (non-padded)
frames.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from ..core.motion import Motion

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch


def _extract_motion(sample: Any) -> tuple[Motion, dict[str, Any]]:
    """Return ``(motion, metadata_dict)`` from a dataset sample.

    Accepts either a bare :class:`Motion` or a dict that contains a
    ``"motion"`` key (the layout used by :class:`MotionFolderDataset` and
    friends). Any other keys are kept as per-sample metadata.
    """
    if isinstance(sample, Motion):
        return sample, {}
    if isinstance(sample, dict) and "motion" in sample:
        meta = {k: v for k, v in sample.items() if k != "motion"}
        return sample["motion"], meta
    raise TypeError(
        f"motion_collate expects Motion instances or dicts with a 'motion' key, got {type(sample).__name__}"
    )


def motion_collate(batch: list[Any]) -> dict[str, Any]:
    """Pad-and-stack a list of motion samples into batched tensors.

    Returns a dict with:

    * ``joint_rotations`` — ``(B, T_max, J, 3)`` float tensor
    * ``root_translation`` — ``(B, T_max, 3)`` float tensor
    * ``lengths`` — ``(B,)`` int64 tensor
    * ``mask`` — ``(B, T_max)`` bool tensor (``True`` = real frame)
    * ``framerate`` — ``(B,)`` float tensor
    * ``names`` — list of strings (when samples carry a ``"name"`` field)
    * ``text`` — list of caption lists (when samples carry a ``"text"`` field)

    All motions in a batch are expected to share the same number of joints
    (i.e. the same skeleton). A :class:`ValueError` is raised otherwise.
    """
    if not _HAS_TORCH:
        raise ImportError("motion_collate requires PyTorch. Install with: pip install 'motion-toolbox[ml]'")
    if not batch:
        raise ValueError("motion_collate received an empty batch.")

    motions: list[Motion] = []
    metas: list[dict[str, Any]] = []
    for sample in batch:
        motion, meta = _extract_motion(sample)
        motions.append(motion)
        metas.append(meta)

    num_joints_set = {m.joint_rotations.shape[1] for m in motions}
    if len(num_joints_set) != 1:
        raise ValueError(f"All motions in a batch must share the same number of joints, got {sorted(num_joints_set)}")
    num_joints = next(iter(num_joints_set))

    lengths = [int(m.joint_rotations.shape[0]) for m in motions]
    t_max = max(lengths)
    batch_size = len(motions)

    rot_arr = np.zeros((batch_size, t_max, num_joints, 3), dtype=np.float32)
    trans_arr = np.zeros((batch_size, t_max, 3), dtype=np.float32)
    mask_arr = np.zeros((batch_size, t_max), dtype=bool)

    for i, motion in enumerate(motions):
        t = lengths[i]
        rot_arr[i, :t] = np.asarray(motion.joint_rotations, dtype=np.float32)
        trans_arr[i, :t] = np.asarray(motion.root_translation, dtype=np.float32)
        mask_arr[i, :t] = True

    out: dict[str, Any] = {
        "joint_rotations": torch.from_numpy(rot_arr),
        "root_translation": torch.from_numpy(trans_arr),
        "lengths": torch.as_tensor(lengths, dtype=torch.long),
        "mask": torch.from_numpy(mask_arr),
        "framerate": torch.as_tensor([float(m.framerate) for m in motions], dtype=torch.float32),
        "motions": motions,
    }

    if any("name" in meta for meta in metas):
        out["names"] = [meta.get("name", "") for meta in metas]
    if any("text" in meta for meta in metas):
        out["text"] = [meta.get("text", []) for meta in metas]
    if any("joints" in meta for meta in metas):
        out["joints"] = [meta.get("joints") for meta in metas]
    if any("path" in meta for meta in metas):
        out["paths"] = [meta.get("path", "") for meta in metas]

    return out
