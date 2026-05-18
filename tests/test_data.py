"""Tests for ``motion_toolbox.data`` (datasets + collate)."""
# ruff: noqa: I001, E402

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motion_toolbox.core.motion import Motion  # noqa: E402
from motion_toolbox.data import (  # noqa: E402
    Format272Dataset,
    HumanML3DDataset,
    MotionFolderDataset,
    motion_collate,
)


SAMPLE_272_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "format_272"


# ---------------------------------------------------------------------------
# MotionFolderDataset / Format272Dataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_272_DIR.exists(), reason="bundled 272-D sample data missing")
def test_motion_folder_dataset_iterates_272(tmp_path):
    dataset = MotionFolderDataset(SAMPLE_272_DIR, pattern="*.npy", format="272d")
    assert len(dataset) >= 1

    sample = dataset[0]
    assert set(sample.keys()) == {"motion", "name", "path"}
    motion = sample["motion"]
    assert isinstance(motion, Motion)
    assert motion.joint_rotations.shape[-1] == 3
    assert motion.num_frames > 0


@pytest.mark.skipif(not SAMPLE_272_DIR.exists(), reason="bundled 272-D sample data missing")
def test_format272_dataset_skips_stats(tmp_path):
    dataset = Format272Dataset(SAMPLE_272_DIR)
    names = dataset.names()
    assert "Mean" not in names
    assert "Std" not in names
    # mean / std are exposed when their files exist; missing-side is OK.
    if dataset.mean is not None:
        assert dataset.mean.ndim >= 1
    if dataset.std is not None:
        assert dataset.std.ndim >= 1


def test_motion_folder_dataset_missing_root(tmp_path):
    missing = tmp_path / "no-such-folder"
    with pytest.raises(FileNotFoundError):
        MotionFolderDataset(missing)


def test_motion_folder_dataset_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        MotionFolderDataset(empty)


def test_motion_folder_dataset_transform(tmp_path):
    if not SAMPLE_272_DIR.exists():
        pytest.skip("bundled 272-D sample data missing")

    calls: list[int] = []

    def truncate(motion: Motion) -> Motion:
        calls.append(motion.num_frames)
        return motion.slice_frames(0, min(8, motion.num_frames))

    dataset = MotionFolderDataset(SAMPLE_272_DIR, pattern="*.npy", format="272d", transform=truncate)
    sample = dataset[0]
    assert sample["motion"].num_frames == 8
    assert calls  # transform was applied


# ---------------------------------------------------------------------------
# HumanML3DDataset
# ---------------------------------------------------------------------------


def test_humanml3d_dataset_layout(tmp_path):
    np.save(tmp_path / "Mean.npy", np.zeros((263,), dtype=np.float32))
    np.save(tmp_path / "Std.npy", np.ones((263,), dtype=np.float32))

    feats = np.zeros((12, 263), dtype=np.float32)
    np.save(tmp_path / "000000.npy", feats)
    np.save(tmp_path / "000000_joints.npy", np.zeros((12, 22, 3), dtype=np.float32))
    (tmp_path / "000000.txt").write_text("a person walks forward\nthe character takes a step\n")

    dataset = HumanML3DDataset(tmp_path)
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["name"] == "000000"
    assert sample["features"].shape == (12, 263)
    # "motion" mirrors "features" so dict-aware visualizer helpers pass features through.
    assert sample["motion"] is sample["features"]
    assert sample["text"] == ["a person walks forward", "the character takes a step"]
    assert sample["joints"].shape == (12, 22, 3)
    assert dataset.mean is not None and dataset.std is not None


def test_humanml3d_dataset_filters_split(tmp_path):
    for sid in ("000000", "000001", "000002"):
        np.save(tmp_path / f"{sid}.npy", np.zeros((4, 263), dtype=np.float32))

    dataset = HumanML3DDataset(tmp_path, split_ids=["000000", "000002"])
    assert sorted(dataset.names()) == ["000000", "000002"]


# ---------------------------------------------------------------------------
# motion_collate
# ---------------------------------------------------------------------------


def _make_motion(num_frames: int, num_joints: int = 4) -> Motion:
    from motion_toolbox.core.skeleton import Skeleton

    joint_names = tuple(f"j{i}" for i in range(num_joints))
    parents = (None,) + tuple(range(num_joints - 1))
    offsets = np.zeros((num_joints, 3), dtype=np.float32)
    offsets[1:, 1] = 0.1  # arbitrary non-trivial offsets
    skeleton = Skeleton(joint_names, parents, offsets)
    return Motion(
        joint_rotations=np.zeros((num_frames, num_joints, 3), dtype=np.float32),
        root_translation=np.zeros((num_frames, 3), dtype=np.float32),
        skeleton=skeleton,
        framerate=30.0,
        floor_alignment="none",
    )


def test_motion_collate_pads_to_max_length():
    samples = [_make_motion(4), _make_motion(7), _make_motion(5)]
    batch = motion_collate(samples)
    assert batch["joint_rotations"].shape == (3, 7, 4, 3)
    assert batch["root_translation"].shape == (3, 7, 3)
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"].shape == (3, 7)
    assert batch["mask"].sum().item() == 4 + 7 + 5
    assert batch["lengths"].tolist() == [4, 7, 5]
    assert batch["framerate"].tolist() == [30.0, 30.0, 30.0]


def test_motion_collate_accepts_metadata_dicts():
    samples = [
        {"motion": _make_motion(3), "name": "a", "path": "/tmp/a"},
        {"motion": _make_motion(5), "name": "b", "path": "/tmp/b"},
    ]
    batch = motion_collate(samples)
    assert batch["names"] == ["a", "b"]
    assert batch["paths"] == ["/tmp/a", "/tmp/b"]
    assert batch["lengths"].tolist() == [3, 5]


def test_motion_collate_rejects_mixed_joint_counts():
    with pytest.raises(ValueError, match="same number of joints"):
        motion_collate([_make_motion(4, num_joints=4), _make_motion(4, num_joints=6)])


def test_motion_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        motion_collate([])
