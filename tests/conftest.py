"""Pytest configuration: optional skips for heavy / optional dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest


def _has_smpl_models() -> bool:
    p = Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl"
    return p.is_dir() and any(p.iterdir())


def _has_mhr_assets() -> bool:
    p = Path.home() / ".motion_toolbox/datasets/mhr/assets"
    return p.is_dir() and any(p.iterdir())


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[no-untyped-def]
    skip_smpl = pytest.mark.skip(reason="SMPL body models not found (~/.motion_toolbox/datasets/body_models_clean_pkl)")
    skip_mhr = pytest.mark.skip(reason="MHR assets not found (~/.motion_toolbox/datasets/mhr/assets)")
    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "requires_smpl" in item.keywords and not _has_smpl_models():
            item.add_marker(skip_smpl)
        if "requires_mhr" in item.keywords and not _has_mhr_assets():
            item.add_marker(skip_mhr)
        if "gpu" in item.keywords and not _cuda_available():
            item.add_marker(skip_gpu)
