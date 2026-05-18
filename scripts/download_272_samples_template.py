#!/usr/bin/env python3
"""Template: download 272-D motion samples from a public Hugging Face dataset.

This is the focused 272-D equivalent of
``scripts/fetch_sample_data_template.sh``. It is intentionally a *template* —
fill in the placeholders below to wire it to your published Hugging Face
dataset.

Usage::

    # 1) Edit HF_USER / DATASET_NAME (or pass via env / CLI).
    # 2) Ensure ``huggingface_hub`` is installed (it ships with ``[ml]``).
    # 3) Run:
    python scripts/download_272_samples_template.py
    python scripts/download_272_samples_template.py --hf-user your-org --dataset motion-toolbox-samples

Alternative sources (if no public mirror is available yet):

* Generate 272-D features from SMPL data using
  ``motion_toolbox.converters.motion_converters.SMPLTo272Converter``.
* Use the upstream 272-dim representation project:
  https://github.com/Li-xingXiao/272-dim-Motion-Representation

Place ``.npy`` files into ``sample_data/format_272/``; the toolbox's
:class:`motion_toolbox.data.Format272Dataset` and
:func:`motion_toolbox.io.load_motion` will pick them up automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "sample_data" / "format_272"

# ---------------------------------------------------------------------------
# CONFIGURE ME
# ---------------------------------------------------------------------------
DEFAULT_HF_USER = "<HF_USER>"                # e.g. "your-org"
DEFAULT_DATASET = "<DATASET_NAME>"           # e.g. "motion-toolbox-samples"
DEFAULT_REVISION = "main"
DEFAULT_SUBDIR = "format_272"                # subdirectory inside the HF repo
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-user", default=os.environ.get("HF_USER", DEFAULT_HF_USER))
    parser.add_argument("--dataset", default=os.environ.get("DATASET_NAME", DEFAULT_DATASET))
    parser.add_argument("--revision", default=os.environ.get("REVISION", DEFAULT_REVISION))
    parser.add_argument(
        "--subdir",
        default=os.environ.get("HF_PATH_IN_REPO", DEFAULT_SUBDIR),
        help="Subdirectory inside the HF dataset to fetch; empty = full dataset.",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    if args.hf_user in (DEFAULT_HF_USER, "") or args.dataset in (DEFAULT_DATASET, ""):
        print(
            "This is the placeholder template. Set --hf-user and --dataset (or the\n"
            "matching env vars) to point at your published mirror, e.g.\n"
            "    python scripts/download_272_samples_template.py \\\n"
            "        --hf-user your-org --dataset motion-toolbox-samples\n"
            "Until then, drop .npy files manually under "
            f"{args.dest.relative_to(REPO_ROOT)}/.",
            file=sys.stderr,
        )
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is required. Install with:\n"
            "    uv pip install huggingface_hub\n"
            "or include the [ml] extra: pip install -e '.[ml]'",
            file=sys.stderr,
        )
        return 2

    args.dest.mkdir(parents=True, exist_ok=True)
    repo_id = f"{args.hf_user}/{args.dataset}"
    allow_patterns = [f"{args.subdir.rstrip('/')}/*"] if args.subdir else None

    print(f"Downloading {repo_id}@{args.revision} → {args.dest}")
    local = snapshot_download(
        repo_id=repo_id,
        revision=args.revision,
        repo_type="dataset",
        local_dir=str(args.dest),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print(f"Snapshot synced into: {Path(local).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
