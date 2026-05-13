#!/usr/bin/env python3
"""272D sample download helper (optional).

This repository no longer points to a specific proprietary S3 bucket. To obtain
272-dimensional motion samples:

1. Use the public 272-dim representation project:
   https://github.com/Li-xingXiao/272-dim-Motion-Representation
2. Or generate 272D features from SMPL using the converters in this package.

Place downloaded ``.npy`` files under ``sample_data/format_272/``.

If you maintain a public HTTP mirror, set ``MOTION_TOOLBOX_N272_BASE_URL`` to a
prefix URL and implement your own download logic; this script only prints guidance.
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    base = os.environ.get("MOTION_TOOLBOX_N272_BASE_URL", "")
    out = Path(__file__).resolve().parent.parent / "sample_data" / "format_272"
    print("272D sample data is not downloaded automatically.")
    print(f"Target directory: {out}")
    print("See sample_data/format_272/README.md for sources and licenses.")
    if base:
        print(f"MOTION_TOOLBOX_N272_BASE_URL is set ({base}); add your download logic here if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
