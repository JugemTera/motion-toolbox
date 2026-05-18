"""Demo: 272D Motion Representation Visualization.

Loads 272-D motion files from ``sample_data/format_272/``, converts them
through ``Format272ToMotionConverter``, and visualises each one as a
skeleton HTML viewer.

Run from the repository root:

    uv run python demos/format_272_visualization_demo.py

Or without uv:

    PYTHONPATH=src python3 demos/format_272_visualization_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from motion_toolbox.converters.motion_converters import Format272ToMotionConverter  # noqa: E402
from motion_toolbox.data import Format272Dataset  # noqa: E402
from motion_toolbox.visualization import Visualizer  # noqa: E402


def main() -> int:
    print("=" * 80)
    print("272D MOTION REPRESENTATION VISUALIZATION DEMO")
    print("=" * 80)

    sample_dir = _REPO / "sample_data" / "format_272"
    output_dir = _REPO / "outputs" / "format_272_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in sample_dir.glob("[0-9]*.npy"))
    if not files:
        print(f"\nNo 272D sample data found in {sample_dir}.")
        print("Place .npy files there (see sample_data/format_272/README.md) or")
        print("regenerate them with scripts/convert_humanml3d_to_272.py.")
        return 1

    files = files[:3]
    print(f"\nFound {len(files)} sample(s); visualising:")
    for f in files:
        data = np.load(f)
        print(f"  - {f.name}: {data.shape[0]} frames, {data.shape[1]} features")

    # Demonstrate the dataset-driven path (single API, no manual conversion).
    print("\n[1/2] Dataset-driven path:")
    dataset = Format272Dataset(sample_dir)
    vis = Visualizer()
    html_paths = vis.visualize_batch(dataset, output_dir, limit=len(files), fps=30)
    for path in html_paths:
        print(f"  Wrote {path.relative_to(_REPO)}")

    # Demonstrate the manual converter path as well, in case users want it.
    print("\n[2/2] Manual converter path:")
    converter = Format272ToMotionConverter(framerate=30.0)
    for f in files:
        motion = converter.convert(np.load(f))
        print(
            f"  Converted {f.name}: {motion.num_frames} frames, {motion.num_joints} joints, "
            f"{motion.framerate:.1f} fps"
        )

    print("\nDone.")
    print(f"Open: file://{(output_dir / (files[0].stem + '.html')).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
