"""End-to-end dataset → visualization demo.

Iterates over the bundled 272-D sample folder with
:class:`motion_toolbox.data.Format272Dataset` and renders an HTML viewer
per sample using :class:`motion_toolbox.visualization.Visualizer`.

Run from the repository root:

    uv run python demos/dataset_to_visualization_demo.py

Or, without ``uv``::

    PYTHONPATH=src python3 demos/dataset_to_visualization_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from motion_toolbox.data import Format272Dataset, motion_collate  # noqa: E402
from motion_toolbox.visualization import Visualizer  # noqa: E402


def main() -> int:
    sample_dir = _REPO / "sample_data" / "format_272"
    output_dir = _REPO / "outputs" / "dataset_to_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DATASET → VISUALIZER DEMO")
    print("=" * 80)

    dataset = Format272Dataset(sample_dir)
    print(f"\nLoaded {dataset}")

    sample = dataset[0]
    motion = sample["motion"]
    print(
        f"First sample '{sample['name']}': {motion.num_frames} frames, "
        f"{motion.num_joints} joints, {motion.framerate:.1f} fps"
    )

    # Show that motion_collate yields ML-ready tensors.
    try:
        import torch  # noqa: F401  (only needed if torch is installed)

        batch = motion_collate([dataset[i] for i in range(min(2, len(dataset)))])
        print(
            f"\nCollated batch: joint_rotations={tuple(batch['joint_rotations'].shape)}, "
            f"mask sum (real frames)={int(batch['mask'].sum())}, "
            f"lengths={batch['lengths'].tolist()}"
        )
    except ImportError:
        print("\n(torch not installed; skipping motion_collate demo)")

    # Render the dataset as a directory of HTML viewers via the unified API.
    print(f"\nRendering HTML viewers to {output_dir.relative_to(_REPO)} ...")
    paths = Visualizer().visualize_batch(dataset, output_dir, fps=20)
    for path in paths:
        print(f"  Wrote {path.relative_to(_REPO)}")

    print("\nDone.")
    print(f"Open the first viewer: file://{paths[0].resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
