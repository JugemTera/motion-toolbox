"""Demo: Separate Viewers Grid.

This demo shows how to create a grid of *independent* HTML viewer instances,
one per animation. Each viewer has its own playback controls and can be
interacted with separately (in contrast to ``spatial_animation_grid_demo.py``
which places every character in a single shared 3D scene).

The pipeline:

1. Load each ``.pt`` motion sample and convert it to a :class:`Motion`.
2. Export each Motion to a ``.glb`` using :class:`GLBExporter` (requires SMPL).
3. Hand the GLB paths to :func:`create_separate_viewers_grid` which writes a
   single HTML page that embeds the GLBs into a grid of ``<model-viewer>``
   instances.

Prerequisites:
    - SMPL body models on disk. Either set ``SMPL_BODY_MODELS_PATH`` or
      place them at ``~/.motion_toolbox/datasets/body_models_clean_pkl``.
      See ``docs/SMPL_MODELS_SETUP.md``.
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from motion_toolbox.converters import GVHMRToMotionConverter
from motion_toolbox.visualization import create_separate_viewers_grid
from motion_toolbox.visualization.glb_exporter import GLBExporter


def find_body_models() -> str | None:
    """Locate SMPL body models on disk.

    Honors ``SMPL_BODY_MODELS_PATH`` first, then falls back to common locations.
    """
    env = os.environ.get("SMPL_BODY_MODELS_PATH")
    if env and Path(env).exists():
        return env

    candidates = [
        Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl",
        Path.home() / ".motion_toolbox/datasets/body_models",
        Path.home() / ".model",
        _REPO / ".model",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def main() -> int:
    print("=" * 70)
    print("SEPARATE VIEWERS GRID DEMO")
    print("=" * 70)
    print("\nBuilds a grid of independent 3D viewers, each with its own")
    print("animation and playback controls.\n")

    sample_dir = _REPO / "sample_data" / "gvhmr"
    pt_files = sorted(sample_dir.glob("*.pt"))[:4]
    if not pt_files:
        print(f"No sample .pt files found in {sample_dir}.")
        print("Add a few GVHMR-format .pt files to test with.")
        return 1
    print(f"Found {len(pt_files)} sample motion file(s).")

    body_model_path = find_body_models()
    if not body_model_path:
        print("\nSMPL body models not found.")
        print("Download from https://smpl.is.tue.mpg.de/download.php and either")
        print("  - export SMPL_BODY_MODELS_PATH=/path/to/body_models, or")
        print("  - install to ~/.motion_toolbox/datasets/body_models_clean_pkl/")
        return 1
    print(f"Using SMPL body models at: {body_model_path}\n")

    output_dir = _REPO / "outputs" / "separate_viewers_demo"
    glb_dir = output_dir / "glbs"
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_dir.mkdir(parents=True, exist_ok=True)

    print("Converting motions and exporting GLBs ...")
    gvhmr = GVHMRToMotionConverter(body_model_path=body_model_path)
    exporter = GLBExporter(model_path=body_model_path)

    glb_paths: list[str] = []
    labels: list[str] = []
    for idx, pt_file in enumerate(pt_files, start=1):
        motion = gvhmr.convert(str(pt_file))
        glb_path = glb_dir / f"{pt_file.stem}.glb"
        exporter.export(motion, str(glb_path))
        glb_paths.append(str(glb_path))
        labels.append(f"Animation {idx}")
        print(f"  [{idx}/{len(pt_files)}] {pt_file.name} -> {glb_path.name}")

    html_path = output_dir / "separate_viewers_grid.html"
    print(f"\nAssembling grid HTML ({len(glb_paths)} viewers, 2 columns) ...")
    create_separate_viewers_grid(
        glb_paths=glb_paths,
        output_path=html_path,
        title="Separate Viewers Grid Demo",
        labels=labels,
        grid_cols=2,
    )

    print("\nDone.")
    print(f"  HTML viewer: {html_path}")
    print(f"  GLB files:   {glb_dir}\n")
    print("Open the HTML file in a browser to view the grid of independent")
    print("3D viewers (each has its own playback controls).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
