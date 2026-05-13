"""Demo: MHR (Momentum Human Rig) Visualization.

Visualizes MHR motion data from .pt files.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# MHR assets location (created by setup_mhr.sh)
MHR_ASSETS_DIR = Path.home() / ".motion_toolbox" / "datasets" / "mhr" / "assets"


def main():
    print("=" * 70)
    print("MHR VISUALIZATION DEMO")
    print("=" * 70)

    from motion_toolbox.visualization import create_spatial_animation_grid

    # Check MHR assets
    if not MHR_ASSETS_DIR.exists():
        print(f"❌ MHR assets not found at {MHR_ASSETS_DIR}")
        print("   Run: bash scripts/setup_mhr.sh")
        return 1

    # Find MHR data files
    mhr_data_dir = project_root / "sample_data" / "mhr"
    mhr_files = sorted(mhr_data_dir.glob("*.pt"))

    if not mhr_files:
        print(f"❌ No MHR data files found in {mhr_data_dir}")
        return 1

    print(f"\n📂 Found {len(mhr_files)} MHR file(s):")
    for f in mhr_files:
        print(f"   • {f.name}")

    # Create visualization
    output_dir = project_root / "outputs" / "mhr_demo"

    print(f"\n🎬 Creating visualization...")

    create_spatial_animation_grid(
        motion_paths=[str(f) for f in mhr_files],
        output_dir=output_dir / "glbs",
        output_html_path=output_dir / "viewer.html",
        title="MHR Visualization",
        body_model_path=str(MHR_ASSETS_DIR.parent),
        body_model_type="mhr",
        spacing=2.5,
    )

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"\n🌐 Open: {output_dir}/viewer.html")

    return 0


if __name__ == "__main__":
    sys.exit(main())
