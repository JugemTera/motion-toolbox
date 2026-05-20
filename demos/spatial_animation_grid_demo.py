"""Demo: Spatial Animation Grid Viewer

This demo shows how to create a single 3D scene with multiple animated characters
arranged in a spatial grid. All characters share the same camera, lighting, and
playback controls.

Features:
- Single unified 3D scene
- Multiple characters positioned in a grid
- Synchronized playback
- Customizable materials (Phong, Lambert, Physical)
- Customizable colors (White, Blue, Clay, Grey)
- Optional checkerboard floor for spatial reference

Default Settings (optimized for research papers):
- Physical material (realistic PBR rendering)
- Blue color (cornflower blue #6495ED)
- Checkerboard floor (1m × 1m grid)
- Enhanced antialiasing
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from motion_toolbox.visualization import create_spatial_animation_grid


def find_body_models():
    """Find SMPL body models in common locations."""
    possible_paths = [
        Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl",
        Path.home() / ".motion_toolbox/datasets/body_models",
        Path.home() / ".model",
        _REPO / ".model",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


def main():
    """Run the spatial animation grid demo."""
    print("=" * 70)
    print("SPATIAL ANIMATION GRID DEMO")
    print("=" * 70)
    print("\nCreates a single 3D scene with multiple animated characters")
    print("arranged in a spatial grid with synchronized playback.\n")

    # Find sample data
    sample_dir = _REPO / "sample_data" / "gvhmr"
    pt_files = sorted(sample_dir.glob("*.pt"))[:4]  # Use up to 4 samples

    if not pt_files:
        print("❌ No sample PT files found in sample_data/gvhmr/")
        print("   Please add some .pt motion files to test with.")
        return 1

    print(f"✅ Found {len(pt_files)} sample motion files")

    # Find SMPL models
    body_model_path = find_body_models()
    if not body_model_path:
        print("\n❌ SMPL body models not found!")
        print("   Download from: https://smpl.is.tue.mpg.de/download.php")
        print("   Install to: ~/.motion_toolbox/datasets/body_models_clean_pkl/")
        return 1

    print(f"✅ Found SMPL models at: {body_model_path}\n")

    # Setup output directory
    output_dir = _REPO / "outputs" / "spatial_grid_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Creating spatial grid with DEFAULT settings")
    print("=" * 70)
    print("  Material: Physical (realistic PBR)")
    print("  Color: Blue (cornflower blue)")
    print("  Floor: Checkerboard (1m × 1m grid)")
    print("  Antialiasing: Enhanced\n")

    # Create visualization with defaults
    create_spatial_animation_grid(
        motion_paths=[str(p) for p in pt_files],
        output_dir=output_dir / "glbs",
        output_html_path=output_dir / "spatial_grid.html",
        title="Spatial Animation Grid Demo",
        body_model_path=body_model_path,
        spacing=2.5,  # 2.5m spacing between characters
        # Note: Using all defaults - blue, physical, checkerboard
    )

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🌐 HTML viewer: {output_dir}/spatial_grid.html")
    print(f"📦 GLB files: {output_dir}/glbs/\n")

    print("🎬 Open the HTML file in a browser to view:")
    print("   • Multiple characters in a single 3D scene")
    print("   • Blue characters with realistic Physical material")
    print("   • 1m × 1m checkerboard floor for spatial reference")
    print("   • Smooth antialiased edges")
    print("   • Synchronized playback with shared camera controls\n")

    print("💡 Customization examples:")
    print("   # White characters with Phong shading")
    print("   create_spatial_animation_grid(..., material_type='phong', mesh_color='white')")
    print("")
    print("   # Clay color with solid floor")
    print("   create_spatial_animation_grid(..., mesh_color='clay', checkerboard_floor=False)")
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
