"""Demo: Keyframe Overlay Viewer

This demo shows how to create a visualization of a motion sequence with overlaid
keyframe poses. Keyframes are positioned at their actual spatial locations in the
sequence, creating a visual "trail" effect that shows motion progression.

Features:
- Main animation with pre-computed vertex positions
- 10 uniformly sampled keyframe poses overlaid at their actual positions
- Interactive opacity control for keyframes
- Two display modes: Overlap (all keyframes visible) and Disappear (keyframes fade as animation passes)
- Customizable materials (Phong, Lambert, Physical)
- Customizable colors (White, Blue, Clay, Grey)
- Checkerboard floor for spatial reference

Default Settings (optimized for research papers):
- Physical material (realistic PBR rendering)
- Blue color (cornflower blue #6495ED)
- Checkerboard floor (1m × 1m grid)
- 10 keyframes uniformly sampled
- 50% initial keyframe opacity
"""

import sys
from pathlib import Path

# Add project root to path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from motion_toolbox.visualization import create_keyframe_overlay


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
    """Run the keyframe overlay demo."""
    print("=" * 70)
    print("KEYFRAME OVERLAY VIEWER DEMO")
    print("=" * 70)
    print("\nCreates a visualization with animated character and overlaid")
    print("keyframe poses positioned at their actual spatial locations.\n")

    # Find sample data
    sample_dir = _REPO / "sample_data" / "gvhmr"
    pt_files = sorted(sample_dir.glob("*.pt"))

    if not pt_files:
        print("❌ No sample PT files found in motion_toolbox/sample_data/gvhmr/")
        print("   Please add some .pt motion files to test with.")
        return 1

    # Use the first sample file
    motion_file = pt_files[0]
    print(f"✅ Found sample motion file: {motion_file.name}")

    # Find SMPL models
    body_model_path = find_body_models()
    if not body_model_path:
        print("\n❌ SMPL body models not found!")
        print("   Download from: https://smpl.is.tue.mpg.de/download.php")
        print("   Install to: ~/.motion_toolbox/datasets/body_models_clean_pkl/")
        return 1

    print(f"✅ Found SMPL models at: {body_model_path}\n")

    # Setup output directory
    output_dir = _REPO / "outputs" / "keyframe_overlay_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Creating keyframe overlay with DEFAULT settings")
    print("=" * 70)
    print("  Material: Physical (realistic PBR)")
    print("  Color: Blue (cornflower blue)")
    print("  Floor: Checkerboard (1m × 1m grid)")
    print("  Keyframes: 10 (uniformly sampled)")
    print("  Opacity: 50% (adjustable in viewer)")
    print("  Modes: Overlap & Disappear (toggle in viewer)\n")

    # Create visualization with defaults
    create_keyframe_overlay(
        motion_path=str(motion_file),
        output_html_path=output_dir / "keyframe_overlay.html",
        title="Keyframe Overlay Demo",
        body_model_path=body_model_path,
        num_keyframes=10,
        keyframe_opacity=0.5,
        sampling_strategy="uniform",
        # Note: Using all defaults - blue, physical, checkerboard
    )

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🌐 HTML viewer: {output_dir}/keyframe_overlay.html\n")

    print("🎬 Open the HTML file in a browser to view:")
    print("   • Main animation with overlaid keyframe poses")
    print("   • Keyframes positioned at their actual locations in sequence")
    print("   • Blue characters with realistic Physical material")
    print("   • 1m × 1m checkerboard floor for spatial reference")
    print("   • Interactive opacity slider (0-100%)")
    print("   • Two display modes:")
    print("     - Overlap Mode: All keyframes visible (shows motion trail)")
    print("     - Disappear Mode: Keyframes disappear as animation passes\n")

    print("💡 Customization examples:")
    print("   # More keyframes with white color")
    print("   create_keyframe_overlay(..., num_keyframes=20, mesh_color='white')")
    print("")
    print("   # Lower opacity with clay color")
    print("   create_keyframe_overlay(..., keyframe_opacity=0.3, mesh_color='clay')")
    print("")
    print("   # Different material with solid floor")
    print("   create_keyframe_overlay(..., material_type='phong', checkerboard_floor=False)")
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
