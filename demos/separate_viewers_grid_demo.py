"""Demo: Separate Viewers Grid

This demo shows how to create a grid of independent HTML viewers, each displaying
a single animation. Each viewer has its own playback controls and can be
interacted with independently.

Features:
- Grid layout of multiple independent 3D viewers
- Each viewer has individual playback controls
- Useful for comparing multiple animations side-by-side
- Optional labels for each animation

Use Cases:
- Comparing different motion styles or quality levels
- Creating animation galleries
- Individual inspection of multiple sequences
"""

import os
import sys
from pathlib import Path

# Add project root to path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from motion_toolbox.visualization import create_separate_viewers_grid


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
    """Run the separate viewers grid demo."""
    print("=" * 70)
    print("SEPARATE VIEWERS GRID DEMO")
    print("=" * 70)
    print("\nCreates a grid of independent 3D viewers, each with its own")
    print("animation and playback controls.\n")
    
    # Find sample data
    sample_dir = _REPO / "sample_data" / "gvhmr"
    pt_files = sorted(sample_dir.glob("*.pt"))[:4]  # Use up to 4 samples
    
    if not pt_files:
        print("❌ No sample PT files found in motion_toolbox/sample_data/gvhmr/")
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
    output_dir = _REPO / "outputs" / "separate_viewers_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Creating grid of separate viewers")
    print("=" * 70)
    print(f"  Grid layout: 2 columns")
    print(f"  Total viewers: {len(pt_files)}\n")
    
    # Create labels for each animation
    labels = [f"Animation {i+1}" for i in range(len(pt_files))]
    
    # Create visualization
    create_separate_viewers_grid(
        motion_paths=[str(p) for p in pt_files],
        output_html_path=output_dir / "separate_viewers_grid.html",
        title="Separate Viewers Grid Demo",
        body_model_path=body_model_path,
        grid_cols=2,  # 2 columns
        labels=labels,
        viewer_width=600,
        viewer_height=600,
    )
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🌐 HTML viewer: {output_dir}/separate_viewers_grid.html\n")
    
    print("🎬 Open the HTML file in a browser to view:")
    print("   • Grid of independent 3D viewers")
    print("   • Each animation has its own playback controls")
    print("   • Useful for side-by-side comparison")
    print("   • Each viewer can be controlled independently\n")
    
    print("💡 Customization examples:")
    print("   # Create a 3-column grid")
    print("   create_separate_viewers_grid(..., grid_cols=3)")
    print("")
    print("   # Larger viewers")
    print("   create_separate_viewers_grid(..., viewer_width=800, viewer_height=800)")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

