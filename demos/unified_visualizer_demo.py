"""Demo: Unified Visualizer API

This demo showcases the new unified visualization system that works
across all motion formats and body models with a single, consistent API.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motion_toolbox.visualization import Visualizer
from motion_toolbox.visualization.constants import DEFAULT_FPS
from motion_toolbox.core import Motion, Skeleton


def main():
    """Run unified visualizer demo."""
    output_dir = Path("outputs/unified_visualizer_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Unified Visualizer Demo")
    print("=" * 70)
    print()
    
    # Create visualizer (single instance for all visualizations)
    vis = Visualizer()
    
    # Demo 1: HumanML3D 263D format
    print("1. Visualizing HumanML3D 263D format as skeleton...")
    data_263d = np.random.randn(100, 263)
    output_1 = vis.visualize(
        data_263d,
        output_dir / "263d_skeleton.html",
        mode='skeleton',
        title="HumanML3D 263D Animation",
    )
    print(f"   ✓ Created: {output_1}")
    print()
    
    # Demo 2: 272D format
    print("2. Visualizing 272D format as skeleton...")
    data_272d = np.random.randn(100, 272)
    output_2 = vis.visualize(
        data_272d,
        output_dir / "272d_skeleton.html",
        mode='skeleton',
        title="272D Format Animation",
    )
    print(f"   ✓ Created: {output_2}")
    print()
    
    # Demo 3: Joint positions
    print("3. Visualizing joint positions...")
    positions = np.random.randn(100, 22, 3)
    output_3 = vis.visualize(
        positions,
        output_dir / "joints_skeleton.html",
        mode='skeleton',
        title="Joint Positions",
    )
    print(f"   ✓ Created: {output_3}")
    print()
    
    # Demo 4: Motion object
    print("4. Visualizing Motion object...")
    skeleton = Skeleton.from_humanml3d()
    rotations = np.random.randn(100, 22, 3) * 0.1
    translation = np.zeros((100, 3))
    motion = Motion(rotations, translation, skeleton, floor_alignment='joint')
    
    output_4 = vis.visualize(
        motion,
        output_dir / "motion_skeleton.html",
        mode='skeleton',
        title="Motion Object",
    )
    print(f"   ✓ Created: {output_4}")
    print()
    
    # Demo 5: AUTO mode
    print("5. Using AUTO mode (automatically chooses best visualization)...")
    output_5 = vis.visualize(
        data_263d,
        output_dir / "auto_mode.html",
        mode='auto',  # Automatically chooses skeleton for 263D
        title="Auto Mode Demo",
    )
    print(f"   ✓ Created: {output_5}")
    print()
    
    # Demo 6: Export scene data for frontend
    print("6. Exporting scene data for frontend integration...")
    scene_data = vis.to_scene_data(data_263d, mode='skeleton', fps=DEFAULT_FPS)
    print(f"   ✓ Scene data exported:")
    print(f"      - Type: {scene_data['type']}")
    print(f"      - Frames: {scene_data['num_frames']}")
    print(f"      - Joints: {scene_data['num_joints']}")
    print(f"      - FPS: {scene_data['fps']}")
    print()
    
    # Demo 7: Error handling with fallback
    print("7. Testing automatic fallback to skeleton...")
    vis_fallback = Visualizer(on_incompatible='auto_skeleton')
    
    # Try to visualize joint positions as mesh (not possible - no rotations)
    # Should automatically fall back to skeleton
    try:
        output_7 = vis_fallback.visualize(
            positions,
            output_dir / "fallback_demo.html",
            mode='mesh',
            body_model='smpl',
        )
        print(f"   ✓ Fell back to skeleton: {output_7}")
    except NotImplementedError as e:
        print(f"   ℹ Expected behavior: {e}")
    print()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nKey features demonstrated:")
    print("  ✓ Automatic format detection")
    print("  ✓ Single API for all formats")
    print("  ✓ Multiple visualization modes")
    print("  ✓ AUTO mode intelligence")
    print("  ✓ Scene data export for web apps")
    print("  ✓ Automatic fallback handling")
    print()
    print("Next steps:")
    print("  - Open the HTML files in your browser")
    print("  - See docs/UNIFIED_VISUALIZER_GUIDE.md for full documentation")
    print("  - Check visualization/examples/ for web integration")


if __name__ == '__main__':
    main()
