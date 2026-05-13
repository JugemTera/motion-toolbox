#!/usr/bin/env python3
"""Quick script to create keyframe overlay visualization for any motion file.

Usage:
    uv run python motion_toolbox/demos/quick_keyframe_overlay.py path/to/motion.pt

Or with custom settings:
    uv run python motion_toolbox/demos/quick_keyframe_overlay.py path/to/motion.pt --num-keyframes 15 --opacity 0.3
"""

import argparse
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
    parser = argparse.ArgumentParser(
        description="Create keyframe overlay visualization for a motion file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run python motion_toolbox/demos/quick_keyframe_overlay.py my_motion.pt
  
  # With more keyframes and lower opacity
  uv run python motion_toolbox/demos/quick_keyframe_overlay.py my_motion.pt --num-keyframes 20 --opacity 0.3
  
  # White characters with Phong material
  uv run python motion_toolbox/demos/quick_keyframe_overlay.py my_motion.pt --color white --material phong
        """
    )
    
    parser.add_argument("motion_file", type=str, help="Path to motion file (.pt)")
    parser.add_argument("--output", "-o", type=str, help="Output HTML path (default: outputs/keyframe_overlay.html)")
    parser.add_argument("--num-keyframes", "-k", type=int, default=10, help="Number of keyframes (default: 10)")
    parser.add_argument("--opacity", type=float, default=0.5, help="Keyframe opacity 0-1 (default: 0.5)")
    parser.add_argument("--color", "-c", choices=["blue", "white", "clay", "grey"], default="blue", help="Mesh color (default: blue)")
    parser.add_argument("--material", "-m", choices=["physical", "phong", "lambert"], default="physical", help="Material type (default: physical)")
    parser.add_argument("--body-models", type=str, help="Path to SMPL body models (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    # Check motion file exists
    motion_path = Path(args.motion_file)
    if not motion_path.exists():
        print(f"❌ Motion file not found: {args.motion_file}")
        return 1
    
    # Find body models
    body_model_path = args.body_models or find_body_models()
    if not body_model_path:
        print("\n❌ SMPL body models not found!")
        print("   Download from: https://smpl.is.tue.mpg.de/download.php")
        print("   Install to: ~/.motion_toolbox/datasets/body_models_clean_pkl/")
        print("   Or specify path with --body-models")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = _REPO / "outputs" / "keyframe_overlay"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{motion_path.stem}_overlay.html"
    
    # Create visualization
    print("=" * 70)
    print("KEYFRAME OVERLAY CREATOR")
    print("=" * 70)
    print(f"\n📁 Motion file: {motion_path}")
    print(f"🎨 Color: {args.color}")
    print(f"✨ Material: {args.material}")
    print(f"🔢 Keyframes: {args.num_keyframes}")
    print(f"👁️  Opacity: {args.opacity}")
    print(f"💾 Output: {output_path}\n")
    
    create_keyframe_overlay(
        motion_path=str(motion_path),
        output_html_path=str(output_path),
        body_model_path=body_model_path,
        num_keyframes=args.num_keyframes,
        keyframe_opacity=args.opacity,
        mesh_color=args.color,
        material_type=args.material,
        title=f"Keyframe Overlay: {motion_path.stem}",
    )
    
    print(f"\n✅ Done! Open in browser: {output_path.absolute()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

