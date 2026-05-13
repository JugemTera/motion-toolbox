"""Demo: GVHMR to MHR Conversion with Side-by-Side Comparison.

Loads GVHMR motion, converts to MHR using vertex fitting, and displays
both SMPL (original) and MHR (converted) side-by-side.
"""

import sys
from pathlib import Path
import json
import torch
import base64

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Asset locations (created by setup_mhr.sh)
SMPL_MODEL_PATH = Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl"
MHR_ASSETS_DIR = Path.home() / ".motion_toolbox" / "datasets" / "mhr" / "assets"


def main():
    print("=" * 70)
    print("GVHMR → MHR CONVERSION DEMO")
    print("=" * 70)

    from motion_toolbox.converters.motion_converters import (
        GVHMRToMotionConverter,
        MHRToMotionConverter,
    )
    from motion_toolbox.visualization.glb_exporter import GLBExporter
    from motion_toolbox.core.body_models.smpl import SMPLBodyModel
    from motion_toolbox.core.body_models.mhr import MHRBodyModel
    from motion_toolbox.visualization.template_renderer import (
        TemplateRenderer,
        render_ground_plane_code,
    )

    # Check paths
    if not SMPL_MODEL_PATH.exists():
        print(f"❌ SMPL models not found at {SMPL_MODEL_PATH}")
        return 1
    if not MHR_ASSETS_DIR.exists():
        print(f"❌ MHR assets not found at {MHR_ASSETS_DIR}")
        print("   Run: bash scripts/setup_mhr.sh")
        return 1

    print(f"✅ SMPL models: {SMPL_MODEL_PATH}")
    print(f"✅ MHR assets:  {MHR_ASSETS_DIR}")

    # Find GVHMR and corresponding MHR files
    gvhmr_dir = project_root / "sample_data" / "gvhmr"
    mhr_dir = project_root / "sample_data" / "mhr"

    pairs = []
    for gvhmr_file in sorted(gvhmr_dir.glob("gvhmr_*.pt")):
        # Look for matching MHR file
        mhr_file = mhr_dir / f"{gvhmr_file.stem}_mhr.pt"
        if mhr_file.exists():
            pairs.append((gvhmr_file, mhr_file))

    if not pairs:
        print(f"❌ No GVHMR/MHR pairs found")
        return 1

    print(f"\n📂 Found {len(pairs)} GVHMR→MHR pair(s)")

    # Initialize converters and body models
    gvhmr_converter = GVHMRToMotionConverter(body_model_path=str(SMPL_MODEL_PATH))
    mhr_converter = MHRToMotionConverter()

    smpl_body_model = SMPLBodyModel(model_path=str(SMPL_MODEL_PATH))
    mhr_body_model = MHRBodyModel(model_path=str(MHR_ASSETS_DIR.parent), lod=1)

    smpl_exporter = GLBExporter(
        model_path=str(SMPL_MODEL_PATH),
        body_model=smpl_body_model,
        include_floor=False,
    )
    mhr_exporter = GLBExporter(
        model_path=str(MHR_ASSETS_DIR.parent),
        body_model=mhr_body_model,
        include_floor=False,
    )

    # Output
    output_dir = project_root / "outputs" / "gvhmr_mhr_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_data = []

    print(f"\n🎬 Processing...")
    for i, (gvhmr_file, mhr_file) in enumerate(pairs):
        print(f"\n   [{i+1}/{len(pairs)}] {gvhmr_file.stem}")

        # Load and export SMPL (GVHMR)
        gvhmr_data = torch.load(gvhmr_file, weights_only=False)
        smpl_motion = gvhmr_converter.convert(gvhmr_data)
        smpl_glb = output_dir / f"{gvhmr_file.stem}_smpl.glb"
        smpl_exporter.export(smpl_motion, str(smpl_glb))
        print(f"       SMPL: {smpl_motion.num_frames} frames")

        # Load and export MHR
        mhr_data = torch.load(mhr_file, weights_only=False)
        mhr_motion = mhr_converter.convert(mhr_data)
        mhr_glb = output_dir / f"{gvhmr_file.stem}_mhr.glb"
        mhr_exporter.export(mhr_motion, str(mhr_glb))
        print(f"       MHR:  {mhr_motion.num_frames} frames")

        # Add to scene (SMPL left, MHR right for each pair)
        # Embed GLBs as base64 data URLs for local file viewing
        x_offset = i * 5
        
        with open(smpl_glb, "rb") as f:
            smpl_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(mhr_glb, "rb") as f:
            mhr_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        scene_data.append({
            "glb": f"data:application/octet-stream;base64,{smpl_b64}",
            "label": f"GVHMR #{i+1} (SMPL)",
            "position": [x_offset - 1.25, 0, 0],
        })
        scene_data.append({
            "glb": f"data:application/octet-stream;base64,{mhr_b64}",
            "label": f"MHR #{i+1} (Fitted)",
            "position": [x_offset + 1.25, 0, 0],
        })

    # Create HTML viewer
    renderer = TemplateRenderer()
    ground_plane_code = render_ground_plane_code(checkerboard=True)

    context = {
        "TITLE": "GVHMR → MHR Conversion",
        "SCENE_DATA": json.dumps(scene_data, indent=8),
        "MATERIAL_TYPE": "standard",
        "MESH_COLOR_HEX": "0x6495ED",
        "GROUND_PLANE_CODE": ground_plane_code,
        "CAMERA_DISTANCE": 6.0,
    }

    html_content = renderer.render("spatial_grid_viewer.html", context)
    html_path = output_dir / "viewer.html"
    html_path.write_text(html_content)

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"\n🌐 Open: {html_path}")
    print("\nLayout: SMPL (left) | MHR (right) for each animation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
