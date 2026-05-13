#!/usr/bin/env python3
"""Verify SMPL parameter mirroring by visual and numerical comparison.

Either load a local AMASS .npz file, or download from S3 using --s3-bucket and --s3-key.

Usage:
    python scripts/verify_smpl_mirror.py --local-npz /path/to/amass_sample.npz
    python scripts/verify_smpl_mirror.py --s3-bucket my-bucket --s3-key datasets/AMASS/KIT/3/kick_high_left02_poses.npz
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np

DEFAULT_S3_KEY = "datasets/AMASS/KIT/3/kick_high_left02_poses.npz"
BODY_MODEL_PATH = str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl")


def load_amass_from_local(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_amass_from_s3(bucket: str, s3_key: str) -> dict:
    from dotenv import load_dotenv

    load_dotenv()
    import os

    import boto3

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
    )

    print(f"Downloading s3://{bucket}/{s3_key} ...")
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    data = np.load(io.BytesIO(obj["Body"].read()), allow_pickle=True)
    result = {k: data[k] for k in data.files}
    print(f"  poses: {result['poses'].shape}, fps: {result['mocap_framerate']}, gender: {result['gender']}")
    return result


def extract_smpl_params(amass_data: dict, max_frames: int = 200) -> dict:
    """Extract SMPL params from AMASS npz, downsampled to 20 FPS."""
    fps = float(amass_data["mocap_framerate"])
    down = max(1, int(fps / 20))

    poses = amass_data["poses"][::down]
    trans = amass_data["trans"][::down]

    if len(poses) > max_frames:
        poses = poses[:max_frames]
        trans = trans[:max_frames]

    root_orient = poses[:, :3].astype(np.float32)
    pose_body = poses[:, 3:66].astype(np.float32)
    trans = trans.astype(np.float32)
    betas = amass_data["betas"][:10].astype(np.float32)
    gender = str(amass_data["gender"])

    return {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "trans": trans,
        "betas": betas,
        "gender": gender,
    }


def create_motion_from_smpl(root_orient, pose_body, trans, betas, gender="neutral"):
    from motion_toolbox.converters.motion_converters import SMPLToMotionConverter

    converter = SMPLToMotionConverter(body_model_path=BODY_MODEL_PATH)
    smpl_dict = {
        "pose_body": pose_body,
        "root_orient": root_orient,
        "trans": trans,
        "betas": betas,
        "gender": gender,
    }
    return converter.convert(smpl_dict)


def generate_side_by_side_html(glb_left: str, glb_right: str, title: str, num_frames: int) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title></head>
<body><p>Side-by-side viewer placeholder for {glb_left} vs {glb_right} ({num_frames} frames).</p></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Verify SMPL mirroring")
    parser.add_argument("--local-npz", type=Path, default=None, help="Local AMASS .npz file")
    parser.add_argument("--s3-bucket", type=str, default=None, help="S3 bucket (or MOTION_TOOLBOX_S3_BUCKET)")
    parser.add_argument("--s3-key", type=str, default=DEFAULT_S3_KEY, help="S3 object key for AMASS npz")
    parser.add_argument("--output-dir", default="sample_data/mirror_verification", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to process")
    args = parser.parse_args()

    bucket = args.s3_bucket or __import__("os").environ.get("MOTION_TOOLBOX_S3_BUCKET")

    if args.local_npz:
        amass_data = load_amass_from_local(args.local_npz.expanduser().resolve())
        sample_name = args.local_npz.stem
    elif bucket:
        amass_data = load_amass_from_s3(bucket, args.s3_key)
        sample_name = Path(args.s3_key).stem
    else:
        print(
            "Provide --local-npz /path/to/file.npz or set --s3-bucket / MOTION_TOOLBOX_S3_BUCKET for S3 download.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = extract_smpl_params(amass_data, max_frames=args.max_frames)
    print(f"\nExtracted SMPL params: {params['root_orient'].shape[0]} frames, gender={params['gender']}")

    from motion_toolbox.utils.mirror_smpl import mirror_smpl_params, validate_mirror

    m_ro, m_pb, m_tr, m_betas = mirror_smpl_params(
        params["root_orient"], params["pose_body"], params["trans"], params["betas"]
    )
    print("Mirrored SMPL params")

    print("\nRunning numerical validation (FK(mirror) vs mirror(FK))...")
    result = validate_mirror(
        params["root_orient"],
        params["pose_body"],
        params["trans"],
        params["betas"],
        gender=params["gender"],
        body_model_path=BODY_MODEL_PATH,
    )
    print(f"  Max error:  {result['max_error']:.6e}")
    print(f"  Mean error: {result['mean_error']:.6e}")
    print(f"  Passed:     {result['passed']}")

    if not result["passed"]:
        print("\n  WARNING: Validation failed! Max error exceeds threshold.")

    print("\nCreating Motion objects...")
    motion_orig = create_motion_from_smpl(
        params["root_orient"],
        params["pose_body"],
        params["trans"],
        params["betas"],
        params["gender"],
    )
    motion_mirror = create_motion_from_smpl(
        m_ro,
        m_pb,
        m_tr,
        m_betas if m_betas is not None else params["betas"],
        params["gender"],
    )

    from motion_toolbox.visualization.mesh_visualizer import MeshVisualizer

    vis = MeshVisualizer(body_model="smpl", body_model_path=BODY_MODEL_PATH, fps=20)

    glb_orig = output_dir / f"{sample_name}_original.glb"
    glb_mirror = output_dir / f"{sample_name}_mirrored.glb"

    print(f"Exporting original GLB: {glb_orig}")
    vis.visualize_motion(motion_orig, str(glb_orig), generate_html=False)

    print(f"Exporting mirrored GLB: {glb_mirror}")
    vis.visualize_motion(motion_mirror, str(glb_mirror), generate_html=False)

    html_path = output_dir / f"{sample_name}_mirror_comparison.html"
    html = generate_side_by_side_html(
        glb_orig.name,
        glb_mirror.name,
        title=sample_name,
        num_frames=params["root_orient"].shape[0],
    )
    html_path.write_text(html)
    print(f"\nSide-by-side visualization: {html_path}")
    print("Done!")


if __name__ == "__main__":
    main()
