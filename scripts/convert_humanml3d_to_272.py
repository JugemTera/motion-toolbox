#!/usr/bin/env python3
"""Convert HumanML3D dataset to 272-dim format via AMASS SMPL source data.

Instead of using the lossy 263D->SMPL optimization converter, this script
maps each HumanML3D sample back to its original AMASS source, extracts
SMPL rotations, and produces clean 272D representations.

Outputs:
  - humanml3d_smpl/{id}.npz  -- SMPL params (pose_body, root_orient, trans, betas, gender)
  - humanml3d_272/{id}.npy   -- 272-dim motion features

Data flow:
  AMASS-sourced: AMASS npz -> downsample 20fps -> crop -> SMPL params -> SMPLTo272Converter
  humanact12:    HumanML3D 263D npy -> HumanML3DTo272Converter (direct, no optimization)
  M-prefix:      mirror_smpl_params() on the corresponding non-M sample

Usage:
    python scripts/convert_humanml3d_to_272.py --limit 10
    python scripts/convert_humanml3d_to_272.py --output-dir /data/humanml3d --s3-bucket my-bucket --upload
    python scripts/convert_humanml3d_to_272.py --resume
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import argparse
import csv
import io
import json
import logging
import os
import time
import traceback
import urllib.request
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from motion_toolbox.converters.motion_converters import HumanML3DTo272Converter, SMPLTo272Converter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AMASS_PREFIX = "datasets/AMASS"
HML3D_PREFIX = "datasets/HumanML3D"
BODY_MODEL_PATH = str(Path.home() / ".motion_toolbox/datasets/body_models_clean_pkl")

INDEX_CSV_URL = "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/main/index.csv"
HML3D_JSON_URL = "https://huggingface.co/spaces/Mathux/TMR/resolve/6415e7814d0d3818aa23b3473bb5791f45cf8ab9/amass-annotations/humanml3d.json"

# Dataset-specific frame offsets (at 20 FPS) applied before index.csv cropping.
# From HumanML3D raw_pose_processing.ipynb.
DATASET_OFFSETS_20FPS = {
    "Eyes_Japan_Dataset": 60,  # 3s
    "MPI_HDM05": 60,  # 3s
    "TotalCapture": 20,  # 1s
    "MPI_Limits": 20,  # 1s
    "Transitions_mocap": 10,  # 0.5s
}


# ---------------------------------------------------------------------------
# Mapping / metadata
# ---------------------------------------------------------------------------


def download_mappings(cache_dir: Path) -> tuple[dict, dict]:
    """Download and parse index.csv and humanml3d.json, with local caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_csv_path = cache_dir / "index.csv"
    if not index_csv_path.exists():
        log.info("Downloading index.csv ...")
        urllib.request.urlretrieve(INDEX_CSV_URL, index_csv_path)

    hml3d_json_path = cache_dir / "humanml3d.json"
    if not hml3d_json_path.exists():
        log.info("Downloading humanml3d.json ...")
        urllib.request.urlretrieve(HML3D_JSON_URL, hml3d_json_path)

    # Parse index.csv
    index_map = {}
    with open(index_csv_path) as f:
        for row in csv.DictReader(f):
            name = row.get("new_name")
            if not name:
                continue
            sid = name.replace(".npy", "")
            sf = row.get("start_frame")
            ef = row.get("end_frame")
            if sf is None or ef is None:
                continue
            index_map[sid] = {
                "source_path": row["source_path"],
                "start_frame": int(sf),
                "end_frame": int(ef),
            }

    # Parse humanml3d.json
    with open(hml3d_json_path) as f:
        hml3d_json = json.load(f)

    log.info(f"Loaded {len(index_map)} entries from index.csv, {len(hml3d_json)} from humanml3d.json")
    return index_map, hml3d_json


def build_sample_plan(
    index_map: dict,
    hml3d_json: dict,
    limit: int | None = None,
) -> list[dict]:
    """Build the ordered list of samples to process.

    Returns list of dicts, each with:
      - sid: sample ID (e.g. '000000')
      - amass_path: AMASS relative path (e.g. 'KIT/3/kick_high_left02_poses')
      - source_type: 'amass' | 'humanact12' | 'amass_extra'
      - amass_start: AMASS downsampled frame start
      - amass_end: AMASS downsampled frame end (or None for extras)
    """
    samples = []
    for sid, entry in sorted(hml3d_json.items()):
        amass_path = entry["path"]
        is_humanact12 = "humanact12" in amass_path

        if is_humanact12:
            samples.append(
                {
                    "sid": sid,
                    "amass_path": amass_path,
                    "source_type": "humanact12",
                    "amass_start": None,
                    "amass_end": None,
                }
            )
            continue

        idx = index_map.get(sid)
        if idx is not None:
            dataset = amass_path.split("/")[0]
            offset = DATASET_OFFSETS_20FPS.get(dataset, 0)
            amass_start = offset + idx["start_frame"]
            amass_end = offset + idx["end_frame"]
            samples.append(
                {
                    "sid": sid,
                    "amass_path": amass_path,
                    "source_type": "amass",
                    "amass_start": amass_start,
                    "amass_end": amass_end,
                }
            )
        else:
            # Extra entry (not in index.csv): derive crop from annotation times.
            # Start frame from earliest annotation start; length will be inferred
            # from the 263D npy file at processing time.
            annotations = entry.get("annotations", [])
            if annotations:
                ann_start = min(a["start"] for a in annotations)
                amass_start = round(ann_start * 20)
            else:
                amass_start = 0
            samples.append(
                {
                    "sid": sid,
                    "amass_path": amass_path,
                    "source_type": "amass_extra",
                    "amass_start": amass_start,
                    "amass_end": None,  # to be filled from 263D npy length
                }
            )

    if limit is not None:
        samples = samples[:limit]

    type_counts = defaultdict(int)
    for s in samples:
        type_counts[s["source_type"]] += 1
    log.info(f"Sample plan: {len(samples)} total -- {dict(type_counts)}")
    return samples


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def get_s3_client():
    from dotenv import load_dotenv

    load_dotenv()
    import boto3

    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
    )


def s3_download_npz(s3, bucket: str, key: str) -> dict:
    """Download an npz file from S3 and return its contents as a dict."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = np.load(io.BytesIO(obj["Body"].read()), allow_pickle=True)
    return {k: data[k] for k in data.files}


def s3_download_npy(s3, bucket: str, key: str) -> np.ndarray:
    """Download a single npy file from S3."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return np.load(io.BytesIO(obj["Body"].read()))


def s3_upload_file(s3, bucket: str, local_path: Path, s3_key: str):
    """Upload a local file to S3."""
    s3.upload_file(str(local_path), bucket, s3_key)


# ---------------------------------------------------------------------------
# AMASS processing
# ---------------------------------------------------------------------------

_amass_cache: dict[str, dict] = {}


def load_amass_data(s3, bucket: str, amass_path: str) -> dict | None:
    """Load AMASS npz from S3 with in-memory caching."""
    if amass_path in _amass_cache:
        return _amass_cache[amass_path]

    s3_key = f"{AMASS_PREFIX}/{amass_path}.npz"
    try:
        data = s3_download_npz(s3, bucket, s3_key)
        _amass_cache[amass_path] = data
        return data
    except Exception as e:
        log.warning(f"Failed to download {s3_key}: {e}")
        return None


def evict_amass_cache(keep_paths: set[str] | None = None):
    """Evict AMASS files from cache that are no longer needed."""
    if keep_paths is None:
        _amass_cache.clear()
        return
    to_remove = [k for k in _amass_cache if k not in keep_paths]
    for k in to_remove:
        del _amass_cache[k]


def extract_smpl_clip(amass_data: dict, amass_start: int, amass_end: int) -> dict | None:
    """Extract a downsampled SMPL clip from AMASS data."""
    fps = float(amass_data.get("mocap_framerate", 120))
    down = max(1, int(fps / 20))

    poses = amass_data["poses"][::down]
    trans = amass_data["trans"][::down]

    if amass_end > len(poses):
        log.warning(f"Crop [{amass_start}:{amass_end}] exceeds downsampled length {len(poses)}")
        amass_end = len(poses)
    if amass_start >= len(poses):
        log.warning(f"Start frame {amass_start} >= downsampled length {len(poses)}")
        return None

    clip_poses = poses[amass_start:amass_end]
    clip_trans = trans[amass_start:amass_end]

    if len(clip_poses) < 2:
        return None

    gender_raw = amass_data.get("gender", "neutral")
    if isinstance(gender_raw, np.ndarray):
        gender_raw = gender_raw.item()
    if isinstance(gender_raw, bytes):
        gender = gender_raw.decode()
    else:
        gender = str(gender_raw).lower()
    betas = amass_data.get("betas", np.zeros(10))
    if len(betas) > 10:
        betas = betas[:10]

    return {
        "root_orient": clip_poses[:, :3].astype(np.float32),
        "pose_body": clip_poses[:, 3:66].astype(np.float32),
        "trans": clip_trans.astype(np.float32),
        "betas": betas.astype(np.float32),
        "gender": gender,
    }


# ---------------------------------------------------------------------------
# Converters (lazy-loaded singletons)
# ---------------------------------------------------------------------------

_smpl_to_272: SMPLTo272Converter | None = None
_hml3d_to_272: HumanML3DTo272Converter | None = None


def get_smpl_to_272():
    global _smpl_to_272
    if _smpl_to_272 is None:
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        _smpl_to_272 = SMPLTo272Converter(body_model_path=BODY_MODEL_PATH)
    return _smpl_to_272


def get_hml3d_to_272():
    global _hml3d_to_272
    if _hml3d_to_272 is None:
        from motion_toolbox.converters.motion_converters import HumanML3DTo272Converter

        _hml3d_to_272 = HumanML3DTo272Converter()
    return _hml3d_to_272


# ---------------------------------------------------------------------------
# Processing a single sample
# ---------------------------------------------------------------------------


def process_amass_sample(
    s3,
    bucket: str,
    sample: dict,
    smpl_dir: Path,
    f272_dir: Path,
    resume: bool = False,
) -> dict:
    """Process one AMASS-sourced sample (original + mirror).

    Returns dict with status info.
    """
    from motion_toolbox.utils.mirror_smpl import mirror_smpl_params

    sid = sample["sid"]
    amass_path = sample["amass_path"]
    amass_start = sample["amass_start"]
    amass_end = sample["amass_end"]

    smpl_path = smpl_dir / f"{sid}.npz"
    smpl_m_path = smpl_dir / f"M{sid}.npz"
    f272_path = f272_dir / f"{sid}.npy"
    f272_m_path = f272_dir / f"M{sid}.npy"

    if resume and all(p.exists() for p in [smpl_path, smpl_m_path, f272_path, f272_m_path]):
        return {"sid": sid, "status": "skipped", "reason": "already exists"}

    # For amass_extra samples, we need to get the clip length from 263D npy
    if amass_end is None:
        try:
            npy_263 = s3_download_npy(s3, bucket, f"{HML3D_PREFIX}/new_joint_vecs/{sid}.npy")
            clip_len = npy_263.shape[0] + 1  # 263D has T-1 frames
            amass_end = amass_start + clip_len
        except Exception as e:
            return {"sid": sid, "status": "error", "reason": f"Failed to get 263D npy for length: {e}"}

    # Load AMASS source
    amass_data = load_amass_data(s3, bucket, amass_path)
    if amass_data is None:
        return {"sid": sid, "status": "error", "reason": f"AMASS file not found: {amass_path}"}

    # Extract clip
    clip = extract_smpl_clip(amass_data, amass_start, amass_end)
    if clip is None:
        return {"sid": sid, "status": "error", "reason": "Failed to extract clip"}

    # Save original SMPL
    np.savez_compressed(
        smpl_path,
        pose_body=clip["pose_body"],
        root_orient=clip["root_orient"],
        trans=clip["trans"],
        betas=clip["betas"],
        gender=clip["gender"],
    )

    # Mirror and save
    m_ro, m_pb, m_tr, m_betas = mirror_smpl_params(clip["root_orient"], clip["pose_body"], clip["trans"], clip["betas"])
    np.savez_compressed(
        smpl_m_path,
        pose_body=m_pb,
        root_orient=m_ro,
        trans=m_tr,
        betas=m_betas if m_betas is not None else clip["betas"],
        gender=clip["gender"],
    )

    # Convert to 272
    converter = get_smpl_to_272()
    try:
        data_272 = converter.convert(clip)
        np.save(f272_path, data_272)
    except Exception as e:
        return {"sid": sid, "status": "error", "reason": f"SMPLTo272 failed: {e}"}

    mirror_clip = {
        "pose_body": m_pb,
        "root_orient": m_ro,
        "trans": m_tr,
        "betas": m_betas if m_betas is not None else clip["betas"],
        "gender": clip["gender"],
    }
    try:
        data_272_m = converter.convert(mirror_clip)
        np.save(f272_m_path, data_272_m)
    except Exception as e:
        return {"sid": sid, "status": "error", "reason": f"SMPLTo272 mirror failed: {e}"}

    return {"sid": sid, "status": "ok", "frames": clip["root_orient"].shape[0]}


def process_humanact12_sample(
    s3,
    bucket: str,
    sample: dict,
    f272_dir: Path,
    resume: bool = False,
) -> dict:
    """Process one humanact12 sample (263D -> 272D fallback)."""
    sid = sample["sid"]
    f272_path = f272_dir / f"{sid}.npy"
    f272_m_path = f272_dir / f"M{sid}.npy"

    if resume and f272_path.exists() and f272_m_path.exists():
        return {"sid": sid, "status": "skipped", "reason": "already exists"}

    converter = get_hml3d_to_272()

    for out_path, npy_key in [
        (f272_path, f"{HML3D_PREFIX}/new_joint_vecs/{sid}.npy"),
        (f272_m_path, f"{HML3D_PREFIX}/new_joint_vecs/M{sid}.npy"),
    ]:
        if resume and out_path.exists():
            continue
        try:
            data_263 = s3_download_npy(s3, bucket, npy_key)
        except Exception as e:
            return {"sid": sid, "status": "error", "reason": f"Failed to download {npy_key}: {e}"}
        try:
            data_272 = converter.convert(data_263)
            np.save(out_path, data_272)
        except Exception as e:
            return {"sid": sid, "status": "error", "reason": f"HumanML3DTo272 failed on {npy_key}: {e}"}

    return {"sid": sid, "status": "ok"}


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------


def upload_outputs(s3, bucket: str, output_dir: Path, dry_run: bool = False):
    """Upload humanml3d_smpl/ and humanml3d_272/ to S3."""
    for subdir_name in ["humanml3d_smpl", "humanml3d_272"]:
        local_dir = output_dir / subdir_name
        if not local_dir.exists():
            continue
        files = sorted(local_dir.iterdir())
        log.info(f"Uploading {len(files)} files from {subdir_name}/ to S3 ...")
        for i, fpath in enumerate(files):
            s3_key = f"datasets/{subdir_name}/{fpath.name}"
            if dry_run:
                if i < 3:
                    log.info(f"  [dry-run] Would upload {fpath.name} -> s3://{bucket}/{s3_key}")
            else:
                s3_upload_file(s3, bucket, fpath, s3_key)
                if (i + 1) % 500 == 0:
                    log.info(f"  Uploaded {i + 1}/{len(files)} ...")
        if not dry_run:
            log.info(f"  Uploaded {len(files)} files to s3://{bucket}/datasets/{subdir_name}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert HumanML3D (263D) to 272D via AMASS SMPL source data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/humanml3d_conversion",
        help="Base output directory",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have output files")
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket containing AMASS and HumanML3D mirrors (or set MOTION_TOOLBOX_S3_BUCKET).",
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload results to S3 after processing (uses --s3-bucket)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without processing")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching downloaded mappings (default: <output-dir>/.cache)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    smpl_dir = output_dir / "humanml3d_smpl"
    f272_dir = output_dir / "humanml3d_272"
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / ".cache"

    smpl_dir.mkdir(parents=True, exist_ok=True)
    f272_dir.mkdir(parents=True, exist_ok=True)

    # Download mappings
    index_map, hml3d_json = download_mappings(cache_dir)

    # Build processing plan
    samples = build_sample_plan(index_map, hml3d_json, limit=args.limit)

    if args.dry_run:
        log.info("[dry-run] Would process the above samples. Exiting.")
        return

    bucket = args.s3_bucket or os.environ.get("MOTION_TOOLBOX_S3_BUCKET")
    if not bucket:
        log.error(
            "Set --s3-bucket or MOTION_TOOLBOX_S3_BUCKET to the bucket that mirrors "
            "datasets/AMASS and datasets/HumanML3D (same layout as the public HumanML3D pipeline)."
        )
        sys.exit(1)

    # Initialize S3 client
    s3 = get_s3_client()

    # Process samples
    stats = {"ok": 0, "skipped": 0, "error": 0, "amass_files_loaded": 0}
    errors = []
    t0 = time.time()

    # Group by AMASS path for efficient caching
    from itertools import groupby

    amass_samples = [s for s in samples if s["source_type"] in ("amass", "amass_extra")]
    humanact12_samples = [s for s in samples if s["source_type"] == "humanact12"]

    # Group AMASS samples by path so we download each file once
    amass_samples_sorted = sorted(amass_samples, key=lambda s: s["amass_path"])
    grouped = groupby(amass_samples_sorted, key=lambda s: s["amass_path"])

    total_amass = len(amass_samples)
    total_ha12 = len(humanact12_samples)
    processed = 0

    log.info(f"Processing {total_amass} AMASS-sourced samples ...")
    for amass_path, group_iter in grouped:
        group = list(group_iter)
        for sample in group:
            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                log.info(
                    f"  [{processed}/{total_amass}] {rate:.1f} samples/s "
                    f"| ok={stats['ok']} err={stats['error']} skip={stats['skipped']}"
                )

            try:
                result = process_amass_sample(s3, bucket, sample, smpl_dir, f272_dir, resume=args.resume)
            except Exception:
                result = {"sid": sample["sid"], "status": "error", "reason": traceback.format_exc()}

            stats[result["status"]] += 1
            if result["status"] == "error":
                errors.append(result)
                if len(errors) <= 10:
                    log.warning(f"  Error {sample['sid']}: {result.get('reason', '')[:120]}")

        # Evict this AMASS file from cache (no longer needed)
        _amass_cache.pop(amass_path, None)

    log.info(f"Processing {total_ha12} humanact12 samples (263D->272D fallback) ...")
    for i, sample in enumerate(humanact12_samples):
        if (i + 1) % 100 == 0:
            log.info(f"  [{i + 1}/{total_ha12}]")

        try:
            result = process_humanact12_sample(s3, bucket, sample, f272_dir, resume=args.resume)
        except Exception:
            result = {"sid": sample["sid"], "status": "error", "reason": traceback.format_exc()}

        stats[result["status"]] += 1
        if result["status"] == "error":
            errors.append(result)
            if len(errors) <= 20:
                log.warning(f"  Error {sample['sid']}: {result.get('reason', '')[:120]}")

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"CONVERSION COMPLETE in {elapsed:.1f}s")
    log.info(f"  OK:      {stats['ok']}")
    log.info(f"  Skipped: {stats['skipped']}")
    log.info(f"  Errors:  {stats['error']}")
    log.info(f"  Output:  {output_dir}")

    smpl_count = len(list(smpl_dir.glob("*.npz")))
    f272_count = len(list(f272_dir.glob("*.npy")))
    log.info(f"  SMPL files: {smpl_count}")
    log.info(f"  272D files: {f272_count}")

    if errors:
        err_path = output_dir / "errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2, default=str)
        log.info(f"  Error details: {err_path}")

    # Upload if requested
    if args.upload:
        log.info("Uploading to S3 ...")
        upload_outputs(s3, bucket, output_dir, dry_run=False)
        log.info("Upload complete.")


if __name__ == "__main__":
    main()
