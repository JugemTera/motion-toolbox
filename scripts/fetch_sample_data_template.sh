#!/usr/bin/env bash
# Template: fetch bundled sample motion files from a public Hugging Face dataset.
#
# This script is intentionally a *template* — it does not hit a network until
# you fill in the placeholders below. We keep the wiring here so that, once
# the public mirror is finalised, switching it on is a one-line change.
#
# Usage:
#     # 1) Replace the placeholders (HF_USER / DATASET_NAME / REVISION).
#     # 2) Make sure `huggingface_hub` is installed: `uv pip install huggingface_hub`.
#     # 3) Run:
#     bash scripts/fetch_sample_data_template.sh
#
# What this fetches:
#   - sample_data/gvhmr/*.pt          GVHMR-format SMPL sequences
#   - sample_data/format_272/*.npy    272-D MotionMillion-style features
#   - sample_data/format_205/*.npy    205-D STMC features
#   - sample_data/mhr/*_mhr.pt        MHR-format sequences (optional)
#
# Update the dataset / paths below to match your published mirror.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/sample_data"

# ---------------------------------------------------------------------------
# CONFIGURE ME
# ---------------------------------------------------------------------------
# Hugging Face dataset coordinates. Until the public mirror is set up, these
# are placeholders; treat them as TODOs.
HF_USER="${HF_USER:-<HF_USER>}"                       # e.g. "your-org"
DATASET_NAME="${DATASET_NAME:-<DATASET_NAME>}"        # e.g. "motion-toolbox-samples"
REVISION="${REVISION:-main}"                          # branch / tag / commit
# Optional subdirectory inside the dataset repo; leave blank to copy everything.
HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-}"
# ---------------------------------------------------------------------------

if [[ "${HF_USER}" == "<HF_USER>" || "${DATASET_NAME}" == "<DATASET_NAME>" ]]; then
    cat <<'EOF'
This is the placeholder template. Set HF_USER and DATASET_NAME first, e.g.:

    HF_USER=your-org DATASET_NAME=motion-toolbox-samples \
        bash scripts/fetch_sample_data_template.sh

Or edit the CONFIGURE ME block in this script.
EOF
    exit 1
fi

REPO_ID="${HF_USER}/${DATASET_NAME}"

if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Installing huggingface_hub into the current environment..."
    python3 -m pip install --quiet huggingface_hub
fi

mkdir -p "${DEST}"

echo "Downloading dataset ${REPO_ID}@${REVISION} -> ${DEST}"
python3 - "$REPO_ID" "$REVISION" "$DEST" "$HF_PATH_IN_REPO" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id, revision, dest, subdir = sys.argv[1:5]
allow_patterns = None
if subdir:
    allow_patterns = [f"{subdir.rstrip('/')}/*"]

local = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    repo_type="dataset",
    local_dir=dest,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
)
print(f"Snapshot synced into: {Path(local).resolve()}")
PY

echo "Done."
echo "See sample_data/*/README.md and THIRD_PARTY_LICENSES.md for license notes."
