#!/usr/bin/env bash
# Optional: fetch large or licensed sample assets into sample_data/
# Edit URLs after you publish mirrors (e.g. Hugging Face dataset for your paper).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "This script is a template. Place public URLs for:"
echo "  - sample_data/gvhmr/*.pt"
echo "  - sample_data/mhr/*_mhr.pt"
echo "  - sample_data/smpl/*.npz (where license permits)"
echo "Root: $ROOT"
echo "See sample_data/*/README.md and THIRD_PARTY_LICENSES.md."
