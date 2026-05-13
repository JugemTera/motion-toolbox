#!/bin/bash
# Download pre-trained T2M evaluation checkpoints and GloVe data.
#
# These are required for computing semantic evaluation metrics
# (FID, R-Precision, Matching Score, Diversity, MultiModality).
#
# The checkpoints are from Guo et al.'s "Generating Diverse and Natural
# 3D Human Motions from Text" (CVPR 2022) and are used by all standard
# text-to-motion evaluation benchmarks (MoMask, MDM, T2M-GPT, etc.)
#
# Requirements:
#   pip install gdown
#
# Usage:
#   cd /path/to/checkpoints/root
#   bash scripts/download_t2m_evaluators.sh
#
# After running, you should have:
#   ./t2m/text_mot_match/model/finest.tar    (HumanML3D evaluator)
#   ./kit/text_mot_match/model/finest.tar    (KIT-ML evaluator)
#   ./glove/our_vab_data.npy                 (GloVe vectors)
#   ./glove/our_vab_words.pkl                (GloVe vocabulary)
#   ./glove/our_vab_idx.pkl                  (GloVe word indices)

set -e

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "Error: gdown is not installed. Install it with: pip install gdown"
    exit 1
fi

echo "=========================================="
echo "Downloading T2M Evaluation Checkpoints"
echo "=========================================="
echo ""

# ------------------------------------------------------------------
# 1. HumanML3D evaluator (t2m)
# ------------------------------------------------------------------
echo "[1/3] Downloading HumanML3D evaluator (t2m)..."
gdown --fuzzy https://drive.google.com/file/d/1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb/view
rm -rf t2m
unzip t2m.zip
rm t2m.zip
echo "  -> t2m/text_mot_match/model/finest.tar"
echo ""

# ------------------------------------------------------------------
# 2. KIT-ML evaluator
# ------------------------------------------------------------------
echo "[2/3] Downloading KIT-ML evaluator..."
gdown --fuzzy https://drive.google.com/file/d/12liZW5iyvoybXD8eOw4VanTgsMtynCuU/view
rm -rf kit
unzip kit.zip
rm kit.zip
echo "  -> kit/text_mot_match/model/finest.tar"
echo ""

# ------------------------------------------------------------------
# 3. GloVe data (used by the text encoder)
# ------------------------------------------------------------------
echo "[3/3] Downloading GloVe data..."
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing
rm -rf glove
unzip glove.zip
rm glove.zip
echo "  -> glove/our_vab_data.npy"
echo "  -> glove/our_vab_words.pkl"
echo "  -> glove/our_vab_idx.pkl"
echo ""

echo "=========================================="
echo "Download complete!"
echo ""
echo "Directory structure:"
echo "  ./t2m/text_mot_match/model/finest.tar"
echo "  ./kit/text_mot_match/model/finest.tar"
echo "  ./glove/our_vab_data.npy"
echo "  ./glove/our_vab_words.pkl"
echo "  ./glove/our_vab_idx.pkl"
echo ""
echo "Usage in Python:"
echo "  from motion_toolbox.evaluation.semantic import SemanticEvaluator"
echo "  evaluator = SemanticEvaluator("
echo "      dataset_name='humanml',"
echo "      checkpoint_dir='.',"
echo "      glove_dir='./glove',"
echo "  )"
echo "=========================================="
