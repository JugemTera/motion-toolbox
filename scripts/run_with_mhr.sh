#!/bin/bash
# Helper script to run Python commands with MHR library paths configured
# Usage: bash scripts/run_with_mhr.sh python your_script.py
#
# NOTE: This requires PyTorch 2.8.0 to match MHR's compiled dependencies.
#       If you get torch conflicts, ensure pyproject.toml pins torch==2.8.0

set -e

# Get the project root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROJECT_ROOT/.venv/lib/mhr_deps"

# Detect OS
OS_TYPE="$(uname -s)"

# Check if MHR libraries have been set up
if [ ! -d "$LIB_DIR" ]; then
    echo "❌ MHR libraries not found at $LIB_DIR"
    echo ""
    echo "Please run: bash scripts/setup_mhr.sh"
    echo ""
    exit 1
fi

# Set the appropriate library path based on OS
case "$OS_TYPE" in
    Darwin*)
        export DYLD_FALLBACK_LIBRARY_PATH="$LIB_DIR:$DYLD_FALLBACK_LIBRARY_PATH"
        ;;
    Linux*)
        export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
        ;;
    *)
        echo "⚠️  Unknown OS: $OS_TYPE - library path may not be set correctly"
        ;;
esac

# Change to project root
cd "$PROJECT_ROOT"

# Execute the command with uv run
uv run "$@"
