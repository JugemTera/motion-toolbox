#!/bin/bash
# Complete MHR (Momentum Human Rig) setup script
# This script:
#   1. Installs Pixi if not present
#   2. Clones MHR repository to /tmp
#   3. Downloads and extracts MHR assets
#   4. Installs MHR with Pixi
#   5. Copies required libraries to the uv venv
#   6. Downloads assets to ~/.motion_toolbox for runtime use

set -e

echo "======================================================================"
echo "MHR (Momentum Human Rig) Complete Setup"
echo "======================================================================"
echo ""

# Detect OS
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
    Darwin*)
        TMP_DIR="/private/tmp"
        LIB_EXT="dylib"
        ;;
    Linux*)
        TMP_DIR="/tmp"
        LIB_EXT="so"
        ;;
    *)
        echo "❌ Unsupported OS: $OS_TYPE"
        exit 1
        ;;
esac

# Detect shell config file
if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "/bin/zsh" ] || [ "$SHELL" = "/usr/bin/zsh" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
else
    SHELL_CONFIG="$HOME/.bashrc"
fi

# Get the project root (directory containing pyproject.toml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Validate we're in the right directory
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "❌ Could not find pyproject.toml in $PROJECT_ROOT"
    echo ""
    echo "Please run this script from the project directory:"
    echo "  cd /path/to/motion_toolbox"
    echo "  bash scripts/setup_mhr.sh"
    exit 1
fi

MHR_REPO_DIR="$TMP_DIR/MHR"
PIXI_ENV_DIR="$MHR_REPO_DIR/.pixi/envs/default"
UV_VENV_PATH="$PROJECT_ROOT/.venv"
LIB_DIR="$UV_VENV_PATH/lib/mhr_deps"
ASSETS_URL="https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"
LOCAL_ASSETS_DIR="$HOME/.motion_toolbox/datasets/mhr"

echo "OS: $OS_TYPE"
echo "Shell config: $SHELL_CONFIG"
echo "Project root: $PROJECT_ROOT"
echo "MHR repo: $MHR_REPO_DIR"
echo "Assets destination: $LOCAL_ASSETS_DIR"
echo ""

# Check for required tools (except pixi which we'll install)
for cmd in git curl unzip uv; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ Required tool not found: $cmd"
        if [ "$cmd" = "uv" ]; then
            echo "   Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        fi
        exit 1
    fi
done
echo "✅ All required tools found"
echo ""

# Step 0a: Create uv virtual environment if needed
echo "----------------------------------------------------------------------"
echo "Step 0a: Create Python virtual environment"
echo "----------------------------------------------------------------------"
cd "$PROJECT_ROOT"

# Unset VIRTUAL_ENV to avoid conflicts with other environments
unset VIRTUAL_ENV

# Check if venv exists and is valid
if [ ! -f "$UV_VENV_PATH/bin/python" ] && [ ! -f "$UV_VENV_PATH/Scripts/python.exe" ]; then
    # Remove invalid .venv if it exists
    if [ -d "$UV_VENV_PATH" ]; then
        echo "   Removing invalid .venv directory..."
        rm -rf "$UV_VENV_PATH"
    fi
    echo "   Creating virtual environment with uv..."
    uv venv
    uv sync
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment already exists"
fi
echo ""

# Step 0b: Install Pixi if not present
echo "----------------------------------------------------------------------"
echo "Step 0b: Check/Install Pixi"
echo "----------------------------------------------------------------------"
if ! command -v pixi &> /dev/null; then
    echo "   Pixi not found. Installing..."
    curl -fsSL https://pixi.sh/install.sh | bash
    
    # Source the shell config to get pixi in PATH
    export PATH="$HOME/.pixi/bin:$PATH"
    
    if ! command -v pixi &> /dev/null; then
        echo "   ❌ Pixi installation failed or not in PATH"
        echo "   Please restart your terminal and run this script again"
        exit 1
    fi
    echo "   ✅ Pixi installed successfully"
else
    echo "   ✅ Pixi already installed"
fi
echo ""

# Step 1: Clone MHR repository
echo "----------------------------------------------------------------------"
echo "Step 1: Clone MHR repository"
echo "----------------------------------------------------------------------"
if [ -d "$MHR_REPO_DIR" ]; then
    echo "   MHR repo already exists at $MHR_REPO_DIR"
    echo "   Updating..."
    cd "$MHR_REPO_DIR"
    git pull --quiet || true
else
    echo "   Cloning to $MHR_REPO_DIR..."
    git clone --quiet https://github.com/facebookresearch/MHR.git "$MHR_REPO_DIR"
fi
echo "   ✅ Done"
echo ""

# Step 2: Download and extract assets
echo "----------------------------------------------------------------------"
echo "Step 2: Download MHR assets"
echo "----------------------------------------------------------------------"
cd "$MHR_REPO_DIR"
if [ -d "$MHR_REPO_DIR/assets" ]; then
    echo "   Assets already exist in repo"
else
    echo "   Downloading assets.zip..."
    curl -sL -o assets.zip "$ASSETS_URL"
    echo "   Extracting..."
    unzip -q assets.zip
    rm assets.zip
fi
echo "   ✅ Done"
echo ""

# Step 3: Install with Pixi
echo "----------------------------------------------------------------------"
echo "Step 3: Install MHR with Pixi"
echo "----------------------------------------------------------------------"
cd "$MHR_REPO_DIR"
if [ -d "$PIXI_ENV_DIR" ]; then
    echo "   Pixi environment already exists"
else
    echo "   Running pixi install (this may take a few minutes)..."
    pixi install
fi
echo "   ✅ Done"
echo ""

# Step 4: Copy libraries to uv venv
echo "----------------------------------------------------------------------"
echo "Step 4: Copy libraries to uv venv"
echo "----------------------------------------------------------------------"
mkdir -p "$LIB_DIR"
cd "$PIXI_ENV_DIR/lib"

if [ "$OS_TYPE" = "Darwin" ]; then
    COPIED=$(cp -f lib*.$LIB_EXT "$LIB_DIR/" 2>/dev/null && ls "$LIB_DIR"/lib*.$LIB_EXT 2>/dev/null | wc -l | tr -d ' ')
else
    COPIED=$(cp -a lib*.$LIB_EXT* "$LIB_DIR/" 2>/dev/null && ls "$LIB_DIR"/lib*.$LIB_EXT* 2>/dev/null | wc -l | tr -d ' ')
fi
echo "   Copied $COPIED library files to $LIB_DIR"
echo "   ✅ Done"
echo ""

# Step 5: Copy Python packages from Pixi environment
echo "----------------------------------------------------------------------"
echo "Step 5: Copy Python packages (mhr, pymomentum) from Pixi"
echo "----------------------------------------------------------------------"
cd "$PROJECT_ROOT"

# Find the site-packages in both environments
PIXI_SITE_PACKAGES="$PIXI_ENV_DIR/lib/python3.12/site-packages"

# Get UV's actual site-packages path from Python itself
UV_SITE_PACKAGES=$(cd "$PROJECT_ROOT" && uv run python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)

# Fallback if that didn't work
if [ -z "$UV_SITE_PACKAGES" ] || [ ! -d "$UV_SITE_PACKAGES" ]; then
    UV_SITE_PACKAGES=$(find "$UV_VENV_PATH/lib" -maxdepth 1 -name "python3.*" -type d | head -1)/site-packages
fi

echo "   Pixi packages: $PIXI_SITE_PACKAGES"
echo "   UV packages: $UV_SITE_PACKAGES"

# Copy mhr package from MHR repo root (not in site-packages!)
if [ -d "$MHR_REPO_DIR/mhr" ]; then
    echo "   Copying mhr from repo..."
    rm -rf "$UV_SITE_PACKAGES/mhr"
    cp -r "$MHR_REPO_DIR/mhr" "$UV_SITE_PACKAGES/"
else
    echo "   ⚠️  Package mhr not found at $MHR_REPO_DIR/mhr"
fi

# Copy pymomentum from Pixi environment
if [ -d "$PIXI_SITE_PACKAGES/pymomentum" ]; then
    echo "   Copying pymomentum..."
    rm -rf "$UV_SITE_PACKAGES/pymomentum"
    cp -r "$PIXI_SITE_PACKAGES/pymomentum" "$UV_SITE_PACKAGES/"
else
    echo "   ⚠️  Package pymomentum not found in Pixi environment"
fi

# Copy any .so files for pymomentum
for so_file in "$PIXI_SITE_PACKAGES"/pymomentum*.so; do
    if [ -f "$so_file" ]; then
        echo "   Copying $(basename "$so_file")..."
        cp "$so_file" "$UV_SITE_PACKAGES/"
    fi
done

echo "   ✅ Done"
echo ""

# Step 6: Copy assets to local directory for runtime use
echo "----------------------------------------------------------------------"
echo "Step 6: Copy assets for runtime use"
echo "----------------------------------------------------------------------"
mkdir -p "$LOCAL_ASSETS_DIR"
if [ -d "$LOCAL_ASSETS_DIR/assets" ]; then
    echo "   Assets already exist at $LOCAL_ASSETS_DIR/assets"
else
    echo "   Copying assets to $LOCAL_ASSETS_DIR..."
    cp -r "$MHR_REPO_DIR/assets" "$LOCAL_ASSETS_DIR/"
fi
echo "   ✅ Done"
echo ""

# Done
echo "======================================================================"
echo "✅ MHR SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "To run MHR demos:"
echo "  bash scripts/run_with_mhr.sh python demos/mhr_visualization_demo.py"
echo "  bash scripts/run_with_mhr.sh python demos/gvhmr_to_mhr_demo.py"
echo ""
