# MHR (Momentum Human Rig) Setup Guide

This guide explains how to set up MHR support in the Motion Toolbox.

## Prerequisites

1. **Pixi** - conda-based environment manager (for building MHR dependencies)
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   source ~/.zshrc  # or restart your terminal
   ```

2. **uv** - Python package manager (for running the toolbox)

## Quick Setup

Run the single setup script:

```bash
bash scripts/setup_mhr.sh
```

This script automatically:
1. Clones MHR repository to `/tmp/MHR`
2. Downloads and extracts MHR assets
3. Installs MHR with Pixi (builds `pymomentum`)
4. Copies required shared libraries to the `uv` venv
5. Copies assets to `~/.motion_toolbox/datasets/mhr/`

## Running MHR Demos

After setup, use the helper script:

```bash
bash scripts/run_with_mhr.sh python demos/mhr_visualization_demo.py
bash scripts/run_with_mhr.sh python demos/gvhmr_to_mhr_demo.py
```

## What the Scripts Do

### `scripts/setup_mhr.sh`

Complete setup script that:
- Detects OS (macOS/Linux) and uses appropriate paths
- Clones MHR repo to `/tmp` (or `/private/tmp` on macOS)
- Downloads assets from GitHub release
- Runs `pixi install` to build `pymomentum`
- Copies shared libraries (`.dylib` on macOS, `.so` on Linux)
- Copies assets for runtime use

### `scripts/run_with_mhr.sh`

Runtime wrapper that:
- Sets library path (`DYLD_FALLBACK_LIBRARY_PATH` on macOS, `LD_LIBRARY_PATH` on Linux)
- Runs the command with `uv run`

## Troubleshooting

### "Pixi MHR environment not found"

Re-run the setup script:
```bash
bash scripts/setup_mhr.sh
```

### "Library not loaded" errors

1. Check that libraries were copied:
   ```bash
   ls .venv/lib/mhr_deps/ | wc -l
   ```
   Should show ~690 files.

2. Make sure you're using `bash scripts/run_with_mhr.sh` to run your code.

### Missing `pixi` command

Install Pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## File Structure

After setup:

```
motion_toolbox/
├── .venv/lib/mhr_deps/     # Shared libraries from conda-forge
├── scripts/
│   ├── setup_mhr.sh        # Complete MHR setup
│   └── run_with_mhr.sh     # Run with library paths
└── ...

~/.motion_toolbox/datasets/mhr/
└── assets/                  # MHR model assets
```

## Why This Approach?

MHR depends on `pymomentum`, which requires conda-forge libraries. We use Pixi to build these once, then copy them to the `uv` environment for fast daily development.
