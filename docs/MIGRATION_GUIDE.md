# Migration guide

## `src/` layout (2026)

The installable Python package now lives under `src/motion_toolbox/`. When running scripts **without** installing the package, prepend `src` to `PYTHONPATH`:

```bash
export PYTHONPATH=src
python demos/unified_visualizer_demo.py
```

Or use `uv run` / `pip install -e .` from the repository root.

## Import paths

Public imports remain `from motion_toolbox...` — only the repository layout changed.
