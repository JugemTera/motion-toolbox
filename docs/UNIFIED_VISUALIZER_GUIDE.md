# Unified Visualizer Guide

Complete guide for the new unified visualization system that works across all motion formats and body models.

## Quick Start

```python
from motion_toolbox.visualization import Visualizer
import numpy as np

# Create visualizer
vis = Visualizer()

# Visualize any motion format - it just works!
motion_data = np.load("motion.npy")  # Any format: 263D, 272D, 205D, joints, Motion
vis.visualize(motion_data, "output.html")

# Open output.html in browser - done!
```

## Overview

The Unified Visualizer provides a single, consistent API for visualizing motion data regardless of:
- **Input format**: HumanML3D 263D, 272D, 205D (STMC), Motion objects, joint positions
- **Visualization mode**: Skeleton or mesh
- **Body model**: SMPL, MHR

### Key Features

- **Automatic format detection**: No need to specify format manually
- **Intelligent compatibility checking**: Warns about incompatible combinations
- **Fallback behavior**: Automatically uses skeleton when mesh isn't available
- **Multi-body model support**: SMPL, MHR in one API
- **Frontend integration**: Export JSON for web apps

## Installation

```bash
# From a clone:
pip install -e .
pip install -e ".[viz]"    # mesh / video-related extras
pip install -e ".[ml]"     # optional: full torch stack
```

## Basic Usage

### Visualize Any Format

```python
from motion_toolbox.visualization import Visualizer

vis = Visualizer()

# HumanML3D 263D
data_263d = np.random.randn(120, 263)
vis.visualize(data_263d, "263d_skeleton.html")

# 272D format
data_272d = np.random.randn(120, 272)
vis.visualize(data_272d, "272d_skeleton.html")

# Motion object
from motion_toolbox.core import Motion, Skeleton
motion = Motion(rotations, translation, Skeleton.from_humanml3d())
vis.visualize(motion, "motion.html")

# Joint positions
positions = np.random.randn(120, 22, 3)
vis.visualize(positions, "joints.html")
```

### Choose Visualization Mode

```python
# Skeleton visualization (fast, works for all formats)
vis.visualize(data, "skeleton.html", mode='skeleton')

# Mesh visualization (requires compatible format)
vis.visualize(motion, "mesh.html", mode='mesh', body_model='smpl')

# Auto mode (chooses best option)
vis.visualize(data, "auto.html", mode='auto')
```

### Choose Body Model

```python
# SMPL mesh (24 joints, widely used)
vis.visualize(motion, "smpl.html", mode='mesh', body_model='smpl')

# MHR mesh (53 joints, high-fidelity)
# Note: MHR requires special setup
vis.visualize(mhr_motion, "mhr.html", mode='mesh', body_model='mhr')
```

## Advanced Configuration

### Custom Settings

```python
vis = Visualizer(
    body_model_path="/path/to/smpl/models",
    default_mode='skeleton',  # 'skeleton', 'mesh', 'auto'
    default_body_model='smpl',  # 'smpl', 'mhr'
    on_incompatible='auto_skeleton',  # 'error' or 'auto_skeleton'
)
```

### Visualization Options

```python
vis.visualize(
    data,
    "output.html",
    mode='skeleton',
    fps=30,  # Frames per second
    title="My Custom Animation",
    # Additional options passed to backend
)
```

### Handle Incompatible Combinations

```python
# Automatic fallback to skeleton
vis = Visualizer(on_incompatible='auto_skeleton')
vis.visualize(joint_positions, "output.html", mode='mesh')
# → Falls back to skeleton (joints lack rotation info)

# Strict mode - raise error
vis = Visualizer(on_incompatible='error')
try:
    vis.visualize(joint_positions, "output.html", mode='mesh')
except IncompatibleFormatError as e:
    print(f"Error: {e}")
    # Provides clear message with suggestions
```

## Format Compatibility

### Skeleton Visualization
Works with ALL formats:
- ✅ 263D (HumanML3D)
- ✅ 272D
- ✅ 205D (STMC)
- ✅ Motion objects
- ✅ Joint positions
- ✅ SMPL parameters

### Mesh Visualization

| Format | SMPL | MHR |
|--------|------|-----|
| 263D (HumanML3D) | ⚠️ Requires SMPLify* | ❌ |
| 272D | ✅ | ✅ |
| 205D (STMC) | ✅ | ✅ |
| 232D (ALIGN) | ✅ | ⚠️ |
| Motion (SMPL) | ✅ | ⚠️ |
| Motion (MHR) | ⚠️ | ✅ |
| Joint positions | ❌ Needs rotations | ❌ |

*SMPLify optimization is not run automatically (it's slow). Use skeleton visualization or convert to 272D first.

## Frontend Integration

### Export Scene Data for Web Apps

```python
vis = Visualizer()

# Export as JSON-serializable dict
scene_data = vis.to_scene_data(motion_data, mode='skeleton')

# Use in Flask/FastAPI/Django
import json
json_str = json.dumps(scene_data)

# scene_data contains:
# - positions: joint positions array
# - kinematic_chains: bone connections
# - chain_colors: colors
# - fps, num_frames, etc.
```

### Example: Flask Integration

```python
from flask import Flask, jsonify
from motion_toolbox.visualization import Visualizer

app = Flask(__name__)

@app.route('/api/motion/<motion_id>')
def get_motion(motion_id):
    motion_data = load_motion(motion_id)
    vis = Visualizer()
    scene_data = vis.to_scene_data(motion_data, mode='skeleton')
    return jsonify(scene_data)
```

See [`visualization/examples/README_embedding.md`](../src/motion_toolbox/visualization/examples/README_embedding.md) for complete examples.

## Body Model Compatibility Matrix

Detailed information about which formats work with which body models and what conversions are required.

See [`compatibility.py`](../src/motion_toolbox/visualization/compatibility.py) for the full compatibility matrix implementation.

## Error Messages

The visualizer provides clear, actionable error messages:

```python
# Example error for incompatible format
IncompatibleFormatError: Format 'joint_positions' is not compatible with mode='mesh', body_model='smpl'

Warnings:
  - Joint positions lack rotation information needed for mesh animation

Suggestions:
  - Use skeleton visualization instead (mode='skeleton')
  - Provide full motion data (263D, 272D, or Motion object)
```

## Performance Tips

1. **Skeleton vs Mesh**: Skeleton is ~10x faster
2. **Caching**: Cache generated HTML/GLB files
3. **Downsampling**: Use `motion.downsample(2)` for preview
4. **Format**: 272D and Motion objects are fastest to visualize

## Troubleshooting

### "Format not detected"
- Check data shape: should be (T, 263), (T, 272), (T, 205), or (T, N, 3)
- Use `format` parameter to override: `vis.visualize(data, path, format='263d')`

### "SMPLify required"
- HumanML3D 263D requires SMPLify for mesh (slow)
- Solutions:
  1. Use skeleton mode: `mode='skeleton'`
  2. Convert to 272D first
  3. Use HumanML3DToSMPLConverter directly

### "Body model path not found"
- Set path explicitly: `Visualizer(body_model_path="/path/to/models")`
- Default: `~/.motion_toolbox/datasets/body_models_clean_pkl`

## API Reference

### Visualizer Class

```python
class Visualizer:
    def __init__(
        self,
        body_model_path: Optional[str] = None,
        default_mode: str = 'auto',
        default_body_model: str = 'auto',
        on_incompatible: str = 'auto_skeleton',
    )
    
    def visualize(
        self,
        data: Any,
        output_path: Union[str, Path],
        format: Optional[str] = None,
        mode: Optional[str] = None,
        body_model: Optional[str] = None,
        on_incompatible: Optional[str] = None,
        fps: int = 20,
        title: Optional[str] = None,
        **kwargs,
    ) -> Path
    
    def to_scene_data(
        self,
        data: Any,
        format: Optional[str] = None,
        mode: Optional[str] = None,
        body_model: Optional[str] = None,
        **kwargs,
    ) -> dict
```

### Convenience Functions

```python
# Quick visualization
from motion_toolbox.visualization import visualize
visualize(data, "output.html", mode='skeleton')

# Quick skeleton visualization
from motion_toolbox.visualization import visualize_skeleton
visualize_skeleton(positions, "skeleton.html")

# Quick mesh visualization
from motion_toolbox.visualization import visualize_mesh
visualize_mesh(motion, "mesh.html", body_model='smpl')

# Quick scene data export
from motion_toolbox.visualization import export_scene_data
scene = export_scene_data(data, mode='skeleton')
```

## Examples

See [`visualization/examples/`](../src/motion_toolbox/visualization/examples/) for:
- Flask embedding example
- Web framework integration patterns
- Custom Three.js visualization
- User study interfaces

## Next Steps

- Explore [embedding guide](../src/motion_toolbox/visualization/examples/README_embedding.md)
- See [migration guide](MIGRATION_GUIDE.md) for upgrading from old API
- Check [visualization guide](VISUALIZATION_GUIDE.md) for existing features
