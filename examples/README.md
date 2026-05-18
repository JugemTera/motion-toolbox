# Unified Visualizer Examples

This directory contains examples for testing and using the unified visualization system.

## Quick Start

### Run the Visualizer Test Suite

Runs the unit tests in ``tests/test_unified_visualizer.py`` and writes a
self-contained HTML summary report:

```bash
# From the repository root
PYTHONPATH=src python3 examples/test_visualizer_simple.py
```

**Output:** Opens a single `test_report.html` in your browser with:
- 📊 Interactive dashboard with all test results
- 🎨 Visual cards for each test case
- 🔗 Direct links to each visualization
- 📈 Statistics and capabilities overview
- ✅ Pass/fail status for each test

**What it tests:**
- ✅ Automatic format detection (263D, 272D, 205D, joints, Motion)
- ✅ Compatibility matrix validation
- ✅ Skeleton visualization (all formats)
- ✅ Mesh visualization (SMPL, MHR)
- ✅ AUTO mode intelligence
- ✅ Fallback behavior
- ✅ Scene data export for frontends
- ✅ Custom options (FPS, title, etc.)

**After running:**
```bash
# The script will print:
🎉 Open the report in your browser:
   file:///path/to/outputs/unified_visualizer_test/test_report.html

# Or open directly:
open outputs/unified_visualizer_test/test_report.html  # macOS
xdg-open outputs/unified_visualizer_test/test_report.html  # Linux
```

## Available Examples

### 1. Visualizer Test Suite
**File:** `test_visualizer_simple.py`
**Purpose:** Runs the visualizer unit tests and emits an HTML report
**Recommended for:** Initial validation after install / changes

**Tests covered:**
```
TEST 1: Automatic Format Detection (7 formats)
TEST 2: Compatibility Checking (9 combinations)
TEST 3: Skeleton Visualizations (7 formats)
TEST 4: Mesh Visualizations (SMPL)
TEST 5: AUTO Mode Intelligence
TEST 6: Fallback Behavior
TEST 7: Scene Data Export (JSON)
TEST 8: Custom Options
```

**Sample output:**
```
Generated Files:
   HTML files: 15+
   GLB files: 3+
   Total size: ~500 KB

Capabilities Tested:
   ✓ Automatic format detection (7 formats)
   ✓ Compatibility checking (matrix validation)
   ✓ Skeleton visualization (all formats)
   ...
```

## Usage Patterns

### Basic Usage
```python
from motion_toolbox.visualization import Visualizer
import numpy as np

# Create visualizer
vis = Visualizer()

# Visualize any format
data = np.random.randn(100, 263)  # HumanML3D
vis.visualize(data, "output.html")
```

### All Supported Formats

```python
# 1. HumanML3D 263D
data_263d = np.random.randn(100, 263)
vis.visualize(data_263d, "263d.html")

# 2. 272D Format
data_272d = np.random.randn(100, 272)
vis.visualize(data_272d, "272d.html")

# 3. STMC 205D
data_205d = np.random.randn(100, 205)
vis.visualize(data_205d, "205d.html")

# 4. Joint Positions
positions = np.random.randn(100, 22, 3)
vis.visualize(positions, "joints.html")

# 5. Motion Object
from motion_toolbox.core import Motion, Skeleton
skeleton = Skeleton.from_humanml3d()
motion = Motion(rotations, translation, skeleton)
vis.visualize(motion, "motion.html")
```

### Visualization Modes

```python
# Skeleton mode (fast, works with all formats)
vis.visualize(data, "skeleton.html", mode='skeleton')

# Mesh mode (requires compatible format)
vis.visualize(motion, "mesh.html", mode='mesh', body_model='smpl')

# AUTO mode (intelligent default)
vis.visualize(data, "auto.html", mode='auto')
```

### Body Models

```python
# SMPL (24 joints, academic datasets)
vis.visualize(motion, "smpl.html", mode='mesh', body_model='smpl')

# MHR (53+ joints, high-fidelity)
vis.visualize(mhr_motion, "mhr.html", mode='mesh', body_model='mhr')
```

## SMPL Mesh Visualization

The unified visualizer supports SMPL mesh visualization for compatible formats. This creates fully animated 3D mesh visualizations viewable in your browser.

### Compatible Formats

The visualizer can create SMPL mesh visualizations from these formats:

- **272D format** (simple conversion)
- **205D/STMC format** (simple conversion)  
- **Motion objects** with SMPL skeleton (direct)

### Usage

```python
from motion_toolbox.visualization import Visualizer

vis = Visualizer()

# Visualize 272D as SMPL mesh
data_272d = np.random.randn(100, 272)
vis.visualize(
    data_272d, 
    "output.html",
    mode='mesh',
    body_model='smpl',
    body_model_path='/path/to/smpl/models'  # Optional, uses default if omitted
)

# Visualize Motion object as SMPL mesh
from motion_toolbox.core import Motion, Skeleton

skeleton = Skeleton.from_smpl()
rotations = np.random.randn(100, 24, 3) * 0.1
translation = np.zeros((100, 3))
motion = Motion(rotations, translation, skeleton, floor_alignment='joint')

vis.visualize(motion, "smpl_mesh.html", mode='mesh', body_model='smpl')
```

### Note on 263D (HumanML3D) Format

**Important:** 263D (HumanML3D) format requires expensive SMPLify optimization to convert to SMPL mesh. The visualizer will **not** run SMPLify automatically.

**What happens:**
- If you try to visualize 263D as SMPL mesh, you'll get a clear error message
- The visualizer can fall back to skeleton visualization if `on_incompatible='auto_skeleton'`
- SMPLify must be run manually before mesh visualization

**Options for 263D data:**

1. **Use skeleton visualization** (recommended, fast):
```python
vis.visualize(data_263d, "output.html", mode='skeleton')
```

2. **Convert to 272D first** (if you need mesh):
```python
from motion_toolbox.converters import HumanML3DTo272Converter

converter = HumanML3DTo272Converter()
data_272d = converter.convert(data_263d)
vis.visualize(data_272d, "output.html", mode='mesh', body_model='smpl')
```

3. **Run SMPLify manually** (expensive, for production):
```python
# This requires running the full SMPLify optimization pipeline
# See docs/SMPLIFY_GUIDE.md for details (optimization can take minutes per motion)
```

### Output Files

SMPL mesh visualization generates:
- **HTML file** with embedded Three.js viewer
- **GLB file** with animated mesh (referenced by HTML)
- Interactive 3D viewer with orbit controls
- Play/pause/reset animation controls

### Error Handling

```python
# Automatic fallback to skeleton
vis = Visualizer(on_incompatible='auto_skeleton')
vis.visualize(data, "output.html", mode='mesh')  # Falls back if needed

# Strict mode (raise errors)
vis = Visualizer(on_incompatible='error')
try:
    vis.visualize(data, "output.html", mode='mesh')
except IncompatibleFormatError as e:
    print(f"Error: {e}")
```

### Frontend Integration

```python
# Export as JSON for web apps
scene_data = vis.to_scene_data(data, mode='skeleton')

# Use in Flask/FastAPI/Django
import json
json_str = json.dumps(scene_data)

# scene_data contains:
# - positions: joint positions
# - kinematic_chains: bone connections
# - chain_colors: visualization colors
# - fps, num_frames, metadata
```

## Testing with Real Data

### Using HumanML3D Dataset

```python
import numpy as np
from motion_toolbox.visualization import Visualizer

# Load real HumanML3D data
data = np.load("path/to/humanml3d/000001.npy")

vis = Visualizer()

# Skeleton visualization
vis.visualize(data, "humanml3d_skeleton.html", mode='skeleton')

# Try mesh (will warn about SMPLify)
vis.visualize(data, "humanml3d_mesh.html", mode='mesh', body_model='smpl')
```

### Using Motion Objects

```python
from motion_toolbox.core import Motion
from motion_toolbox.visualization import Visualizer

# Load motion
motion = Motion.from_file("path/to/motion.pkl")

vis = Visualizer()

# Works directly
vis.visualize(motion, "motion_vis.html")
```

## Common Use Cases

### 1. Quick Preview
```python
vis = Visualizer()
vis.visualize(data, "preview.html")  # Auto-everything
```

### 2. Comparison Study
```python
# Generate multiple visualizations
for motion_id in motion_ids:
    data = load_motion(motion_id)
    vis.visualize(data, f"study/{motion_id}.html", mode='skeleton')
```

### 3. Web Application
```python
from flask import Flask, jsonify
from motion_toolbox.visualization import Visualizer

app = Flask(__name__)
vis = Visualizer()

@app.route('/api/motion/<id>')
def get_motion(id):
    data = load_motion(id)
    return jsonify(vis.to_scene_data(data, mode='skeleton'))
```

### 4. Batch Processing
```python
import glob
from pathlib import Path

vis = Visualizer()
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

for data_file in glob.glob("data/*.npy"):
    data = np.load(data_file)
    name = Path(data_file).stem
    vis.visualize(data, output_dir / f"{name}.html")
```

## Troubleshooting

### Format Not Detected
```python
# Use format hint
vis.visualize(data, "output.html", format='263d')
```

### SMPLify Required Error
```python
# Option 1: Use skeleton mode
vis.visualize(data, "output.html", mode='skeleton')

# Option 2: Convert to 272D first
from motion_toolbox.converters import HumanML3DTo272Converter
converter = HumanML3DTo272Converter()
data_272d = converter.convert(data_263d)
vis.visualize(data_272d, "output.html", mode='mesh')
```

### Body Model Path Not Found
```python
vis = Visualizer(body_model_path="/path/to/smpl/models")
```

## Performance Tips

1. **Skeleton is 10x faster** than mesh
2. **Cache HTML files** for repeated viewing
3. **Downsample for preview**: `motion.downsample(2)`
4. **Use 272D format** for fastest mesh visualization
5. **Scene data export** is fast for web apps

## Further Documentation

- **Complete Guide**: `docs/UNIFIED_VISUALIZER_GUIDE.md`
- **Migration Guide**: `docs/MIGRATION_GUIDE.md`
- **Frontend Integration**: `visualization/examples/README_embedding.md`
- **API Reference**: See docstrings in `src/motion_toolbox/visualization/unified_visualizer.py`

## Questions?

- Check documentation in `docs/`
- Run comprehensive test to see all examples
- Examine generated HTML files for implementation details
- Review unit tests in `tests/test_unified_visualizer.py`
