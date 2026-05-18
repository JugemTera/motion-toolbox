# Motion Toolbox

A Python library for 3D character animation and motion processing, supporting multiple rotation representations and ML frameworks.

## Features

- **🎨 Unified Visualization System** (NEW): Single API for all motion formats and body models
  - Automatic format detection (263D, 272D, 205D, 232D ALIGN, Motion, joints)
  - Multi-body model support (SMPL, MHR)
  - Skeleton and mesh visualization
  - Frontend integration with JSON export
- **Multiple Rotation Formats**: Quaternions and axis-angles with seamless conversion
- **Forward Kinematics**: Compute joint positions from skeletal motion data
- **ML Framework Support**: PyTorch integration with tensor conversion utilities
- **Motion Processing**: Slicing, downsampling, and motion manipulation tools
- **Format Converters**: Transform between common motion formats (HumanML3D, SMPL, STMC, MHR, etc.)
- **Body Model Support**: SMPL and MHR (Momentum Human Rig) body models
- **File I/O**: Robust serialization with PyTorch and numpy compatibility

## Project status

Research code for the **ROMO** project: APIs may change between releases. Not intended for production use without your own hardening.

## Dataset

The companion in-the-wild motion dataset is hosted on Hugging Face Hub.
The link below is a placeholder until the public mirror is finalised; the
toolbox itself doesn't depend on the dataset to function — every loader
also accepts a local folder of motion files.

> 📦 **Hugging Face mirror (placeholder):** `https://huggingface.co/datasets/<HF_USER>/<DATASET_NAME>`
>
> Once the dataset is published, drop the URL into
> `scripts/fetch_sample_data_template.sh` and
> `scripts/download_272_samples_template.py` (they already contain the
> wiring for `huggingface_hub.snapshot_download`).

A typical end-to-end loop — load a folder of `.npy` motions, batch them,
and visualise the batch as HTML — looks like this:

```python
from torch.utils.data import DataLoader

from motion_toolbox.data import Format272Dataset, motion_collate
from motion_toolbox.visualization import Visualizer

dataset = Format272Dataset("path/to/your/272d/folder")
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=motion_collate)

for batch in loader:
    rotations = batch["joint_rotations"]    # (B, T_max, J, 3) torch.float32
    translation = batch["root_translation"]  # (B, T_max, 3)
    mask = batch["mask"]                     # (B, T_max) bool

# Render one HTML viewer per sample for the first eight items
Visualizer().visualize_batch(dataset, output_dir="outputs/preview", limit=8)
```

See [`docs/DATALOADERS.md`](docs/DATALOADERS.md) for the full dataloader
API (generic folder dataset, 272-D dataset, HumanML3D dataset, custom
transforms, and using the collated batches for training).

## Installation

From a clone of [the public repository](https://github.com/<github-org>/motion-toolbox) (update the URL when published):

```bash
cd motion_toolbox
uv sync                    # core dependencies
uv sync --group dev        # + pytest, ruff, mypy, torch (for tests)
# Optional extras:
uv sync --extra ml         # JAX + full ML stack (see pyproject optional-dependencies)
uv sync --extra viz        # OpenCV / imageio / video helpers
uv sync --extra mhr        # Meta MHR (see docs/MHR_SETUP_GUIDE.md)
uv sync --extra semantic-eval
uv sync --extra report
```

Editable install (recommended for development):

```bash
pip install -e ".[ml]"     # or uv pip install -e .
```

The importable package name is `motion_toolbox` (Python module); the distribution name on PyPI-style metadata is `motion-toolbox`.

## MHR (Momentum Human Rig) Support

MHR is a high-fidelity body model from Meta with 127 joints and realistic deformations.

### Prerequisites

- **Pixi**: conda-based environment manager (auto-installed by setup script if missing)
- **UV**: Python package manager (for the main project)

### Quick Setup

```bash
# 1. Run the setup script (one-time setup)
bash scripts/setup_mhr.sh

# This script will:
# - Install Pixi if not present
# - Clone MHR repository to /tmp/MHR
# - Download MHR assets
# - Build pymomentum with Pixi
# - Copy libraries and Python packages to UV environment
# - Install MHR assets to ~/.motion_toolbox/datasets/mhr/
```

**Important**: This project uses **PyTorch 2.8.0** to match MHR's compiled dependencies. The setup script and `pyproject.toml` handle this automatically.

### Running MHR Code

Always use the `run_with_mhr.sh` wrapper to ensure library paths are set correctly:

```bash
# Visualize MHR motion data
bash scripts/run_with_mhr.sh python demos/mhr_visualization_demo.py

# Compare GVHMR (SMPL) vs MHR side-by-side
bash scripts/run_with_mhr.sh python demos/gvhmr_to_mhr_demo.py
```

### Troubleshooting

See `docs/MHR_SETUP_GUIDE.md` for detailed setup instructions, troubleshooting, and common issues.

### Generate Complete Visualization Showcase

To generate the full showcase including all 12 visualizations (with MHR):

```bash
# Setup MHR first (one-time setup)
bash scripts/setup_mhr.sh

# Generate complete showcase
bash scripts/run_with_mhr.sh python -m pytest tests/test_generate_showcase.py -v

# Open outputs/showcase_3d/index.html in your browser
```

**Note**: Without `run_with_mhr.sh`, MHR visualizations will be skipped (10/12 will work).

## Quick Start

### Unified Visualization (NEW!)

Visualize motion in any format with one line:

```python
from motion_toolbox.visualization import Visualizer

# Works with ANY format - auto-detects and visualizes!
vis = Visualizer()
vis.visualize(motion_data, "output.html")  # That's it!

# Supports: 263D (HumanML3D), 272D, 205D (STMC), 232D (ALIGN), Motion objects, joint positions
# Modes: skeleton (fast), mesh (SMPL / MHR)
```

See [`docs/UNIFIED_VISUALIZER_GUIDE.md`](docs/UNIFIED_VISUALIZER_GUIDE.md) for complete guide.

### Basic Motion Creation

```python
import numpy as np
from motion_toolbox import Motion, Skeleton

# Define skeleton structure
joint_names = ('root', 'spine', 'head')
parent_indices = (None, 0, 1)  # spine parents to root, head to spine
offsets = np.array([[0, 0, 0], [0, 1, 0], [0, 0.5, 0]])

skeleton = Skeleton(joint_names, parent_indices, offsets)

# Create motion with quaternion rotations (w, x, y, z format)
quaternion_rotations = np.array([
    [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [1, 0, 0, 0]],  # Frame 1
    [[1, 0, 0, 0], [0.8, 0.2, 0, 0], [0.9, 0.1, 0, 0]]   # Frame 2
])

root_translation = np.array([[0, 0, 0], [0.1, 0, 0]])

motion = Motion(quaternion_rotations, root_translation, skeleton, framerate=30.0)
print(f"Motion: {motion}")  # Motion: Motion(2 frames, 3 joints, 30.0fps, 0.1s)
```

### Axis-Angle Rotations (Native Support)

```python
# Motion class natively stores rotations as axis-angles
axis_angle_rotations = np.array([
    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],  # Frame 1: spine tilts forward 0.1 rad
    [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]   # Frame 2: more tilt
])

# Direct creation with axis-angle rotations
motion = Motion(axis_angle_rotations, root_translation, skeleton, framerate=30.0)

# Access quaternions when needed
quaternions = motion.quaternions  # Computed on-demand from axis-angles
```

### Forward Kinematics

```python
# Compute joint positions in world space
joint_positions = motion.positions  # Shape: (frames, joints, 3)
print(f"Head position at frame 0: {joint_positions[0, 2]}")

# Motion class provides unified interface for all rotation formats
positions = motion.positions  # Forward kinematics with axis-angle rotations
```

### Motion Processing

```python
# Slice motion to specific frame range
subset = motion.slice_frames(start=10, end=50)

# Downsample to reduce framerate
downsampled = motion.downsample(factor=2)  # 30fps -> 15fps

# Get specific frame data
rotations, translation = motion.get_frame(frame_idx=0)
```

### PyTorch Integration

```python
import torch

# Convert motion to PyTorch tensors
motion_tensors = motion.to_torch()  # Returns (rotations_tensor, translation_tensor)
rotations_torch, translation_torch = motion_tensors

print(f"Rotation tensor shape: {rotations_torch.shape}")  # (frames, joints, 4)
print(f"Device: {rotations_torch.device}")  # cpu

# Convert back to Motion object
motion_from_torch = Motion.from_torch(rotations_torch, translation_torch, skeleton)

# Motion supports PyTorch with axis-angle rotations
motion_torch = motion.to_torch()  # Convert to PyTorch tensors
motion_back = Motion.from_torch(*motion_torch, skeleton)  # Convert back to Motion
```

### Rotation Utilities

```python
from motion_toolbox import (
    axis_angle_to_quaternion, 
    quaternion_to_axis_angle,
    q_multiply, 
    q_rotate
)

# Convert between rotation formats
axis_angle = np.array([0.5, 0.0, 1.0])  # 0.5 rad around X, 1.0 rad around Z
quaternion = axis_angle_to_quaternion(axis_angle)
back_to_axis = quaternion_to_axis_angle(quaternion)

# Quaternion operations
q1 = np.array([1, 0, 0, 0])  # Identity
q2 = np.array([0.7071, 0, 0, 0.7071])  # 90° around Z
combined = q_multiply(q1, q2)

# Rotate vector by quaternion
vector = np.array([1, 0, 0])  # X-axis
rotated_vector = q_rotate(quaternion, vector)
```

### Motion Format Converters

```python
from motion_toolbox.converters import get_motion_converter

# Convert Motion to HumanML3D format
converter = get_motion_converter('motion2humanml3d')
humanml_data = converter.convert(motion)  # Returns numpy array
print(f"HumanML3D features shape: {humanml_data.shape}")

# Convert Motion to AxisAngle representation
axis_converter = get_motion_converter('motion2axis_angle')
axis_motion = axis_converter.convert(motion)
print(f"Axis-angle rotations shape: {axis_motion.joint_rotations.shape}")

# File-based processing with automatic serialization
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    # Save motion and convert through files
    torch.save(motion, 'input.pt')
    result = converter.process('input.pt', tmpdir + '/')
    print(f"Conversion result saved to: {result}")

# Available converters
converters = [
    'motion2humanml3d',      # Motion → HumanML3D features
    'humanml3d2motion',      # HumanML3D → Motion  
    'gvhmr2motion',          # GVHMR → Motion
    'smpl2motion',           # SMPL positions → Motion
    'motion2stmc205',        # Motion → STMC 205D features
    'stmc2052motion',        # STMC 205D → Motion
    'motion2align',          # Motion → ALIGN 232D features
    'align2motion',          # ALIGN 232D → Motion
]
```

For STMC (Multi-Track Timeline Control) 205D format details, see `docs/STMC_CONVERTER_GUIDE.md`.

For ALIGN (StableMotion's rotation-invariant representation) details, see `docs/ALIGN_GUIDE.md`.

## Motion Quality Evaluation

The toolbox includes a comprehensive evaluation system for assessing motion quality based on research-validated metrics.

```python
from motion_toolbox.evaluation import MotionEvaluator

# Load evaluation configuration
evaluator = MotionEvaluator.from_config_file("configs/evaluation/final.yaml")

# Evaluate motion quality
report = evaluator.evaluate(motion)

# Access ratio-based metrics (percentage of frames with issues)
print(f"Foot skating: {report.physical_metrics['foot_skating_hoidini_joint'].ratio:.1%}")
print(f"Ground penetration: {report.physical_metrics['ground_penetration_hoidini_joint'].ratio:.1%}")
print(f"Floating: {report.physical_metrics['floating_closd'].ratio:.1%}")
print(f"Jerk (smoothness): {report.physical_metrics['jerk_mean'].ratio:.1%}")
print(f"Dynamic score: {report.physical_metrics['dynamic_score'].ratio:.1%}")
```

### Supported Metrics

**Physical Plausibility:**
- **Foot Skating**: Joint and mesh-based detection (HOIDINI convention)
  - Detects feet sliding during ground contact
  - Ratio < 40% (joint) or < 60% (mesh) is good
- **Ground Penetration**: Floor penetration detection (HOIDINI convention)
  - Detects body parts below ground plane
  - Ratio < 15% is excellent, < 30% is moderate
- **Floating**: Body hovering detection (CLoSD convention)
  - Detects feet hovering above ground unnaturally
  - Ratio < 5% is excellent, < 15% is moderate

**Motion Smoothness:**
- **Jerk Metrics**: Motion smoothness analysis (StableMotion, GVHMR standards)
  - Mean Jerk: Finite difference calculation
  - Gradient Jerk: np.gradient-based (more robust to noise)
  - Ratio < 40% is good, 40-60% is moderate

**Motion Dynamics:**
- **Dynamic Score**: Motion activity and naturalness
  - Ratio > 50% indicates active motion (good)
  - < 10% is static, 10-50% is low activity

### Configuration

Evaluation uses YAML configs with customizable thresholds and ratio assessments:

```yaml
# configs/evaluation/final.yaml
ratio_thresholds:
  foot_skating_joint:
    good: 0.40      # < 40% skating frames is good
    moderate: 0.60  # 40-60% is moderate
  jerk:
    good: 0.40      # < 40% high jerk frames is good
    moderate: 0.60  # 40-60% is moderate
  penetration:
    excellent: 0.15  # < 15% is excellent
    moderate: 0.30   # 15-30% is moderate
```

### HTML Reports

Generate comprehensive HTML reports with interactive visualizations:

```bash
uv run python scripts/gvhmr_motion_evaluation.py \
    --data-file path/to/motion.pt \
    --output-dir outputs/evaluation \
    --config configs/evaluation/final.yaml
```

Reports include:
- Color-coded quality metrics (🟢 Good, 🟡 Moderate, 🔴 Poor)
- Interactive Plotly charts with per-frame analysis
- Detected issue frames highlighted with threshold lines
- Per-joint analysis for applicable metrics
- 3D SMPL mesh animation viewer

See `docs/EVALUATION_GUIDE.md` for detailed documentation.

## Demos

The `demos/` directory contains ready-to-run examples of motion visualization:

### Spatial Animation Grid (Recommended)
Creates a single 3D scene with multiple characters - perfect for research papers:

```bash
uv run python demos/spatial_animation_grid_demo.py
```

**Features:**
- Blue characters with realistic Physical material (PBR)
- 1m × 1m checkerboard floor for spatial reference
- Single unified scene with synchronized playback
- Enhanced antialiasing for publication quality

### Separate Viewers Grid
Creates a grid of independent viewers for side-by-side comparison:

```bash
uv run python demos/separate_viewers_grid_demo.py
```

**Features:**
- Each animation has independent playback controls
- Useful for comparing multiple motion styles
- Grid layout with customizable columns

See `demos/README.md` for detailed documentation and customization options.

## API Overview

### Core Classes
- **`Motion`**: Unified motion representation (axis-angle storage with quaternion access)
- **`Skeleton`**: Kinematic chain definition

### Key Methods
- **Motion Processing**: `slice_frames()`, `downsample()`, `normalize_quaternions()`
- **Conversion**: `to_torch()`, `from_torch()`, `from_quaternions()`
- **Format Access**: `positions`, `quaternions` properties

### Motion Converters
- **Factory Function**: `get_motion_converter(conversion_type)`
- **Available Conversions**: HumanML3D, SMPL, STMC 205D, ALIGN 232D, GVHMR formats
- **File Processing**: Automatic serialization with PyTorch and numpy support
- **ALIGN Support**: StableMotion's rotation-invariant representation for motion cleanup

### Utility Functions
- **Rotation Conversion**: `axis_angle_to_quaternion()`, `quaternion_to_axis_angle()`
- **Quaternion Math**: `q_multiply()`, `q_rotate()`, `q_slerp()`

## Development

```bash
uv sync --group dev
uv run ruff check src tests scripts
uv run ruff format src tests scripts
# Default CI excludes optional GPU / SMPL-on-disk tests:
uv run pytest -m "not requires_smpl and not requires_mhr and not gpu" --tb=short
```

For CUDA PyTorch, install wheels from the [PyTorch index](https://download.pytorch.org/whl/cu124) after `uv sync`.

## Citation

If you use this software, please cite the ROMO paper (update with venue / year / DOI when available):

```bibtex
@article{romo2026,
  title   = {ROMO: Title Placeholder},
  author  = {Zhang, Jiahao and Liu, Joseph and Lee, Young-Yoon and Moon, Seonghyeon and
             Zordan, Victor and Tevet, Guy and Liu, C. Karen and Gould, Stephen and
             Jacob, Oren and Jiang, Haomiao and Kapadia, Mubbasir and Ben{-}Shabat, Yizhak},
  journal = {TODO},
  year    = {2026}
}
```

## Third-party licenses

Bundled data and optional downloads are subject to their own terms. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. See [LICENSE](LICENSE) for the full legal text.

**Non-commercial** means you may not use the material for commercial purposes. Academic and personal research uses are permitted under the license terms, with attribution.