# Motion Toolbox Demos

This directory contains demonstration scripts showing how to use the motion visualization features.

## 📁 Available Demos

### 1. Spatial Animation Grid (`spatial_animation_grid_demo.py`)

Creates a **single 3D scene** with multiple animated characters arranged in a spatial grid.

**Features:**
- ✨ All characters in one unified 3D scene
- 🎮 Single camera and playback controls
- 🎨 Realistic Physical material (PBR)
- 🔵 Blue characters (cornflower blue #6495ED)
- 🎯 1m × 1m checkerboard floor for spatial reference
- 🖼️ Enhanced antialiasing

**Best for:**
- Research papers and presentations
- Showcasing multiple motions in one view
- Spatial relationships between animations
- Professional-looking visualizations

**Run:**
```bash
uv run python demos/spatial_animation_grid_demo.py
```

---

### 2. Separate Viewers Grid (`separate_viewers_grid_demo.py`)

Creates a **grid of independent HTML viewers**, each displaying a single animation.

**Features:**
- 📊 Grid layout of multiple 3D viewers
- 🎮 Each viewer has independent playback controls
- 🏷️ Optional labels for each animation
- 🔄 Side-by-side comparison

**Best for:**
- Comparing different animations
- Animation galleries
- Individual inspection of sequences
- Quality comparison studies

**Run:**
```bash
python demos/separate_viewers_grid_demo.py
```

---

### 3. Keyframe Overlay Viewer (`keyframe_overlay_demo.py`)

Visualizes a **single motion sequence with overlaid keyframe poses** positioned at their actual spatial locations.

**Features:**
- 🎬 Main animation with transparent keyframe poses overlaid
- 📍 Keyframes positioned at actual spatial locations (creates motion trail)
- 🎚️ Interactive opacity slider (0-100%)
- 🔄 Two display modes:
  - **Overlap Mode**: All keyframes visible (motion trail effect)
  - **Disappear Mode**: Keyframes vanish as animation passes
- 🔵 Blue characters with Physical material & checkerboard floor
- ⚙️ Modular keyframe sampling (uniform by default, extensible)

**Best for:**
- Visualizing motion trajectories and paths
- Understanding spatial relationships in a single sequence
- Motion analysis and debugging
- Creating motion trail visualizations

**Run:**
```bash
# Basic usage with sample file
uv run python demos/keyframe_overlay_demo.py

# Quick helper script - use with ANY motion file
uv run python demos/quick_keyframe_overlay.py path/to/your_motion.pt

# With custom settings
uv run python demos/quick_keyframe_overlay.py your_motion.pt \
    --num-keyframes 15 --opacity 0.3 --color white
```

**Customize:**
```python
from motion_toolbox.visualization import create_keyframe_overlay

create_keyframe_overlay(
    motion_path="your_motion.pt",
    output_html_path="output/keyframe_overlay.html",
    body_model_path="~/.motion_toolbox/datasets/body_models_clean_pkl",
    num_keyframes=15,          # More keyframes for denser trail
    keyframe_opacity=0.3,      # More transparent
    mesh_color='white',        # or 'blue', 'clay', 'grey'
    material_type='physical',  # or 'phong', 'lambert'
)
```

---

## 🚀 Quick Start

### Requirements

1. **SMPL Body Models** (required for both demos)
   - Download from: https://smpl.is.tue.mpg.de/download.php
   - Install to: `~/.motion_toolbox/datasets/body_models_clean_pkl/`

2. **Motion Data** 
   - The demos use sample data from `sample_data/gvhmr/`
   - You can also provide your own `.pt` files

### Running a Demo

```bash
# From the repository root
cd /path/to/motion-toolbox

# Run the spatial grid demo (multiple animations)
uv run python demos/spatial_animation_grid_demo.py

# Run the separate viewers demo (comparison grid)
uv run python demos/separate_viewers_grid_demo.py

# Run the keyframe overlay demo (single animation with trail)
uv run python demos/keyframe_overlay_demo.py
```

### Output

Both demos create:
- HTML viewer file (open in browser)
- GLB files (3D models with animation)
- Output directory under `outputs/`

---

## 🎨 Customization

### Spatial Animation Grid

```python
from motion_toolbox.visualization import create_spatial_animation_grid

create_spatial_animation_grid(
    motion_paths=["motion1.pt", "motion2.pt"],
    output_dir="outputs/my_viz/glbs",
    output_html_path="outputs/my_viz/viewer.html",
    body_model_path="~/.motion_toolbox/datasets/body_models_clean_pkl",
    
    # Customize appearance
    material_type="physical",      # or "phong", "lambert"
    mesh_color="blue",             # or "white", "clay", "grey"
    checkerboard_floor=True,       # or False for solid floor
    spacing=2.5,                   # meters between characters
)
```

### Separate Viewers Grid

```python
from motion_toolbox.visualization import create_separate_viewers_grid

create_separate_viewers_grid(
    motion_paths=["motion1.pt", "motion2.pt"],
    output_html_path="outputs/grid.html",
    body_model_path="~/.motion_toolbox/datasets/body_models_clean_pkl",
    
    # Customize layout
    grid_cols=2,                   # number of columns
    labels=["Anim 1", "Anim 2"],   # optional labels
    viewer_width=600,              # pixels
    viewer_height=600,             # pixels
)
```

### Keyframe Overlay Viewer

```python
from motion_toolbox.visualization import create_keyframe_overlay

create_keyframe_overlay(
    motion_path="single_motion.pt",
    output_html_path="outputs/overlay.html",
    body_model_path="~/.motion_toolbox/datasets/body_models_clean_pkl",
    
    # Customize keyframes and appearance
    num_keyframes=10,              # number of keyframe poses
    keyframe_opacity=0.5,          # 0.0 to 1.0
    sampling_strategy="uniform",   # keyframe selection method
    material_type="physical",      # or "phong", "lambert"
    mesh_color="blue",             # or "white", "clay", "grey"
    checkerboard_floor=True,       # or False for solid floor
)
```

---

## 💡 Which Demo Should I Use?

| Use Case | Recommended Demo |
|----------|-----------------|
| Research paper figures | **Spatial Animation Grid** |
| Presentation slides | **Spatial Animation Grid** |
| Motion trajectory analysis | **Keyframe Overlay Viewer** |
| Motion trail visualization | **Keyframe Overlay Viewer** |
| Comparing motions | **Separate Viewers Grid** |
| Animation gallery | **Separate Viewers Grid** |
| Spatial relationships (multi) | **Spatial Animation Grid** |
| Spatial relationships (single) | **Keyframe Overlay Viewer** |
| Individual inspection | **Separate Viewers Grid** |
| Professional look | **Spatial Animation Grid** |

---

## 🐛 Troubleshooting

### "SMPL body models not found"
- Download from https://smpl.is.tue.mpg.de/download.php
- Extract to `~/.motion_toolbox/datasets/body_models_clean_pkl/`

### "No sample PT files found"
- Add `.pt` motion files to `sample_data/gvhmr/`
- Or modify the demo scripts to point to your own data

### "Module not found" errors
- Make sure you're running from the project root
- Or install the package: `pip install -e .`

---

## 📚 More Information

For detailed API documentation, see:
- `src/motion_toolbox/visualization/spatial_animation_grid.py`
- `src/motion_toolbox/visualization/separate_viewers_grid.py`
- `src/motion_toolbox/visualization/keyframe_overlay_viewer.py`

