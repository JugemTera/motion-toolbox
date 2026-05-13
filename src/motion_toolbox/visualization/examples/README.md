# Spatial Animation Grid Configuration Examples

This directory contains example configuration files for creating spatial animation grids.

## Quick Start

### Using a Configuration File

```python
from motion_toolbox.visualization import create_spatial_animation_grid_from_config

# Create grid from config
create_spatial_animation_grid_from_config("examples/simple_grid.yaml")
```

Or from command line:

```bash
python -c "from motion_toolbox.visualization import create_spatial_animation_grid_from_config; \
           create_spatial_animation_grid_from_config('src/motion_toolbox/visualization/examples/simple_grid.yaml')"
```

## Configuration Format

Configuration files can be in YAML or JSON format. Here's a complete example:

```yaml
# Output settings
output_dir: outputs/my_grid
output_html: grid.html
title: My Animation Grid

# Animation sources (GLB or .pt files)
animations:
  - path: path/to/animation1.glb
    label: Walk
    details:
      Category: Locomotion
      Subcategory: Walk
      Text: Person walking
      Model: GT
  
  - path: path/to/animation2.glb
    label: Run
    details:
      Category: Locomotion
      Model: GT

# Grid layout
grid_cols: 2      # Number of columns (null for auto)
spacing: 3.0      # Spacing in meters

# Visual settings
material_type: physical    # Options: phong, lambert, physical
mesh_color: blue          # Options: white, blue, clay, grey
show_ground: true         # Show ground plane
checkerboard_floor: true  # Checkerboard pattern (false for solid)

# Embedding
embed_glbs: true  # Embed GLBs as base64 (self-contained HTML)

# Body model (only for .pt conversion)
body_model_path: null
```

## Available Examples

### `simple_grid.yaml`
Basic configuration with 4 animations in a 2x2 grid.

### `grid_config_example.yaml`
Comprehensive example showing all available options with detailed annotations.

## Configuration Options

### Output Settings

- **`output_dir`** (string): Directory for output files
- **`output_html`** (string): HTML filename
- **`title`** (string): Page title

### Animations

- **`animations`** (list): List of animations to display

Each animation can be:
- A simple string path: `"path/to/file.glb"`
- An object with path, label, and details:
  ```yaml
  - path: path/to/file.glb
    label: Short Label
    details:
      Field1: Value1
      Field2: Value2
  ```

### Grid Layout

- **`grid_cols`** (int or null): Number of columns (auto-calculated if null)
- **`spacing`** (float): Spacing between characters in meters

### Visual Settings

- **`material_type`** (string): Material type
  - `"phong"`: Phong shading with specular highlights
  - `"lambert"`: Lambertian (diffuse) shading
  - `"physical"`: Physically-based rendering (PBR)

- **`mesh_color`** (string): Mesh color
  - `"white"`: Pure white
  - `"blue"`: Cornflower blue
  - `"clay"`: Light tan/clay color
  - `"grey"`: Medium grey

- **`show_ground`** (bool): Whether to show ground plane
- **`checkerboard_floor`** (bool): Use checkerboard pattern (false for solid color)

### Embedding

- **`embed_glbs`** (bool): Embed GLB files as base64 data URIs
  - `true`: Self-contained HTML file (larger size)
  - `false`: HTML references external GLB files (smaller HTML, but needs GLB files)

### Body Model

- **`body_model_path`** (string or null): Path to SMPL body model
  - Only needed when converting `.pt` motion files
  - Can be null for GLB inputs

## New Viewer Features

The latest spatial animation grid viewer includes:

### ⏯️ Timeline Scrubber
- Drag the slider to scrub through animations
- Real-time time display (current / total duration)
- Positioned above playback controls

### 🎮 Playback Controls
- **▶ Play**: Start/resume animation
- **⏸ Pause**: Pause animation
- **⏮ Restart**: Jump to beginning

### 🏷️ Labels
- Billboard labels above each character
- Toggle with "Labels" button or "L" key

### ℹ️ Details Panel
- Click any character to view details
- Shows category, subcategory, text, model info
- Toggle panel with "Info" button or "I" key

### 🎨 Visual Quality
- Multiple material types (Phong, Lambert, Physical)
- Customizable mesh colors
- Optional checkerboard floor
- High-quality shadows and lighting
- Antialiasing for smooth edges

### 📐 Compact UI
- Smaller, more space-efficient controls
- Timeline scrubber positioned for easy access
- View options panel in bottom-right corner

## Creating Your Own Config

1. Copy `simple_grid.yaml` as a starting point
2. Update `animations` with your GLB or .pt file paths
3. Customize visual settings (material, color, floor)
4. Adjust grid layout (columns, spacing)
5. Run: `create_spatial_animation_grid_from_config("your_config.yaml")`

## Tips

- Use **relative paths** in config files for portability
- Set `embed_glbs: false` for faster iteration (smaller HTML files)
- Use `embed_glbs: true` for final output (self-contained, shareable)
- The `details` dict can contain any key-value pairs you want to display
- Use `grid_cols: null` to auto-calculate optimal grid layout


