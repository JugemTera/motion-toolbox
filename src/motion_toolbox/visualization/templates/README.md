# Visualization Templates

This directory contains HTML/CSS/JavaScript templates for motion visualization tools.

## Overview

Templates are extracted from Python code to improve:
- **Separation of concerns** - HTML/CSS/JS separated from Python logic
- **Maintainability** - Edit templates with proper syntax highlighting
- **Readability** - No massive f-strings in Python code
- **Reusability** - Templates can be shared across visualizers

## Template System

Templates use a simple placeholder syntax: `{{VARIABLE_NAME}}`

The `TemplateRenderer` class (in `template_renderer.py`) handles:
- Loading templates from this directory
- Replacing placeholders with actual values
- Generating conditional code sections (e.g., ground plane variations)

## Available Templates

### `spatial_grid_viewer.html`

Three.js-based viewer for displaying multiple animated characters in a 3D spatial grid.

**Placeholders:**
- `{{TITLE}}` - Page title
- `{{SCENE_DATA}}` - JSON array of animation data (GLB paths, positions, labels, details)
- `{{MATERIAL_TYPE}}` - Material type ("phong", "lambert", or "physical")
- `{{MESH_COLOR_HEX}}` - Mesh color as hex (e.g., "0x6495ED")
- `{{GROUND_PLANE_CODE}}` - JavaScript code for ground plane (checkerboard or solid)

**Features:**
- Multiple characters in a single 3D scene
- Synchronized playback controls
- Billboard labels for each character
- Click-to-show details panel
- Keyboard shortcuts (L for labels, I for info)
- Customizable materials and colors
- Optional checkerboard floor

## Usage Example

```python
from motion_toolbox.visualization.template_renderer import (
    TemplateRenderer,
    render_ground_plane_code
)
import json

# Prepare data
scene_data = [
    {"glb": "animation1.glb", "position": [0, 0, 0], "label": "Walk"},
    {"glb": "animation2.glb", "position": [2, 0, 0], "label": "Run"}
]

# Create context
context = {
    "TITLE": "My Animation Viewer",
    "SCENE_DATA": json.dumps(scene_data, indent=8),
    "MATERIAL_TYPE": "phong",
    "MESH_COLOR_HEX": "0x6495ED",
    "GROUND_PLANE_CODE": render_ground_plane_code(checkerboard=True)
}

# Render template
renderer = TemplateRenderer()
html = renderer.render("spatial_grid_viewer.html", context)

# Save to file
Path("output.html").write_text(html)
```

## Adding New Templates

1. Create a new `.html` file in this directory
2. Use `{{PLACEHOLDER}}` syntax for variables
3. Add documentation above describing placeholders
4. Update `template_renderer.py` if you need custom logic (like ground plane code generation)
5. Use the `TemplateRenderer` class to render your template

## Best Practices

- Use UPPERCASE for placeholder names (e.g., `{{TITLE}}`, not `{{title}}`)
- Keep templates self-contained (include all CSS and JS inline)
- Document all placeholders at the top of each template file
- Test templates independently before integration
- Consider adding helper functions in `template_renderer.py` for complex conditional logic


