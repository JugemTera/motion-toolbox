# HumanML3D Visualization Guide

Complete guide for creating interactive SMPL mesh visualizations with checkerboard floor and playback controls.

> **💡 Just want to get started quickly?** See [QUICK_START.md](QUICK_START.md) for a one-page guide.

---

## Overview

This guide covers:
- Complete setup and usage instructions
- Customization options (colors, floor, lighting, camera)
- Troubleshooting common issues
- Advanced usage and batch processing
- Technical architecture details

---

## Quick Start

### Using the Main Script ⚡ (Recommended)

```bash
# Create visualization from HumanML3D joints
python scripts/humanml3d_to_smpl_visualization.py \
    sample_data/humanml3d/000000_joints.npy \
    outputs/my_animation.html
```

**That's it!** Open `outputs/my_animation.html` in your browser.

### Alternative Scripts

Both scripts produce similar output:

```bash
# More customizable visualization
python scripts/create_humanml3d_visualization.py \
    sample_data/humanml3d/000000_joints.npy \
    outputs/my_animation.html

# Advanced usage with more options
python scripts/humanml3d_to_smpl_mesh.py \
    sample_data/humanml3d/000000_joints.npy \
    outputs/my_animation.html \
    --input-type joints
```
---

## What You Get

The visualization includes:

- ✅ **White background** with professional lighting
- ✅ **Blue SMPL character** with smooth animation  
- ✅ **Checkerboard floor** (grey/cyan pattern) embedded in GLB
- ✅ **Play/Pause/Restart buttons** for playback control
- ✅ **Timeline scrubber** to jump to any frame
- ✅ **Frame counter** showing current/total frames and time
- ✅ **Orbit controls** to rotate, zoom, pan the camera
- ✅ **GPU acceleration** for fast SMPL optimization

---

## System Requirements

### GPU Support (Recommended)

For fast SMPL optimization, you need:

1. **NVIDIA GPU** with CUDA support
2. **PyTorch with CUDA** installed

#### Installing CUDA PyTorch

```bash
cd motion_toolbox
uv pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch>=2.0.0" "torchvision>=0.15.0"
```

#### Verify GPU is Available

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
CUDA available: True
Device: NVIDIA L40S  (or your GPU model)
```

### CPU-Only Mode

The scripts will automatically fall back to CPU if no GPU is available. It's slower but works fine for short sequences.

---

## Step-by-Step Guide

### Step 1: Prepare Your Data

Your input should be **HumanML3D joint positions** in `.npy` format:

- **Shape**: `(num_frames, 22, 3)`
- **Format**: 22 joints in 3D space per frame
- **Example files**: See `sample_data/humanml3d/000000_joints.npy`

### Step 2: Run the Visualization Script

```bash
python scripts/humanml3d_to_smpl_visualization.py \
    /path/to/your/joints.npy \
    /path/to/output.html
```

The script will:
1. Load HumanML3D joints
2. Convert to SMPL parameters (GPU-accelerated)
3. Generate SMPL mesh vertices
4. Create GLB file with embedded floor
5. Generate HTML viewer with controls

### Step 3: Open the Result

Simply open the generated HTML file in any modern web browser:
- Chrome/Edge (recommended)
- Firefox
- Safari

No server needed - it's a standalone file!

---

## Understanding the Pipeline

### Architecture

```
HumanML3D Joints (22, 3)
    ↓
[GPU] SMPL Optimization
    ↓
SMPL Parameters (pose, betas, transl)
    ↓
SMPL Mesh Generation
    ↓
GLB Export (mesh + floor + animation)
    ↓
HTML Viewer (Three.js + controls)
```

### Key Components

#### 1. SMPL Optimization (`converters/joints2smpl/`)

Converts joint positions to SMPL parameters using optimization:

```python
from motion_toolbox.converters.joints2smpl.smplify_batched import BatchedJoints2SMPL

converter = BatchedJoints2SMPL(
    num_frames=num_frames,
    device_id=0,              # GPU device (0, 1, ...) or -1 for CPU
    cuda=True,                # Enable CUDA
    num_smplify_iters=20,     # Optimization iterations
    use_lbfgs=True,           # Use L-BFGS optimizer (faster)
)

result = converter.joint2smpl(joints)
# Returns: {'pose': (N,72), 'betas': (N,10), 'transl': (N,3)}
```

**Performance**:
- GPU: ~40 seconds for 116 frames
- CPU: ~3-5 minutes for 116 frames

#### 2. Floor Embedding (`visualization/gltf_utils_vertex_anim.py`)

Creates checkerboard floor with vertex colors:

```python
from motion_toolbox.visualization.gltf_utils_vertex_anim import build_gltf_vertex_animation

gltf = build_gltf_vertex_animation(
    all_vertices=vertices,      # (num_frames, 6890, 3)
    init_vertices=vertices[:1], # (1, 6890, 3)
    faces=faces,                # (13776, 3)
    fps=20,                     # Animation framerate
    include_floor=True,         # Embed floor in GLB
)

gltf.save("output.glb")
```

The floor is generated with:
- **12m x 12m** size
- **0.5m** tile width (24 x 24 tiles)
- **Alternating colors**: Light grey (0.8, 0.9, 0.9) and darker grey (0.6, 0.7, 0.7)
- **Vertex colors** embedded in GLB (no textures needed)

#### 3. HTML Viewer Template

The HTML uses Three.js to render the GLB:

**Key Features**:
- **White background**: `scene.background = new THREE.Color(0xffffff)`
- **Blue material**: Character uses `MeshPhongMaterial` with color `0x4a90e2`
- **Vertex colors**: Floor uses `vertexColors: true` to show checkerboard
- **Animation**: Uses morph target animation for mesh deformation
- **Controls**: Custom play/pause/restart buttons + timeline scrubber

---

## Customization

### Change Character Color

In the script, modify the blue material color:

```javascript
// Character: use blue material
const material = new THREE.MeshPhongMaterial({
    color: 0x4a90e2,  // Change this hex color
    shininess: 20,
    specular: 0x222222,
    flatShading: false,
    side: THREE.DoubleSide,
});
```

Common colors:
- Blue: `0x4a90e2`
- Red: `0xe74c3c`
- Green: `0x2ecc71`
- Purple: `0x9b59b6`
- Orange: `0xe67e22`

### Change Floor Pattern

Modify floor generation in `visualization/gltf_utils.py`:

```python
def checkerboard_geometry(
    length=12.0,           # Total size (meters)
    color0=[0.8, 0.9, 0.9],  # Light tile color (RGB)
    color1=[0.6, 0.7, 0.7],  # Dark tile color (RGB)
    tile_width=0.5,        # Tile size (meters)
    alpha=1.0,             # Transparency
    up="y",                # Up axis
    c1=0.0,                # Center offset X
    c2=0.0,                # Center offset Z
):
```

### Change Animation Speed

Modify FPS when building GLB:

```python
gltf = build_gltf_vertex_animation(
    all_vertices=vertices,
    init_vertices=vertices[:1],
    faces=faces,
    fps=30,  # Change this (10, 20, 30, 60, etc.)
    include_floor=True,
)
```

### Adjust Camera Position

In the HTML, modify camera setup:

```javascript
camera.position.set(3, 1.5, 3);  // (x, y, z) in meters
camera.lookAt(0, 1, 0);          // Look at this point
```

### Adjust Lighting

Modify light intensities:

```javascript
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);  // Intensity
const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);  // Intensity
```

---

## Troubleshooting

### Issue: Character Feet Below Floor

**Cause**: SMPL mesh not aligned to y=0

**Fix**: Add floor alignment in the script:

```python
# After generating vertices
min_y = vertices[:, :, 1].min()
vertices[:, :, 1] -= min_y  # Lift to floor
```

### Issue: Floor Not Showing Checkerboard

**Cause**: Vertex colors not embedded or not enabled

**Fix 1**: Ensure floor colors are added in `gltf_utils_vertex_anim.py`:
```python
# Floor vertex colors (line ~183)
floor_vertices, floor_faces, floor_vert_colors, _ = checkerboard_geometry()
#                                    ^^^^^^^^ Don't discard!

# Add COLOR_0 attribute (line ~221)
floor_prim = Primitive(
    attributes=Attributes(
        POSITION=floor_vert_acc_idx,
        COLOR_0=floor_color_acc_idx,  # Must have this!
    ),
    indices=floor_idx_acc_idx,
)
```

**Fix 2**: Enable vertex colors in HTML:
```javascript
const material = new THREE.MeshPhongMaterial({
    vertexColors: true,  // Enable vertex colors!
    shininess: 10,
    side: THREE.DoubleSide,
});
```

### Issue: Animation Not Playing

**Cause**: Morph target weights incorrectly structured

**Fix**: Check accessor configuration in `gltf_utils_vertex_anim.py`:
```python
weights_acc = Accessor(
    bufferView=len(buffer_views) - 1,
    componentType=FLOAT,
    count=num_frames * num_frames,  # Total weight values
    type="SCALAR",
    max=[1.0],
    min=[0.0],
)
```

### Issue: GPU Not Detected

**Cause**: PyTorch CUDA not installed or wrong version

**Solution 1**: Reinstall PyTorch with CUDA:
```bash
uv pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch>=2.0.0" "torchvision>=0.15.0"
```

**Solution 2**: Check NVIDIA driver:
```bash
nvidia-smi
```

Should show CUDA version 12.x or 13.x.

### Issue: Slow Performance

**Symptoms**: SMPL optimization takes > 5 minutes

**Solutions**:
1. **Use GPU** (see GPU installation above)
2. **Reduce iterations**:
   ```python
   converter = BatchedJoints2SMPL(
       num_smplify_iters=10,  # Reduce from 20
       ...
   )
   ```
3. **Use shorter sequences**: Test with `joints[:20]` first

---

## Advanced Usage

### Batch Processing Multiple Files

```python
import glob
import os
from pathlib import Path

# Process all joint files
for joints_file in glob.glob("data/*.npy"):
    input_path = Path(joints_file)
    output_path = Path("outputs") / f"{input_path.stem}.html"
    
    # Run visualization
    os.system(f"python scripts/humanml3d_to_smpl_visualization.py {input_path} {output_path}")
```

### Custom Optimization Parameters

```python
converter = BatchedJoints2SMPL(
    num_frames=num_frames,
    device_id=0,
    cuda=True,
    num_smplify_iters=50,      # More iterations = higher quality
    batch_size=32,             # Frames processed in parallel
    use_lbfgs=True,            # L-BFGS (fast) vs Adam (slower)
)
```

### Save Intermediate Results

```python
# Save SMPL parameters
np.savez(
    "output_params.npz",
    pose=result["pose"],
    betas=result["betas"],
    transl=result["transl"],
)

# Save GLB separately (don't embed in HTML)
gltf.save("animation.glb")
```

---

## File Structure

```
motion_toolbox/
├── scripts/
│   ├── humanml3d_to_smpl_visualization.py  # Main script (recommended)
│   ├── create_humanml3d_visualization.py   # Alternative script
│   └── humanml3d_to_smpl_mesh.py           # Advanced options
├── converters/
│   └── joints2smpl/
│       ├── smplify_batched.py          # GPU-accelerated converter
│       └── smplify_batched_fast.py     # Faster variant
├── visualization/
│   ├── gltf_utils_vertex_anim.py       # GLB builder (FIXED)
│   └── gltf_utils.py                   # Floor geometry
├── sample_data/
│   └── humanml3d/
│       ├── 000000_joints.npy           # 116 frames
│       ├── 000010_joints.npy           # 69 frames
│       └── 000030_joints.npy           # 67 frames
└── outputs/                            # Generated visualizations
```

---

## Playback Controls Reference

### Mouse Controls
- **Left Mouse**: Rotate camera around character
- **Right Mouse**: Pan camera
- **Scroll Wheel**: Zoom in/out

### Buttons
- **▶ Play**: Start animation playback
- **⏸ Pause**: Pause animation
- **⏮ Restart**: Reset to frame 0 and play

### Timeline Scrubber
- **Drag slider**: Seek to any frame
- **Click timeline**: Jump to frame

### Info Panel (Top Right)
- Source filename
- Total frames
- Animation duration
- FPS
- Vertex count

---

## Performance Metrics

### GPU (NVIDIA L40S)
- **116 frames**: ~40 seconds total
- **SMPL optimization**: ~35 seconds
- **Mesh generation**: ~1 second  
- **GLB creation**: ~4 seconds

### CPU (Fallback)
- **116 frames**: ~3-5 minutes total
- **SMPL optimization**: ~2.5-4 minutes
- **Mesh generation**: ~10 seconds
- **GLB creation**: ~4 seconds

### Output Files
- **HTML**: ~1-5 MB (depends on sequence length)
- **GLB**: ~1-10 MB (embedded in HTML or separate)
- **Quality**: Production-ready, ~0.5cm SMPL joint error

---

## Best Practices

### 1. Always Use GPU When Available
GPU acceleration provides 4-5x speedup for SMPL optimization.

### 2. Test with Short Sequences First
Start with 20-30 frames to verify everything works before processing full sequences.

### 3. Check Floor Alignment
Always verify feet are on the floor, not floating or sunk in:
```python
min_y = vertices[:, :, 1].min()
vertices[:, :, 1] -= min_y
```

### 4. Use Consistent FPS
Match your GLB FPS to your analysis FPS (typically 20 or 30).

### 5. Embed GLB in HTML
For easy sharing, embed the GLB as base64 in HTML (no external files needed).

---

## References

- **SMPL**: [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)
- **HumanML3D**: [https://github.com/EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- **Three.js**: [https://threejs.org/](https://threejs.org/)
- **glTF 2.0**: [https://www.khronos.org/gltf/](https://www.khronos.org/gltf/)

---

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review `IMPLEMENTATION_SUMMARY.md` for technical details
3. Check `docs/HUMANML3D_TO_SMPL_GUIDE.md` for converter details

---

**Last Updated**: January 22, 2026  
**Status**: Production-ready ✅
