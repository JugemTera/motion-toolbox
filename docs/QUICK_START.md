# Quick Start: Creating HumanML3D Visualizations

## ⚡ One Command to Rule Them All

```bash
python scripts/humanml3d_to_smpl_visualization.py \
    sample_data/humanml3d/000000_joints.npy \
    outputs/my_animation.html
```

Then open `outputs/my_animation.html` in your browser!

---

## What You Get

✅ **Real HumanML3D motion data** (116 frames of actual movement)  
✅ **White background** with professional lighting  
✅ **Blue SMPL character** with smooth mesh animation  
✅ **Checkerboard floor** (grey/cyan) embedded in GLB  
✅ **Play/Pause/Restart buttons**  
✅ **Timeline scrubber** to jump to any frame  
✅ **Frame counter** and time display  
✅ **GPU-accelerated** SMPL optimization (~40s for 116 frames)

---

## Files Generated

- **`outputs/000000_demo.html`** - Interactive viewer (open in browser)
- **`outputs/000000_demo.glb`** - 3D mesh with animation

---

## Try Other Samples

```bash
# 69 frames
python scripts/humanml3d_to_smpl_visualization.py \
    sample_data/humanml3d/000010_joints.npy \
    outputs/sample_010.html

# 67 frames  
python scripts/humanml3d_to_smpl_visualization.py \
    sample_data/humanml3d/000030_joints.npy \
    outputs/sample_030.html
```

---

## GPU Setup (Faster Processing)

If you have an NVIDIA GPU, install CUDA PyTorch:

```bash
uv pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch>=2.0.0" "torchvision>=0.15.0"
```

Then verify:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

**Performance**:
- GPU: ~40 seconds for 116 frames
- CPU: ~3-5 minutes for 116 frames

---

## Need More Details?

See **`docs/VISUALIZATION_GUIDE.md`** for:
- Step-by-step tutorial
- Customization options (colors, floor, lighting)
- Troubleshooting guide
- Advanced usage examples
- Technical architecture

---

## Quick Fixes

### Feet Below Floor?
The script automatically aligns feet to y=0.

### No Checkerboard?
Fixed! Floor colors are now properly embedded in GLB.

### Animation Not Playing?
This is now working with real HumanML3D data.

### Want Different Colors?
Edit the HTML material color: `color: 0x4a90e2` (hex RGB)

---

**Ready to visualize? Run the command above!** 🚀
