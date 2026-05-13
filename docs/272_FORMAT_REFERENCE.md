# 272-Dimensional Motion Representation

## Overview

This toolbox implements converters for the 272-dimensional motion representation format, as described in:

**Reference Implementation:**  
https://github.com/Li-xingXiao/272-dim-Motion-Representation

## Format Structure

The 272-dimensional representation encodes motion as:

```
[Root_Lin_Vel (2), Root_Ang_Vel (6), Local_Pos (66), Joint_Vel (66), Rotations_6D (132)]
```

- **Root Linear Velocity XZ** (2 dims): Root translation velocity in X and Z directions
- **Root Angular Velocity** (6 dims): 6D continuous rotation representation of heading changes
- **Local Positions** (66 dims = 22 joints × 3): Joint positions relative to root, in heading-normalized frame
- **Joint Velocities** (66 dims = 22 joints × 3): Frame-to-frame joint velocity  
- **Local Rotations** (132 dims = 22 joints × 6): 6D continuous rotation representation

## Key Properties

### Canonicalized Format
The 272D format is **canonicalized** - motion is represented in a standard upright orientation (Y-up coordinate system) with the character at Y≈0.

**This means:**
- ✅ Format does NOT preserve absolute Y position
- ✅ Format does NOT preserve root Y velocity
- ✅ Characters always decode to upright orientation (by design)
- ✅ Upside-down source data is automatically corrected to upright

### Heading Normalization
All positions and velocities are stored in a "heading-normalized" frame where:
- The character always faces forward (+Z direction)
- Heading changes are encoded separately in the Root Angular Velocity
- This makes the representation rotation-invariant

## Available Converters

### Input to 272D
- **`SMPLTo272Converter`**: SMPL parameters → 272D

### Output from 272D  
- **`Format272ToMotionConverter`**: 272D → Motion class (axis-angle)

### Round-Trip Support
Full round-trip conversions are supported:
- 272D → Motion → SMPL → 272D → Motion ✅
- SMPL → 272D → Motion ✅

### Not Supported
- **HumanML3D → 272D**: This conversion would require IK (HumanML3D positions → SMPL rotations via SMPLify), which is lossy and error-prone. The conversion chain would be HumanML3D → SMPLify → SMPL → 272D (too many hops). Use the direct HumanML3D→Motion or HumanML3D→SMPL converters instead.

## Usage

### Convert SMPL to 272D

```python
from motion_toolbox.converters.motion_converters import SMPLTo272Converter
import numpy as np
from pathlib import Path

smpl_params = {
    'pose_body': pose_body,      # (frames, 21*3)
    'root_orient': root_orient,  # (frames, 3)
    'trans': trans,              # (frames, 3)
    'betas': betas,              # (10,)
}

body_model_path = str(Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl")
converter = SMPLTo272Converter(body_model_path=body_model_path)
motion_272 = converter.convert(smpl_params)
```

### Convert 272D to Motion

```python
from motion_toolbox.converters.motion_converters import Format272ToMotionConverter

# Convert to Motion (with 24 joints for SMPL compatibility)
converter = Format272ToMotionConverter(
    framerate=30.0,
    body_model_path=body_model_path  # Important: Enables 24-joint output!
)
motion = converter.convert(motion_272)

# Motion now has:
# - motion.joint_rotations: (frames, 24, 3) axis-angle rotations
# - motion.root_translation: (frames, 3) root position
# - motion.skeleton: 24-joint SMPL skeleton
```

## Important Notes

### 22 vs 24 Joints

The 272D format stores data for **22 joints** (HumanML3D skeleton).  

For SMPL compatibility (which requires 24 joints), pass `body_model_path` to `Format272ToMotionConverter`:
- **Without `body_model_path`**: Outputs 22 joints (for HumanML3D use)
- **With `body_model_path`**: Outputs 24 joints (for SMPL/visualization use)

The converter automatically adds 2 hand joints (joints 22-23) as children of the wrist joints.

### Orientation Preservation

The 272D format is **canonicalized** and does NOT preserve source orientation:
- Upside-down SMPL data → converts to upright in 272D
- This is **expected behavior**, not a bug
- The format is designed for motion generation/editing where absolute orientation doesn't matter

## Sample Data

Sample 272D files from S3 are available in:
```
sample_data/format_272/
├── 000000.npy  # 221 frames
├── 000001.npy  # 40 frames  
└── Mean.npy    # Mean for normalization
```

## Citation

If you use the 272-dimensional motion representation, please cite the original work:

```bibtex
@InProceedings{Xiao_2025_ICCV,
    author    = {Xiao, Lixing and Lu, Shunlin and Pi, Huaijin and Fan, Ke and Pan, Liang and Zhou, Yueer and Feng, Ziyong and Zhou, Xiaowei and Peng, Sida and Wang, Jingbo},
    title     = {MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025},
}
```

## See Also

- [272D Converters Implementation](../docs/272_DIM_CONVERTERS.md)
- [Reference Repository](https://github.com/Li-xingXiao/272-dim-Motion-Representation)
