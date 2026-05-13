# HumanML3D to SMPL Conversion - Standard Workflow

## Overview

This is the **standard, optimized workflow** for converting HumanML3D motion data to SMPL parameters and visualizations.

## Performance

- **Quality**: ~1.0cm mean error (excellent, production-ready)
- **Speed**: ~10s for 20 frames (5x speedup from original baseline)
- **Configuration**: 20 LBFGS iterations with batching (batch_size=32)

## Quick Start

### 1. Convert Joints to SMPL Parameters

```python
from motion_toolbox.converters.joints2smpl.smplify_batched import BatchedJoints2SMPL
import numpy as np

# Load HumanML3D joints (N, 22, 3)
joints = np.load('sample_data/humanml3d/000030_joints.npy')

# Convert to SMPL parameters
converter = BatchedJoints2SMPL(
    num_frames=len(joints),
    device_id=-1,  # Use -1 for CPU, 0 for GPU
    cuda=False,    # Set True for GPU
)

result = converter.joint2smpl(joints)
# Returns: {'pose': (N,72), 'betas': (N,10), 'transl': (N,3)}
```

### 2. Visualize as SMPL Mesh

```bash
# Direct visualization from joints (recommended)
python scripts/humanml3d_to_smpl_visualization.py \
    sample_data/humanml3d/000030_joints.npy \
    output/visualization.html

# Alternative: More customizable script
python scripts/create_humanml3d_visualization.py \
    sample_data/humanml3d/000030_joints.npy \
    output/visualization.html

# Open output/visualization.html in browser
```

## Configuration Options

### Default (Recommended)
- **Iterations**: 20 (LBFGS optimizer)
- **Batch size**: 32
- **Quality**: ~1.0cm error
- **Speed**: ~10s for 20 frames

### For Higher Quality (if needed)
```python
converter = BatchedJoints2SMPL(
    num_frames=len(joints),
    num_smplify_iters=50,  # More iterations
    batch_size=32,
    use_lbfgs=True,
)
```

### For Maximum Speed (slight quality loss)
```python
converter = BatchedJoints2SMPL(
    num_frames=len(joints),
    num_smplify_iters=10,  # Fewer iterations
    batch_size=32,
    use_lbfgs=True,
)
```

## Input Formats Supported

1. **HumanML3D Joints** (`.npy`): Shape `(N, 22, 3)`
   - Direct joint positions
   - Most common format

2. **HumanML3D 263D** (`.npy`): Shape `(N, 263)`
   - Full HumanML3D representation
   - Automatically converted to joints first

## Output Format

The converter returns a dictionary with:
- `pose`: SMPL pose parameters `(N, 72)` - axis-angle rotations for 24 joints
- `betas`: Shape parameters `(N, 10)` - body shape coefficients
- `transl`: Root translation `(N, 3)` - global position
- `vertices`: Mesh vertices `(N, 6890, 3)` (optional)
- `joints`: Joint positions `(N, 45, 3)` (optional)

## Visualization Features

The generated HTML viewer includes:
- Interactive 3D SMPL mesh with vertex animation
- Blue colored character
- Checkerboard floor
- Play/pause controls
- Frame scrubber
- Embedded GLB (no external dependencies)

## Advanced Usage

### Batch Processing Multiple Sequences

```python
import glob
from motion_toolbox.converters.joints2smpl.smplify_batched import BatchedJoints2SMPL

for joints_file in glob.glob('data/*.npy'):
    joints = np.load(joints_file)
    
    converter = BatchedJoints2SMPL(
        num_frames=len(joints),
        device_id=0,  # Use GPU
        cuda=True,
    )
    
    result = converter.joint2smpl(joints)
    
    # Save results
    np.savez(
        joints_file.replace('.npy', '_smpl.npz'),
        **result
    )
```

### Custom SMPL Model Path

```python
converter = BatchedJoints2SMPL(
    num_frames=len(joints),
    device_id=-1,
    cuda=False,
)

# SMPL models are loaded from:
# ~/.motion_toolbox/datasets/body_models_clean_pkl/
# or set via converters/joints2smpl/src/config.py
```

## Troubleshooting

### Out of Memory (GPU)
- Reduce `batch_size`: Try 16 or 8
- Use CPU instead: `cuda=False, device_id=-1`

### Slow Conversion
- Enable GPU: `cuda=True, device_id=0`
- Reduce iterations if quality is sufficient: `num_smplify_iters=10`

### Poor Quality
- Increase iterations: `num_smplify_iters=50`
- Check input data quality
- Ensure LBFGS is enabled: `use_lbfgs=True`

## Technical Details

### Optimization Method
- **SMPLify3D**: Optimization-based fitting
- **Optimizer**: LBFGS (two-stage: camera→full body)
- **Loss**: L2 joint position loss + pose/shape priors
- **Initialization**: Mean SMPL pose (surprisingly effective!)

### Why This Configuration?

After extensive testing of alternatives (IK, 6D rotations, direct conversion), we found:
1. **Mean pose + LBFGS** is optimal for initialization
2. **20 iterations** balances speed/quality perfectly
3. **Batching (32)** provides 3-5x speedup
4. **LBFGS** converges faster than Adam

See investigation notes for full details on tested alternatives.

## References

- **SMPL**: Loper et al., "SMPL: A Skinned Multi-Person Linear Model"
- **SMPLify**: Bogo et al., "Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape"
- **HumanML3D**: Guo et al., "Generating Diverse and Natural 3D Human Motions from Text"

---

**Status**: Production-ready, optimized, tested
**Quality**: ~1.0cm mean error
**Speed**: ~10s per 20 frames (CPU)
