# SMPL Fitting Configuration

This directory contains configuration files for HumanML3D to SMPL conversion.

## Available Configurations

### 1. `default.yaml` - Quality (Recommended)
**Best for:** Default use, research, high-quality visualizations

- **Optimizer:** LBFGS 150 iterations
- **Strategy:** Two-stage optimization + beta optimization
- **Quality:** 0.43cm error (Excellent - matches HumanML3D-SMPL standard)
- **Speed:** ~2x faster than original baseline
- **Use case:** Default configuration, matches quality of HumanML3D-SMPL, MLD, and MDM

### 2. `balanced.yaml` - Balanced
**Best for:** Faster processing with acceptable quality

- **Optimizer:** LBFGS 100 iterations
- **Strategy:** Two-stage optimization + beta optimization
- **Quality:** ~0.5-0.6cm error (Very Good)
- **Speed:** ~3x faster than original baseline
- **Use case:** When faster processing is needed

## Configuration Format

```yaml
optimizer:
  type: "lbfgs"          # Optimizer type: "lbfgs" or "adam"
  num_iterations: 150    # Number of optimization iterations

strategy:
  single_stage: false    # false = two-stage (better quality), true = single-stage (faster)
  use_batching: true     # Enable batched processing
  batch_size: 32         # Batch size for processing
  beta_optimization: true # Allow body shape optimization (standard approach)

loss_weights:
  joint_loss: 600.0      # Weight for joint position accuracy
  pose_prior: 1.0        # Weight for pose prior (natural poses)
  angle_prior: 15.02     # Weight for angle prior (prevent unnatural bending)
  shape_prior: 5e-3      # Weight for shape prior (realistic body shapes)

device: "cuda:0"         # Device for computation
```

## Usage

### In Python:

```python
from motion_toolbox.converters.humanml3d_to_smpl import HumanML3DToSMPLConverterFast

# Use default (quality) configuration
converter = HumanML3DToSMPLConverterFast()

# Or specify parameters manually
converter = HumanML3DToSMPLConverterFast(
    num_smplify_iters=100,     # Use 100 for balanced
    use_lbfgs=True,             # LBFGS for quality
    single_stage=False,         # Two-stage for quality
)
# Device auto-detects GPU (cuda:0) or falls back to CPU
```

### With Config File:

```python
import yaml
from motion_toolbox.converters.humanml3d_to_smpl import HumanML3DToSMPLConverterFast

# Load config
with open('configs/smpl_fitting/default.yaml') as f:
    config = yaml.safe_load(f)

# Create converter from config
converter = HumanML3DToSMPLConverterFast(
    num_smplify_iters=config['optimizer']['num_iterations'],
    use_lbfgs=config['optimizer']['type'] == 'lbfgs',
    single_stage=config['strategy']['single_stage'],
    batch_size=config['strategy']['batch_size'],
    joint_loss_weight=config['loss_weights']['joint_loss'],
    device=config['device']
)
```

## Implementation Details

### Two-Stage Optimization

The two-stage approach (standard in HumanML3D-SMPL, MLD, MDM):

1. **Stage 1:** Optimize camera translation and global orientation
   - Uses only torso joints (RHip, LHip, RShoulder, LShoulder)
   - Gets the body roughly in the right place

2. **Stage 2:** Optimize full body pose and shape
   - Optimizes all joints with pose priors
   - Enables beta (body shape) optimization on first frame

### Beta Optimization

- **Enabled by default:** Body shape parameters (betas) are optimized to fit the joint positions
- This matches the standard approach used by HumanML3D-SMPL, MLD, and MDM repositories
- Results in better fitting quality (~0.43cm vs ~1.0cm without beta optimization)

## Quality Comparison

| Configuration | Error | Time (64 frames) | Quality |
|---------------|-------|------------------|---------|
| **default.yaml** | 0.43cm | ~92s | ⭐⭐⭐⭐⭐ Excellent |
| **balanced.yaml** | ~0.5cm | ~60s | ⭐⭐⭐⭐ Very Good |

## References

- [HumanML3D-SMPL](https://github.com/XueYing126/HumanML3D-SMPL) - Standard SMPL conversion for HumanML3D
- [Motion Latent Diffusion (MLD)](https://github.com/ChenFengYe/motion-latent-diffusion) - CVPR 2023
- [Motion Diffusion Model (MDM)](https://github.com/GuyTevet/motion-diffusion-model) - Uses LBFGS 150 iterations
