# ALIGN Feature Representation Guide

## Overview

The ALIGN representation (232D) is a rotation-invariant motion encoding used by [StableMotion](https://arxiv.org/abs/2505.03154) for motion cleanup tasks. It provides a canonically aligned, continuous representation optimized for machine learning applications.

## What is ALIGN?

ALIGN stands for **A**ligned **L**ocal-**I**nvariant **G**lobal **N**etwork features. It's a 232-dimensional representation that includes:

- **Canonical alignment**: Rotation-invariant (first frame aligned to canonical direction)
- **6D rotations**: Continuous rotation representation for better neural network training
- **Explicit foot dynamics**: Global foot positions and velocities for contact analysis
- **Motion velocities**: Trajectory and foot velocities for temporal consistency

### Feature Breakdown (232 dimensions)

```
[1]    root_grav_axis       # Root vertical position (Z-axis)
[2]    trajectory           # XY trajectory in canonical frame
[2]    rotZ_2d              # Heading direction (cos, sin)
[132]  poses_local          # Body poses in 6D (22 joints × 6)
[69]   joints_local         # Joint positions (23 joints × 3)
[12]   foot_global          # Global foot positions (4 joints × 3)
[12]   vel_foot_global      # Foot velocities (4 joints × 3)
[2]    vel_traj             # Trajectory velocity
```

## Basic Usage

### Installation

ALIGN support is built into motion_toolbox. No additional installation required.

```python
from motion_toolbox import Motion
from motion_toolbox.converters import ALIGNConverter
```

### Convert Motion to ALIGN Features

```python
# Load or create a Motion object
motion = Motion.from_file("motion.npy")

# Create converter
converter = ALIGNConverter()

# Convert to ALIGN features
features = converter.to_align(motion)  # [T, 232]
print(f"ALIGN features shape: {features.shape}")
```

### Convert ALIGN Features to Motion

```python
# Convert ALIGN features back to Motion
clean_features = your_cleanup_function(features)
clean_motion = converter.from_align(clean_features, fps=20.0)

# Save or visualize
clean_motion.save("cleaned_motion.npy")
```

### Auto-Detection with `convert()`

The converter can automatically detect the direction:

```python
converter = ALIGNConverter()

# Motion → ALIGN (auto-detected)
features = converter.convert(motion)

# ALIGN → Motion (auto-detected)
motion_reconstructed = converter.convert(features)
```

## Feature Extraction

Extract specific components from ALIGN features for analysis:

### Root and Trajectory

```python
# Extract root height (vertical position)
root_height = converter.extract_root_height(features)  # [T]

# Extract XY trajectory (canonical frame)
trajectory = converter.extract_trajectory(features)  # [T, 2]

# Extract heading direction
heading = converter.extract_heading(features)  # [T, 2] (cos, sin)
```

### Body Poses and Joints

```python
# Extract body poses (6D representation)
poses_6d = converter.extract_poses_6d(features)  # [T, 22, 6]

# Extract local joint positions (pelvis-relative)
joints_local = converter.extract_joints_local(features)  # [T, 23, 3]
```

### Foot Dynamics

```python
# Extract global foot positions
foot_pos = converter.extract_foot_positions(features)  # [T, 4, 3]

# Extract foot velocities
foot_vel = converter.extract_foot_velocities(features)  # [T, 4, 3]

# Extract trajectory velocity
traj_vel = converter.extract_trajectory_velocity(features)  # [T, 2]
```

## Analysis Utilities

### Foot Contact Detection

Detect when feet are in contact with the ground:

```python
# Detect foot contacts (boolean array)
contacts = converter.compute_foot_contacts(features, threshold=0.05)  # [T, 4]

# Print contact statistics
for i, foot_name in enumerate(['L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot']):
    contact_frames = contacts[:, i].sum()
    print(f"{foot_name}: {contact_frames}/{len(contacts)} frames")
```

### Foot Sliding Detection

Detect foot skating artifacts (feet moving while in contact):

```python
# Detect sliding (feet in contact but moving)
sliding = converter.detect_sliding(
    features,
    contact_threshold=0.05,
    slide_threshold=0.02
)  # [T, 4]

# Count sliding frames
total_sliding = sliding.sum()
print(f"Sliding detected in {total_sliding} foot-frames")
```

## Visualization

### Skeleton Visualization

Visualize ALIGN features as skeleton animation:

```python
from motion_toolbox.visualization import Visualizer

visualizer = Visualizer()

# ALIGN features are automatically detected and converted
visualizer.visualize(
    features,
    output_path="align_skeleton.html",
    visualization_type="skeleton"
)
```

### SMPL Mesh Visualization

Generate SMPL mesh animation from ALIGN features:

```python
# Convert to Motion first
motion = converter.from_align(features, fps=20.0)

# Visualize as SMPL mesh
visualizer.visualize(
    motion,
    output_path="align_mesh.html",
    visualization_type="mesh",
    body_model_path="/path/to/smpl_models"
)
```

## Integration with StableMotion

ALIGN is designed to be compatible with [StableMotion](https://github.com/korrawe/StableMotion)'s motion cleanup pipeline:

```python
import torch
from motion_toolbox.converters import ALIGNConverter

# Load your trained StableMotion model
stablemotion_model = load_stablemotion_model("checkpoint.pth")

# Convert motion to ALIGN
converter = ALIGNConverter()
features = converter.to_align(motion)

# Run through StableMotion
features_tensor = torch.from_numpy(features).float()
clean_features_tensor = stablemotion_model(features_tensor)
clean_features = clean_features_tensor.cpu().numpy()

# Convert back to motion
clean_motion = converter.from_align(clean_features, fps=20.0)
```

## Advanced Usage

### Custom Body Model Path

Specify custom SMPL model location:

```python
converter = ALIGNConverter(body_model_path="/path/to/smpl_models")
motion = converter.from_align(features, fps=20.0)
```

### Batch Processing

Process multiple motions efficiently:

```python
converter = ALIGNConverter()

# Convert multiple motions
all_features = []
for motion in motion_list:
    features = converter.to_align(motion)
    all_features.append(features)

# Stack for batch processing
features_batch = np.stack(all_features, axis=0)  # [B, T, 232]
```

### Feature Manipulation

Modify specific components before reconstruction:

```python
# Extract and modify trajectory
trajectory = converter.extract_trajectory(features)
trajectory_smoothed = smooth_trajectory(trajectory)

# Reconstruct features with modified trajectory
# (Note: This requires manual feature packing - see implementation)
features_modified = features.copy()
features_modified[:, 1:3] = trajectory_smoothed

# Convert back
motion_modified = converter.from_align(features_modified, fps=20.0)
```

## Technical Details

### Coordinate System

- **Up axis**: Z-up (gravity along Z-axis)
- **Forward**: X-axis (after canonical alignment)
- **Right**: Y-axis

### Skeleton

ALIGN uses the SMPL skeleton:
- 24 total joints (22 body joints + 2 hands)
- Only body joints (22) are used in the representation
- Foot joints: Left Ankle (7), Right Ankle (10), Left Foot (8), Right Foot (11)

### Canonical Alignment

The first frame is aligned to a canonical direction:
- Root trajectory starts at origin (0, 0)
- Heading rotation is relative to first frame
- Removes global rotation ambiguity

### 6D Rotation Representation

Body poses use 6D continuous rotation representation (Zhou et al., CVPR 2019):
- More stable for neural network training than Euler angles or quaternions
- Guarantees valid rotation matrices via Gram-Schmidt orthogonalization

## Common Issues

### Floor Penetration

If reconstructed motion has floor penetration:

```python
# ALIGN already handles floor alignment
# But you can adjust if needed
motion = converter.from_align(features, fps=20.0)
# Positions are already floor-aligned in ALIGN
```

### Coordinate System Mismatch

If your motion uses Y-up instead of Z-up:

```python
# Convert Y-up motion to Z-up before ALIGN conversion
# (Most motion formats use Y-up, ALIGN uses Z-up)
motion_z_up = motion.change_up_axis('z')  # If this method exists
features = converter.to_align(motion_z_up)
```

### Foot Skating

ALIGN explicitly encodes foot velocities to help detect/fix skating:

```python
# Detect sliding
sliding = converter.detect_sliding(features)

# Use sliding information for cleanup
# (Integration with foot skating cleanup algorithms)
```

## Reference

- **Paper**: [StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data](https://arxiv.org/abs/2505.03154)
- **Code**: [StableMotion GitHub](https://github.com/korrawe/StableMotion)
- **6D Rotation**: [On the Continuity of Rotation Representations in Neural Networks (CVPR 2019)](https://arxiv.org/abs/1812.07035)

## See Also

- [272D Format Reference](272_FORMAT_REFERENCE.md) - Similar multi-dimensional representation
- [STMC Converter Guide](STMC_CONVERTER_GUIDE.md) - Another ML-optimized format
- [Visualization Guide](VISUALIZATION_GUIDE.md) - How to visualize motion data
- [SMPL Setup Guide](MHR_SETUP_GUIDE.md) - Setting up SMPL body models
