# 272-Dimensional Motion Representation Converters

## Overview

This document describes the bidirectional converters between SMPL format and the 272-dimensional motion representation introduced in the paper "Go to Zero" (ICCV 2025).

The 272-dim representation enables direct conversion from joint rotations to SMPL body parameters without requiring Inverse Kinematics (IK), which significantly improves reconstruction quality.

## Format Specification

The 272-dimensional representation consists of:

| Indices | Feature | Dimensions | Description |
|---------|---------|------------|-------------|
| 0:2 | Root Linear Velocity XZ | 2 | Root velocity in local frame (no heading) |
| 2:8 | Root Angular Velocity | 6 | 6D rotation representation of heading change |
| 8:74 | Local Joint Positions | 66 | 22 joints × 3, centered at XZ origin |
| 74:140 | Local Joint Velocities | 66 | 22 joints × 3, frame-to-frame differences |
| 140:272 | Local Joint Rotations | 132 | 22 joints × 6D rotation (first 2 cols of rotation matrix) |

**Total: 272 dimensions**

### Key Properties

- **Canonicalized**: All features are in a local coordinate frame with heading removed
- **Velocity-based**: Includes both position and velocity information for better temporal modeling
- **6D Rotation**: Uses continuous 6D representation instead of discontinuous Euler angles
- **SMPL Compatible**: Designed for 22-joint SMPL body model

## Converters

### SMPLTo272Converter

Converts SMPL parameters to 272-dimensional motion representation.

#### Usage

```python
from converters.motion_converters import SMPLTo272Converter
import numpy as np

# Initialize converter
converter = SMPLTo272Converter(body_model_path="/path/to/body_models")

# Load SMPL data
smpl_data = np.load("motion.npz")

# Convert to 272D
motion_272 = converter.convert(smpl_data)

# Save result
np.save("motion_272d.npy", motion_272)
```

#### Input Format

Dictionary or .npz file with keys:
- `pose_body`: (frames, 63) - Body joint axis-angles
- `root_orient`: (frames, 3) - Root orientation axis-angle
- `trans`: (frames, 3) - Root translation
- `betas`: (10,) or (frames, 10) - Optional shape parameters
- `gender`: string - Optional gender ("neutral", "male", "female")

#### Output Format

NumPy array of shape `(frames, 272)` with dtype `float32`.

#### Algorithm

1. **Forward Kinematics**: Run SMPL model to get global joint positions
2. **Floor Alignment**: Align motion to ground plane
3. **Heading Extraction**: Compute global heading angle from root rotation
4. **Canonicalization**: Transform all features to local heading-invariant frame
5. **Feature Extraction**:
   - Root velocities in local frame
   - Heading changes as 6D rotations
   - Joint positions relative to root
   - Joint velocities from position differences
   - Joint rotations in 6D representation

### Format272ToSMPLConverter

Converts 272-dimensional motion representation to SMPL parameters.

#### Usage

```python
from converters.motion_converters import Format272ToSMPLConverter

# Initialize converter
converter = Format272ToSMPLConverter(body_model_path="/path/to/body_models")

# Load 272D data
motion_272 = np.load("motion_272d.npy")

# Convert to SMPL
smpl_params = converter.convert(motion_272)

# Use SMPL parameters
body_pose = smpl_params["body_pose"]  # torch.Tensor
global_orient = smpl_params["global_orient"]  # torch.Tensor
transl = smpl_params["transl"]  # torch.Tensor
```

#### Input Format

NumPy array of shape `(frames, 272)` or .npy file path.

#### Output Format

Dictionary with keys:
- `body_pose`: (frames, 63) - Body joint axis-angles as torch.Tensor
- `global_orient`: (frames, 3) - Root orientation as torch.Tensor
- `transl`: (frames, 3) - Root translation as torch.Tensor
- `betas`: (frames, 10) - Shape parameters as torch.Tensor
- `gender`: string - Gender identifier

#### Algorithm

This converter chains two existing converters:
1. **Format272ToMotionConverter**: 272D → Motion class
2. **MotionToSMPLConverter**: Motion → SMPL parameters

This approach ensures consistency and reuses well-tested code.

## Converter Registry

Both converters are registered in the factory function:

```python
from converters.motion_converters import get_motion_converter

# Get SMPL to 272D converter
converter = get_motion_converter("smpl2format272", body_model_path="/path/to/models")

# Get 272D to SMPL converter
converter = get_motion_converter("format272tosmpl", body_model_path="/path/to/models")

# List all available converters
from converters.motion_converters import list_converters
print(list_converters())
```

## Demos

### Conversion Demo

Demonstrates SMPL ↔ 272D conversion with error analysis:

```bash
python demos/smpl_272_conversion_demo.py
```

Features:
- SMPL → 272D conversion
- 272D → SMPL round-trip conversion
- Error analysis and reconstruction quality metrics
- GLB export and HTML visualization

### Visualization Demo

Visualizes 272D motion data:

```bash
python demos/format_272_visualization_demo.py
```

Features:
- Load and visualize 272D motion files
- Generate 272D from SMPL if no samples available
- Export to GLB format
- Create interactive HTML viewers
- Grid view for multiple motions

## Testing

Comprehensive unit tests are provided in `tests/test_272_converters.py`:

```bash
# Run all tests
pytest tests/test_272_converters.py -v

# Run specific test class
pytest tests/test_272_converters.py::TestSMPLTo272Converter -v

# Run with coverage
pytest tests/test_272_converters.py --cov=converters --cov-report=html
```

Test coverage includes:
- Basic conversion functionality
- Format structure validation
- Round-trip consistency
- Integration with Motion class
- Edge cases (single frame, invalid input, etc.)

## Performance

### Conversion Speed

- **SMPL → 272D**: ~0.1-0.5 seconds per 100 frames (depends on SMPL forward kinematics)
- **272D → SMPL**: ~0.05-0.1 seconds per 100 frames (faster, uses chain conversion)

### Memory Usage

- **272D Format**: ~1 KB per frame (272 × 4 bytes)
- **SMPL Format**: ~300 bytes per frame (compact representation)

For a 10-second motion at 30 fps (300 frames):
- 272D: ~300 KB
- SMPL: ~90 KB

## Error Analysis

### Round-trip Conversion Accuracy

Typical errors for SMPL → 272D → SMPL:

| Metric | Expected Range | Excellent | Good | Acceptable |
|--------|---------------|-----------|------|------------|
| Body Pose Error | radians | < 0.01 | < 0.1 | < 0.5 |
| Root Orientation Error | radians | < 0.01 | < 0.1 | < 0.5 |
| Translation Error | meters | < 0.001 | < 0.01 | < 0.1 |

Errors are primarily due to:
1. Floor alignment differences
2. Numerical precision in rotation conversions
3. Canonicalization (heading removal/restoration)
4. SMPL forward kinematics discretization

### Quality Improvement Tips

For best conversion quality:
1. Use higher precision (float64) if memory allows
2. Ensure SMPL body models are properly configured
3. Use consistent body shape parameters (betas) when possible
4. Validate input data ranges before conversion

## Advantages Over IK-based Methods

The 272-dim representation offers several advantages:

1. **No IK Required**: Direct rotation-to-parameters mapping
2. **Better Temporal Consistency**: Includes velocity information
3. **Numerically Stable**: 6D rotation avoids gimbal lock
4. **Canonicalized**: Invariant to global orientation
5. **Compact**: More efficient than joint positions (66 vs 264 dims for 22 joints × 3 × 4 frames)

## References

- **Paper**: "Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data" (ICCV 2025 Highlight)
- **Reference Implementation**: [Li-xingXiao/272-dim-Motion-Representation](https://github.com/Li-xingXiao/272-dim-Motion-Representation)
- **Dataset**: [HumanML3D-272D on HuggingFace](https://huggingface.co/datasets/lxxiao/272-dim-HumanML3D)

## Troubleshooting

### Common Issues

**Issue**: `ValueError: body_model_path is required`
- **Solution**: Set `SMPL_BODY_MODELS_PATH` in `.env` file or pass `body_model_path` argument

**Issue**: `ModuleNotFoundError: No module named 'smplx'`
- **Solution**: Install SMPL-X: `pip install smplx`

**Issue**: `FileNotFoundError: SMPL data file not found`
- **Solution**: Check file path and ensure .npz format for SMPL data

**Issue**: Large round-trip errors
- **Solution**: Verify input data quality, check body shape consistency, ensure proper SMPL model configuration

### Getting Help

- Check existing demos for usage examples
- Run unit tests to verify installation
- Review error messages for specific issues
- Consult the main README for environment setup

## Future Enhancements

Potential improvements:
1. Support for SMPL-X (more joints, hand/face parameters)
2. Batch processing for large datasets
3. GPU acceleration for forward kinematics
4. Integration with motion generation models
5. Normalization/denormalization utilities using dataset statistics
