# Metric Class System Guide

This guide explains the new metric class system for motion evaluation, which provides flexible support for different implementation conventions and research standards.

## Overview

The metric class system allows you to:
- Use different implementation conventions for the same metric concept
- Easily switch between research standards (StableMotion, RMS-based, etc.)
- Compare different metric approaches side-by-side
- Maintain backward compatibility with existing code

## Architecture

### Core Components

1. **`EvalMetric`**: Abstract base class for all metrics
2. **`MetricConvention`**: Enum defining different implementation approaches
3. **`MetricRegistry`**: Central registry for discovering and creating metrics
4. **`ModernMotionEvaluator`**: New evaluator using the metric class system

### Available Metrics and Conventions

#### 1. Foot Skating Detection

- **`HEIGHT_BASED_CONTACT`**: Contact detection based on foot height above ground (default)
- **`DISPLACEMENT_BASED_CONTACT`**: Contact detection based on foot displacement (StableMotion style)
- **`VELOCITY_BASED_CONTACT`**: Contact detection based on foot velocity magnitude

#### 2. Jerk Analysis

- **`STANDARD_JERK`**: Standard 3rd derivative calculation (aligned with StableMotion)
- **`RMS_JERK`**: Root mean square jerk calculation
- **`NORMALIZED_JERK`**: Jerk normalized by motion scale/magnitude

#### 3. Ground Penetration

- **`SIMPLE_HEIGHT_CHECK`**: Basic Y-coordinate comparison with ground height
- **`MESH_INTERSECTION`**: Check for mesh-ground intersections (placeholder)
- **`CAPSULE_COLLISION`**: Use capsule approximations for body parts

#### 4. Acceleration Peaks

- **`PEAK_DETECTION`**: Identify acceleration peaks above threshold
- **`THRESHOLD_EXCEEDANCE`**: Count frames exceeding acceleration threshold
- **`STATISTICAL_OUTLIERS`**: Detect statistical outliers in acceleration distribution

## Usage Examples

### Recommended Approach: YAML Configuration Files

```python
from motion_toolbox.evaluation import MetricEvaluator

# Load from YAML configuration file (recommended)
evaluator = MetricEvaluator.from_config_file("configs/evaluation/stablemotion_compatible.yaml")

# Or use preset configurations
evaluator = MetricEvaluator.from_preset("stablemotion")

# Evaluate motion
report = evaluator.evaluate(motion)
print(f"Overall quality: {report.summary_scores['overall']:.3f}")
```

**Example YAML Configuration:**
```yaml
name: "StableMotion Compatible Evaluation"
version: "1.0"

global_params:
  velocity_threshold: 0.02

metrics:
  - name: "foot_skating"
    metric: "foot_skating"
    convention: "displacement_based_contact"
    params:
      displacement_threshold: 0.01
      
  - name: "jerk"
    metric: "jerk"
    convention: "standard_jerk"
```

### Legacy Approach: Programmatic Configuration

```python
# Define specific metric configurations (legacy approach)
metric_configs = [
    {
        "name": "foot_skating_height",
        "metric": "foot_skating",
        "convention": "height_based_contact",
        "params": {
            "velocity_threshold": 0.02,
            "contact_height_threshold": 0.05
        }
    },
    {
        "name": "jerk_standard",
        "metric": "jerk",
        "convention": "standard_jerk",
        "params": {}
    }
]

# Create evaluator
evaluator = MetricEvaluator(metric_configs)
report = evaluator.evaluate(motion)
print(f"Overall quality: {report.summary_scores['overall']:.3f}")
print(f"Height-based foot skating: {report.physical_metrics['foot_skating_height'].sequence_average:.4f}")
```

### Basic Registry Usage

```python
from motion_toolbox.evaluation import (
    MetricConvention,
    MetricRegistry,
    ModernMotionEvaluator
)

# Create evaluator with default settings
evaluator = ModernMotionEvaluator()
report = evaluator.evaluate_motion(motion)
print(f"Overall quality: {report.summary_scores['overall']:.3f}")
```

### Using Specific Conventions

```python
# Configure specific metric conventions
configs = {
    "foot_skating": {
        "convention": MetricConvention.HEIGHT_BASED_CONTACT,
        "velocity_threshold": 0.02,
        "contact_height_threshold": 0.05
    },
    "jerk": {
        "convention": MetricConvention.RMS_JERK
    },
    "ground_penetration": {
        "convention": MetricConvention.SIMPLE_HEIGHT_CHECK,
        "ground_height": 0.0
    }
}

evaluator = ModernMotionEvaluator(metric_configs=configs)
```

### StableMotion Compatibility

```python
# Use StableMotion-compatible settings (recommended approach)
stable_evaluator = MetricEvaluator.create_stablemotion_compatible()
report = stable_evaluator.evaluate(motion)

# The foot skating metric will use displacement-based contact detection
# and jerk will use standard 3rd derivative calculation

# Alternative: using ModernMotionEvaluator
stable_evaluator_old = ModernMotionEvaluator.create_stablemotion_compatible()
report = stable_evaluator_old.evaluate_motion(motion)
```

### RMS-Based Analysis

```python
# Use RMS-based metrics for better statistical properties
rms_evaluator = ModernMotionEvaluator.create_rms_based()
report = rms_evaluator.evaluate_motion(motion)
```

### Individual Metric Usage

```python
# Create and use individual metrics
foot_metric = MetricRegistry.create_metric(
    "foot_skating",
    MetricConvention.DISPLACEMENT_BASED_CONTACT,
    velocity_threshold=0.02,
    displacement_threshold=0.01
)

result = foot_metric.compute(motion)
print(f"Skating score: {result.sequence_average:.4f}")
print(f"Method used: {result.metadata['convention']}")
```

### Comparing Conventions

```python
# Easy comparison with MetricEvaluator (recommended)
comparison_evaluator = MetricEvaluator.create_comparison_evaluator()
report = comparison_evaluator.evaluate(motion)

print(f"Height-based skating: {report.physical_metrics['foot_skating_height'].sequence_average:.4f}")
print(f"Displacement-based skating: {report.physical_metrics['foot_skating_displacement'].sequence_average:.4f}")
print(f"Velocity-based skating: {report.physical_metrics['foot_skating_velocity'].sequence_average:.4f}")

# Manual comparison with separate evaluators
height_configs = [{"name": "foot_skating", "metric": "foot_skating", "convention": "height_based_contact", "params": {"velocity_threshold": 0.02, "contact_height_threshold": 0.05}}]
displacement_configs = [{"name": "foot_skating", "metric": "foot_skating", "convention": "displacement_based_contact", "params": {"velocity_threshold": 0.02, "displacement_threshold": 0.01}}]

height_evaluator = MetricEvaluator(height_configs)
displacement_evaluator = MetricEvaluator(displacement_configs)

height_report = height_evaluator.evaluate(motion)
displacement_report = displacement_evaluator.evaluate(motion)
```

## Backward Compatibility

The new system maintains full backward compatibility with the legacy API:

```python
# Legacy API still works
from motion_toolbox.evaluation import MotionEvaluator, calculate_foot_skating

# Old evaluator
legacy_evaluator = MotionEvaluator()
legacy_report = legacy_evaluator.evaluate_motion(motion)

# Old direct function calls
skating_result = calculate_foot_skating(motion)
```

## Discovering Available Options

### List Available Metrics

```python
# Get all available metrics
metrics = MetricRegistry.list_metrics()
print(f"Available metrics: {metrics}")
```

### Get Metric Information

```python
# Get detailed information about a metric
info = MetricRegistry.get_metric_info("foot_skating")
print(f"Supported conventions: {info['supported_conventions']}")
print(f"Default parameters: {info['default_parameters']}")
```

### Get Available Conventions

```python
# Get conventions for a specific metric
conventions = MetricRegistry.get_conventions("foot_skating")
print(f"Foot skating conventions: {[conv.value for conv in conventions]}")

# Get all conventions for all metrics
evaluator = ModernMotionEvaluator()
all_conventions = evaluator.get_available_conventions()
for metric, convs in all_conventions.items():
    print(f"{metric}: {convs}")
```

## Research Alignment

### StableMotion Compatibility

The system provides full compatibility with StableMotion's evaluation approach:

- **Foot Skating**: Uses displacement-based contact detection (< 1cm/frame)
- **Jerk**: Uses standard 3rd derivative calculation with identical formula
- **Same Thresholds**: Uses research-standard threshold values

```python
stable_evaluator = ModernMotionEvaluator.create_stablemotion_compatible()
```

### Custom Research Standards

You can easily create evaluators for other research standards:

```python
# Create your own research-specific configuration
my_research_config = {
    "foot_skating": {
        "convention": MetricConvention.VELOCITY_BASED_CONTACT,
        "velocity_threshold": 0.015,  # Stricter threshold
        "contact_velocity_threshold": 0.008,
    },
    "jerk": {
        "convention": MetricConvention.NORMALIZED_JERK,
        "normalization_method": "motion_range",
    },
    "acceleration": {
        "convention": MetricConvention.STATISTICAL_OUTLIERS,
        "outlier_method": "modified_z_score",
        "z_threshold": 3.5,
    }
}

my_evaluator = ModernMotionEvaluator(metric_configs=my_research_config)
```

## Benefits

1. **Flexibility**: Easy to switch between different implementations
2. **Research Alignment**: Built-in support for major research standards
3. **Extensibility**: Easy to add new metrics and conventions
4. **Reproducibility**: Explicit convention tracking in results
5. **Backward Compatibility**: Existing code continues to work
6. **Transparency**: Clear metadata about which methods were used

## Performance Improvements

### Motion Properties: velocities and accelerations

The system now includes `velocities` and `accelerations` properties on the `Motion` class that compute joint kinematics using finite differences and cache the results for efficiency:

```python
# Velocities are computed once and cached
velocities = motion.velocities  # Shape: (num_frames, num_joints, 3)
print(f"Velocity range: [{velocities.min():.4f}, {velocities.max():.4f}] m/s")

# Accelerations are also computed once and cached
accelerations = motion.accelerations  # Shape: (num_frames, num_joints, 3)
print(f"Acceleration range: [{accelerations.min():.4f}, {accelerations.max():.4f}] m/s²")

# All metrics now use these cached computations instead of recomputing
```

### Kinematic Computation Details:
- **Velocities**: Central differences for interior frames, forward/backward for edges
- **Accelerations**: Second-order finite differences `(x[t+1] - 2*x[t] + x[t-1]) / dt²`
- **Edge Handling**: Appropriate schemes for first/last frames to maintain full frame coverage

### Benefits:
- **No Recomputation**: Metrics use cached kinematics instead of computing them separately
- **Consistent Results**: All metrics use the same kinematic calculation methods
- **Better Performance**: Computations happen once per motion, not per metric
- **Full Coverage**: All frames have kinematic data (no truncation)

## Configuration Files

For detailed information about YAML configuration files, see the [Configuration System Guide](CONFIG_SYSTEM.md).

Key benefits of the configuration approach:
- **📁 File-based**: Store evaluation setups in version-controlled YAML files
- **🔧 Flexible**: Easy parameter tuning without code changes
- **🚀 Future-ready**: Support for metrics not yet implemented
- **📊 Reproducible**: Share exact evaluation setups between projects
- **🎯 Preset Support**: Built-in configurations for common use cases

## Testing

Use the test scripts to explore the system:

```bash
# Test the YAML configuration system (recommended)
uv run python motion_toolbox/scripts/test_config_system.py \
    --data-file motion_toolbox/outputs/sample_data/gvhmr/gvhmr_1.pt \
    --create-configs

# Test the clean MetricEvaluator
uv run python motion_toolbox/scripts/test_metric_evaluator.py \
    --data-file motion_toolbox/outputs/sample_data/gvhmr/gvhmr_1.pt

# Test the full metric class system
uv run python motion_toolbox/scripts/test_modern_evaluator.py \
    --data-file motion_toolbox/outputs/sample_data/gvhmr/gvhmr_1.pt
```

This will demonstrate:
- YAML configuration loading and validation
- Preset configurations (basic, stablemotion, comprehensive)
- MetricEvaluator configuration and usage
- Metric registry functionality
- Individual metric creation
- Different evaluator configurations
- Comparison between conventions
- Legacy vs modern API results
