# Motion Evaluation Guide

This guide explains how to use the motion evaluation system to assess the quality of motion data using research-validated physical plausibility metrics.

## Quick Start

### Basic Evaluation with Config File

```bash
# Evaluate motion using a predefined configuration
cd /path/to/motion-toolbox
uv run python scripts/gvhmr_motion_evaluation.py \
    --data-file sample_data/gvhmr/gvhmr_1.pt \
    --output-dir outputs/motion_evaluation \
    --config configs/evaluation/final.yaml

# Open the generated HTML report
open outputs/motion_evaluation/gvhmr_1_evaluation_report.html
```

### Programmatic Usage

```python
from motion_toolbox.evaluation import MotionEvaluator
from motion_toolbox.core import Motion

# Load evaluator from config
evaluator = MotionEvaluator.from_config_file("configs/evaluation/final.yaml")

# Evaluate motion
report = evaluator.evaluate(motion)

# Access results
print(f"Physical plausibility: {report.summary_scores['physical']:.3f}")
print(f"Foot skating ratio: {report.physical_metrics['foot_skating_hoidini_joint'].ratio:.1%}")
print(f"Jerk (smoothness): {report.physical_metrics['jerk_mean'].sequence_average:.2f}")
```

## Available Metrics

The evaluation system implements research-validated metrics from multiple motion quality papers:

### 1. **Foot Skating Detection**

Detects when feet slide along the ground instead of lifting properly during locomotion.

**Conventions:**
- **HOIDINI (Joint)**: Uses foot joint positions with displacement-based contact detection
  - Contact threshold: 10mm (0.01m)
  - Skating threshold: 5mm/frame (0.005m) displacement
  - Reference: "HOIDINI" paper
  
- **HOIDINI (Mesh)**: Uses SMPL foot mesh vertices for more accurate detection
  - Uses 719 verified foot vertices per foot
  - Same thresholds as joint-based version
  - More sensitive to subtle skating artifacts

**Ratio Metric:** Percentage of contact frames with skating detected
- **Good:** < 40% (joint) or < 60% (mesh)
- **Moderate:** 40-60% (joint) or 60-80% (mesh)
- **Poor:** > 60% (joint) or > 80% (mesh)

### 2. **Ground Penetration**

Detects when body parts go below the ground plane, violating physical constraints.

**Conventions:**
- **HOIDINI (Joint)**: Joint-based penetration detection
  - Floor height: 0.0m
  - Contact threshold: 50mm (0.05m)
  - Penetration threshold: -10mm (-0.01m)
  
- **HOIDINI (Mesh)**: Mesh-based penetration using SMPL vertices
  - Same thresholds, more comprehensive coverage
  - Detects subtle mesh penetrations

**Ratio Metric:** Percentage of frames with penetration detected
- **Excellent:** < 15%
- **Moderate:** 15-30%
- **Poor:** > 30%

### 3. **Floating Detection**

Detects when feet hover above the ground unnaturally during supposed contact.

**Convention:**
- **CLoSD**: Based on the CLoSD paper methodology
  - Tolerance: 5mm (0.005m) for floating calculation
  - Contact threshold: 50mm (0.05m)
  - Floating threshold: 50mm (0.05m)

**Ratio Metric:** Percentage of frames with floating detected
- **Excellent:** < 5%
- **Moderate:** 5-15%
- **Poor:** > 15%

### 4. **Motion Jerk (Smoothness)**

Measures motion smoothness through jerk analysis (third derivative of position).

**Conventions:**
- **Mean Jerk**: Finite difference calculation
  - Reference: StableMotion, GVHMR papers
  - Formula: `jerk = (x[t+3] - 3*x[t+2] + 3*x[t+1] - x[t]) / dt³`
  - Scaling: Mean magnitude divided by 10.0
  - Threshold: 10.0 (after scaling)
  
- **Gradient Jerk**: np.gradient-based calculation
  - Same formulation as mean jerk
  - Uses np.gradient for smoother derivatives
  - More robust to noise
  - Threshold: 10.0 (same as mean jerk)

**Ratio Metric:** Percentage of frames with high jerk detected
- **Good:** < 40%
- **Moderate:** 40-60%
- **Poor:** > 60%

### 5. **Dynamic Score**

Measures motion activity and naturalness through velocity-acceleration analysis.

**Convention:**
- **Hybrid Velocity-Acceleration**: Combined analysis
  - Velocity threshold: 20mm/s (0.02m/s)
  - Acceleration threshold: 0.5 m/s²
  - Detects static, low-activity, and dynamic motion

**Ratio Metric:** Percentage of frames with dynamic motion (higher is better)
- **Poor (Static):** < 10%
- **Fair (Low Activity):** 10-50%
- **Good (Active):** > 50%

## Output Files

The evaluation script generates comprehensive output files:

### 1. **Interactive HTML Report** (`*_evaluation_report.html`)

The main evaluation report includes:

- **Quality Metrics Summary**: Color-coded cards showing key metrics with ratio indicators
  - Green (🟢): Excellent/Good performance
  - Yellow (🟡): Moderate concerns, review needed
  - Red (🔴): Poor performance, high concern
  
- **Detailed Ratio Analysis**: Per-metric breakdown with configurable thresholds
  - Shows percentage of frames with issues detected
  - Assessment based on metric-specific thresholds
  
- **Interactive Plotly Visualizations**:
  - Per-frame metric plots with hover details
  - Detected issue frames highlighted
  - Threshold lines showing detection limits
  - Per-joint analysis for applicable metrics
  
- **3D Motion Animation**: Embedded interactive SMPL mesh viewer
  - Playback controls with frame scrubbing
  - Camera controls (rotate, zoom, pan)
  
- **Configuration Details**: Shows all evaluation parameters used

### 2. **Evaluation Data** (`*_evaluation_data.json`)
- **Raw metric values**: Per-frame and per-joint data
- **Ratio metrics**: Percentage of frames with detected issues
- **Motion metadata**: Frame count, duration, joint names
- **Configuration**: Complete evaluation parameters

## Understanding Ratio-Based Assessment

The evaluation system uses **ratio metrics** to assess motion quality. Ratios represent the percentage of frames where an issue is detected.

### Configurable Thresholds

Ratio thresholds are defined in config files (`ratio_thresholds` section):

```yaml
ratio_thresholds:
  foot_skating_joint:
    good: 0.40      # < 40% is good
    moderate: 0.60  # 40-60% is moderate, > 60% is high concern
  jerk:
    good: 0.40      # < 40% is good
    moderate: 0.60  # 40-60% is moderate, > 60% is high concern
  penetration:
    excellent: 0.15  # < 15% is excellent
    moderate: 0.30   # 15-30% is moderate, > 30% is high concern
```

### Color Coding

- **🟢 Green (Good/Excellent)**: Ratio below "good" or "excellent" threshold
- **🟡 Yellow (Moderate)**: Ratio between thresholds, review recommended
- **🔴 Red (Poor/High Concern)**: Ratio above "moderate" threshold

### Example Assessment

```
Foot Skating (Joint): 35% 🟢 Good
- 35% of contact frames show skating
- Below 40% threshold → Good result

Jerk (Mean): 55% 🟡 Moderate  
- 55% of frames have high jerk
- Between 40-60% → Review needed

Ground Penetration: 8% 🟢 Excellent
- 8% of frames have penetration
- Below 15% threshold → Excellent result
```

## Configuration Options

### Script Arguments

```bash
# Run from the repository root
cd /path/to/motion-toolbox
uv run python scripts/gvhmr_motion_evaluation.py [OPTIONS]

Options:
  --data-file PATH          Path to motion .pt file
  --output-dir DIR          Output directory for results
  --config PATH             Path to evaluation config YAML file
  --max-frames INT          Maximum frames to process (default: all frames)
  --body-models-path DIR    Path to SMPL body models
```

### Available Evaluation Configs

The `configs/evaluation/` directory contains pre-configured evaluation setups:

- **`final.yaml`**: Production-ready metrics (recommended)
  - Foot skating (HOIDINI joint & mesh)
  - Ground penetration (HOIDINI joint & mesh)
  - Floating detection (CLoSD)
  - Jerk analysis (mean & gradient)
  - Dynamic score
  - Configurable ratio thresholds
  
- **`comprehensive.yaml`**: All available metrics for detailed analysis
  - Includes all conventions and variants
  - Multiple jerk formulations
  - Various contact detection methods
  
- **`basic.yaml`**: Minimal metric set for quick evaluation
  - Essential metrics only
  - Faster processing
  
- **`hoidini_style.yaml`**: HOIDINI paper reproduction
  - Exact HOIDINI paper conventions
  - Original thresholds
  
- **`stablemotion_compatible.yaml`**: StableMotion alignment
  - Metrics matching StableMotion paper

### Custom Configuration

Create custom YAML configs to define your own evaluation pipeline:

```yaml
name: "Custom Evaluation"
version: "1.0"

# Global parameters
global_params:
  velocity_threshold: 0.02
  ground_height: 0.0

# Ratio assessment thresholds
ratio_thresholds:
  foot_skating_joint:
    good: 0.40
    moderate: 0.60
  jerk:
    good: 0.40
    moderate: 0.60

# Metrics to compute
metrics:
  - name: "foot_skating_hoidini_joint"
    metric: "foot_skating"
    convention: "hoidini_foot_skating"
    params:
      contact_threshold: 0.01
      skating_threshold: 0.005
      
  - name: "jerk_mean"
    metric: "jerk"
    convention: "mean_jerk"
    params:
      jerk_threshold: 10.0
```

## Integration with Other Formats

### Using with Motion Objects

```python
from motion_toolbox.evaluation import MotionEvaluator
from motion_toolbox.core import Motion

# Load evaluator from config
evaluator = MotionEvaluator.from_config_file("configs/evaluation/final.yaml")

# Evaluate any Motion object
report = evaluator.evaluate(motion)

# Access results
print(f"Foot skating ratio: {report.physical_metrics['foot_skating_hoidini_joint'].ratio:.1%}")
print(f"Jerk mean: {report.physical_metrics['jerk_mean'].sequence_average:.2f}")
print(f"Dynamic score: {report.physical_metrics['dynamic_score'].ratio:.1%}")
```

### Accessing Detailed Results

```python
# Get per-frame data
metric_result = report.physical_metrics['foot_skating_hoidini_joint']
per_frame_values = metric_result.per_frame  # Array of values per frame
per_joint_values = metric_result.per_joint  # Array of values per joint
sequence_avg = metric_result.sequence_average  # Overall average
ratio = metric_result.ratio  # Percentage of frames with issues

# Get metadata
metadata = metric_result.metadata
print(f"Frames with skating: {metadata['frames_with_skating']}")
print(f"Total contact frames: {metadata['total_contact_frames']}")
```

### Comparing Multiple Motions

```python
# Evaluate different motion variants
motions = {
    'original': original_motion,
    'filtered': filtered_motion,
    'refined': refined_motion
}

# Load evaluator once
evaluator = MotionEvaluator.from_config_file("configs/evaluation/final.yaml")

# Evaluate all motions
results = {}
for name, motion in motions.items():
    report = evaluator.evaluate(motion)
    results[name] = {
        'foot_skating': report.physical_metrics['foot_skating_hoidini_joint'].ratio,
        'jerk': report.physical_metrics['jerk_mean'].ratio,
        'dynamic': report.physical_metrics['dynamic_score'].ratio,
    }

# Compare results
import pandas as pd
df = pd.DataFrame(results).T
print(df)
```

## Troubleshooting

### Common Issues

#### 1. **Missing SMPL Body Models**
```
Error: Body model directory not found
```
**Solution**: Download SMPL models and specify path with `--body-models-path`

#### 2. **Config File Not Found**
```
Error: Config file not found: configs/evaluation/final.yaml
```
**Solution**: Ensure you're running from the correct directory or provide absolute path

#### 3. **Missing Metric in Report**
```
KeyError: 'foot_skating_hoidini_joint'
```
**Solution**: Check that the metric is defined in your config file. Metric names must match exactly.

#### 4. **High Memory Usage**
```
Memory error during evaluation
```
**Solution**: Use `--max-frames` to process shorter sequences for testing

### Performance Tips

1. **Testing with subsets**: Use `--max-frames 30` for quick testing and development
2. **Config selection**: Use `basic.yaml` for faster evaluation during development
3. **Batch processing**: Evaluate multiple files in parallel
4. **Caching**: Motion positions are cached for repeated evaluations
5. **Full evaluation**: Use `final.yaml` or `comprehensive.yaml` for complete assessment

## Advanced Usage

### Custom Metrics

To implement custom evaluation metrics:

1. **Create metric class** in `evaluation/metrics/`:
```python
from evaluation.metrics.base import BaseMetric, MetricResult
import numpy as np

class MyCustomMetric(BaseMetric):
    """Custom metric implementation."""
    
    CONVENTIONS = {
        "my_convention": {
            "name": "My Convention",
            "description": "Custom metric convention",
            "default_params": {
                "threshold": 1.0,
            }
        }
    }
    
    def compute(self, motion: Motion, convention: str) -> MetricResult:
        # Your metric calculation
        per_frame = np.zeros(motion.num_frames)
        per_joint = np.zeros(motion.num_joints)
        
        # Calculate ratio
        frames_with_issue = np.sum(per_frame > self.params['threshold'])
        ratio = frames_with_issue / motion.num_frames
        
        return MetricResult(
            per_frame=per_frame,
            per_joint=per_joint,
            frame_average=per_frame,
            sequence_average=float(np.mean(per_frame)),
            metadata={"threshold": self.params['threshold']},
            ratio=float(ratio)
        )
```

2. **Register metric** in `evaluation/metrics/__init__.py`:
```python
from .my_metric import MyCustomMetric

__all__ = [..., "MyCustomMetric"]
```

3. **Add to config**:
```yaml
metrics:
  - name: "my_custom_metric"
    metric: "my_custom"
    convention: "my_convention"
    params:
      threshold: 1.0
```

### Batch Evaluation

For evaluating multiple files:

```bash
# Bash script for batch evaluation (run from the repository root)
cd /path/to/motion-toolbox
for file in sample_data/gvhmr/*.pt; do
    echo "Evaluating $(basename "$file")..."
    uv run python scripts/gvhmr_motion_evaluation.py \
        --data-file "$file" \
        --output-dir "outputs/batch_evaluation/$(basename "$file" .pt)" \
        --config configs/evaluation/final.yaml
done
```

### Python Batch Evaluation

```python
from pathlib import Path
from motion_toolbox.evaluation import MotionEvaluator
from motion_toolbox.converters import get_motion_converter

# Setup
data_dir = Path("outputs/sample_data/gvhmr")
output_dir = Path("outputs/batch_evaluation")
evaluator = MotionEvaluator.from_config_file("configs/evaluation/final.yaml")
converter = get_motion_converter('gvhmr2motion')

# Batch evaluate
for data_file in data_dir.glob("*.pt"):
    print(f"Evaluating {data_file.name}...")
    
    # Convert to Motion
    motion = converter.convert(str(data_file))
    
    # Evaluate
    report = evaluator.evaluate(motion)
    
    # Save results
    output_path = output_dir / data_file.stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate report (using gvhmr_motion_evaluation.py for HTML generation)
    print(f"  Foot skating: {report.physical_metrics['foot_skating_hoidini_joint'].ratio:.1%}")
    print(f"  Jerk: {report.physical_metrics['jerk_mean'].ratio:.1%}")
```

## References and Research

The evaluation system implements metrics from published research:

### Foot Skating and Ground Constraints
- **HOIDINI**: Contact-based skating and penetration detection
- **CLoSD**: Floating detection methodology
- **GMD**: Alternative ground contact metrics

### Motion Smoothness
- **StableMotion**: Jerk-based smoothness analysis with /10.0 scaling
- **GVHMR**: Motion generation quality metrics
- **FlowMDM**: Flow-based motion diffusion quality assessment

### Dynamic Analysis
- **Hybrid Velocity-Acceleration**: Motion activity and naturalness scoring
- Combines velocity and acceleration thresholds for comprehensive analysis

### Configuration and Thresholds
All thresholds and parameters are derived from published papers or validated through empirical analysis. See individual metric documentation and config files for specific references.
