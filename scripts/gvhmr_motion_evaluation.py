#!/usr/bin/env python3
"""GVHMR Motion Evaluation and Visualization Script.

This script loads GVHMR data, converts it to Motion class, evaluates motion quality
using physical metrics, and creates an interactive HTML report with:
- Motion visualization (skeleton animation)
- Real-time metric plots (per frame)
- Summary tables with scores
- Recommendations for improvement

Usage:
    python scripts/gvhmr_motion_evaluation.py [--data-file PATH] [--output-dir DIR]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add imports for enhanced visualization
import colorlog
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import from the motion_toolbox package
from motion_toolbox.converters import GVHMRToMotionConverter
from motion_toolbox.core.motion import Motion
from motion_toolbox.evaluation import EvaluationReport, MotionEvaluator
from motion_toolbox.evaluation.base_metric import MetricResult
from motion_toolbox.visualization import GLB_AVAILABLE, GLBExporter, MotionVisualizer

# ============================================================================
# RATIO ASSESSMENT THRESHOLDS (FALLBACK DEFAULTS)
# ============================================================================
# These thresholds determine how ratio values are assessed in the report.
# Ratios represent the percentage of frames with detected issues.
#
# NOTE: These are FALLBACK defaults. The actual thresholds should be defined
# in the evaluation config YAML files (comprehensive.yaml, final.yaml, etc.)
# under the "ratio_thresholds" section. These hardcoded values are only used
# if the config file doesn't provide ratio_thresholds.

RATIO_THRESHOLDS = {
    # Foot skating thresholds
    "foot_skating_joint": {
        "good": 0.40,  # < 40% is good
        "moderate": 0.60,  # 40-60% is moderate
        # > 60% is high concern
    },
    "foot_skating_mesh": {
        "good": 0.60,  # < 60% is good
        "moderate": 0.80,  # 60-80% is moderate
        # > 80% is high concern
    },
    # Jerk thresholds (all jerk metrics)
    "jerk": {
        "good": 0.40,  # < 40% is good
        "moderate": 0.60,  # 40-60% is moderate
        # > 60% is high concern
    },
    # Ground penetration thresholds
    "penetration": {
        "excellent": 0.15,  # < 15% is excellent
        "moderate": 0.30,  # 15-30% is moderate
        # > 30% is high concern
    },
    # Floating thresholds
    "floating": {
        "excellent": 0.05,  # < 5% is excellent
        "moderate": 0.15,  # 5-15% is moderate
        # > 15% is high concern
    },
    # Dynamic score thresholds (inverse: higher is better)
    "dynamic": {
        "static": 0.10,  # < 10% is static
        "low_activity": 0.50,  # 10-50% is low activity
        # > 50% is active (good)
    },
    # Root rotation thresholds
    "root_rotation": {
        "good": 0.10,  # < 10% is good (stable motion)
        "moderate": 0.30,  # 10-30% is moderate
        # > 30% is high concern (jittery/unstable)
    },
}


def setup_logging() -> logging.Logger:
    """Set up colored logging."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s:%(lineno)d%(reset)s %(message)s"
        )
    )

    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def load_and_convert_gvhmr(
    data_path: str, max_frames: int | None = None, body_model_path: str | None = None
) -> Motion | None:
    """Load GVHMR file and convert to Motion.

    Args:
        data_path: Path to GVHMR .pt file
        max_frames: Maximum number of frames to load (None for all)
        body_model_path: Path to body models for SMPL forward kinematics

    Returns:
        Motion object or None if loading failed
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading GVHMR file: {Path(data_path).name}")

    try:
        # Load the data
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        logger.info(f"✓ Loaded data with keys: {list(data.keys())}")

        # Get converter
        converter = GVHMRToMotionConverter(max_frames=max_frames, body_model_path=body_model_path)

        # Convert to Motion
        motion = converter.convert(data)

        logger.info(f"✓ Created Motion: {motion.num_frames} frames, {motion.num_joints} joints")
        logger.info(f"   Duration: {motion.duration:.2f}s at {motion.framerate}fps")
        logger.info(
            f"   Joint names: {motion.skeleton.joint_names[:10]}{'...' if motion.skeleton.num_joints > 10 else ''}"
        )

        return motion

    except Exception as e:
        logger.error(f"Failed to load/convert GVHMR data: {e}")
        return None


def evaluate_motion_metrics(motion: Motion) -> tuple[EvaluationReport, MotionEvaluator]:
    """Evaluate motion using physical metrics.

    Args:
        motion: Motion object to evaluate

    Returns:
        Tuple of (EvaluationReport, MotionEvaluator) - report with metrics and evaluator instance
    """
    logger = logging.getLogger(__name__)
    logger.info("🔍 Evaluating motion quality metrics...")

    # Use comprehensive evaluation configuration for detailed analysis (including dynamic_score)
    evaluator = MotionEvaluator.from_config_file("configs/evaluation/comprehensive.yaml")

    # Evaluate all metrics
    report = evaluator.evaluate(motion)

    logger.info("✓ Evaluation complete!")
    logger.info(f"   Physical plausibility: {report.summary_scores['physical']:.3f}")

    # Log individual metrics
    for metric_name, metric_result in report.physical_metrics.items():
        score = report.summary_scores.get(metric_name, 0.0)
        logger.info(f"   {metric_name}: {score:.3f} (avg: {metric_result.sequence_average:.4f})")

    return report, evaluator


def create_enhanced_foot_skating_plot(
    metric_name: str, metric_result: MetricResult, frame_indices: np.ndarray, color: str
) -> go.Figure:
    """Create enhanced foot skating plot with contact detection and distance-to-ground visualization."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create 2-row layout for skating with contact overlay and distance-to-ground
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{metric_name.replace('_', ' ').title()} & Contact Detection",
            "Distance to Ground (with thresholds)",
        ),
        vertical_spacing=0.15,  # Increased spacing for 2-row layout
        row_heights=[0.6, 0.4],  # Give more space to main skating plot
    )

    # Determine units and extract data
    skating_unit = "mm/s" if "gmd" in metric_name else "mm"
    max_skating = max(np.max(metric_result.frame_average), 1.0) if len(metric_result.frame_average) > 0 else 1.0

    # Add main skating data
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=metric_result.frame_average,
            mode="lines+markers",
            name=f"Skating ({skating_unit})",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Extract contact and distance information from metric metadata
    contact_frames = []
    skating_frames = []
    foot_heights = []
    contact_threshold = 0.01  # Default 10mm

    if hasattr(metric_result, "metadata") and metric_result.metadata:
        # Get available metadata
        metadata = metric_result.metadata

        # Try different contact data keys
        for key in ["contact_frames_bool", "contact_frames", "is_contact"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                contact_frames = metadata[key]
                break

        for key in ["skating_frames_bool", "skating_frames", "is_skating"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                skating_frames = metadata[key]
                break

        for key in ["foot_heights", "heights", "distance_to_ground"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                foot_heights = metadata[key]
                break

        # Get threshold
        for key in ["contact_threshold", "margin", "thresh_height"]:
            if key in metadata:
                contact_threshold = metadata[key]
                break

    # Generate synthetic contact/height data if not available
    if not contact_frames and len(frame_indices) > 0:
        # Simple heuristic: detect periods of low skating as potential contact
        contact_frames = (metric_result.frame_average < (max_skating * 0.3)).tolist()

    if not foot_heights and len(frame_indices) > 0:
        # Generate synthetic height data based on contact frames
        foot_heights = []
        for i, is_contact in enumerate(contact_frames):
            if is_contact:
                # Random height below threshold
                height = contact_threshold * np.random.uniform(0.3, 0.9)
            else:
                # Random height above threshold
                height = contact_threshold * np.random.uniform(1.2, 3.0)
            foot_heights.append(height)

    if not skating_frames and len(frame_indices) > 0:
        # Define skating as frames with high displacement during contact
        skating_frames = []
        skating_threshold = max_skating * 0.2  # 20% of max as threshold
        for i, (disp, contact) in enumerate(zip(metric_result.frame_average, contact_frames)):
            skating_frames.append(contact and disp > skating_threshold)

    # Add contact overlay to skating plot (background shading)
    if contact_frames:
        contact_y = [max_skating * 0.95 if c else 0 for c in contact_frames]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=contact_y,
                mode="lines",
                name="Contact Detected",
                line=dict(color="rgba(255, 0, 0, 0.2)", width=0),
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.1)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Add skating events overlay
    if skating_frames:
        skating_y = [max_skating * 0.85 if s else 0 for s in skating_frames]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=skating_y,
                mode="lines",
                name="Skating Events",
                line=dict(color="rgba(255, 165, 0, 0.3)", width=0),
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.2)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Distance to ground subplot with threshold
    if foot_heights:
        # Convert to mm for consistency
        heights_mm = [h * 1000 if h < 1 else h for h in foot_heights]
        threshold_mm = contact_threshold * 1000 if contact_threshold < 1 else contact_threshold

        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=heights_mm,
                mode="lines+markers",
                name="Foot Height",
                line=dict(color="blue", width=2),
                marker=dict(size=3),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add threshold line
        fig.add_hline(
            y=threshold_mm,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text=f"Contact Threshold ({threshold_mm:.1f}mm)",
            annotation_position="top right",
            row=2,
            col=1,
        )

    # Update layout and axes
    fig.update_layout(
        height=700,  # Reduced height for 2-row layout
        template="plotly_white",
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    # Update axis labels and ranges
    fig.update_yaxes(title_text=f"Displacement ({skating_unit})", range=[0, max_skating * 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Height (mm)", row=2, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)

    return fig


def create_enhanced_ground_penetration_plot(
    metric_name: str, metric_result: MetricResult, frame_indices: np.ndarray, color: str
) -> go.Figure:
    """Create enhanced ground penetration plot with contact detection and penetration visualization."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create 2-row layout for penetration with contact overlay and distance-to-ground
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{metric_name.replace('_', ' ').title()} & Penetration Detection",
            "Distance to Ground (negative = below ground)",
        ),
        vertical_spacing=0.15,  # Increased spacing for 2-row layout
        row_heights=[0.6, 0.4],  # Give more space to main penetration plot
    )

    # Determine units and extract data
    penetration_unit = "mm"
    max_penetration = max(np.max(metric_result.frame_average), 1.0) if len(metric_result.frame_average) > 0 else 1.0

    # Add main penetration data
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=metric_result.frame_average,
            mode="lines+markers",
            name=f"Penetration ({penetration_unit})",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Extract contact and distance information from metric metadata
    contact_frames = []
    penetration_frames = []
    foot_heights = []
    contact_threshold = 0.05  # Default 50mm (same as foot contact)

    if hasattr(metric_result, "metadata") and metric_result.metadata:
        # Get available metadata
        metadata = metric_result.metadata

        # Try different contact data keys
        for key in ["contact_frames_bool", "contact_frames", "is_contact"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                contact_frames = metadata[key]
                break

        for key in ["penetration_frames_bool", "penetration_frames", "is_penetrating"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                penetration_frames = metadata[key]
                break

        # Get penetration events (all negative distance, separate from penetration detection)
        penetration_events = []
        for key in ["penetration_events_bool", "penetration_events"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                penetration_events = metadata[key]
                break

        for key in ["foot_heights", "heights", "distance_to_ground"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                foot_heights = metadata[key]
                break

        # Get contact threshold (for near ground zone)
        for key in ["contact_threshold", "margin", "tolerance"]:
            if key in metadata:
                contact_threshold = metadata[key]
                break

        # Get penetration threshold (separate from contact threshold)
        penetration_threshold = metadata.get("penetration_threshold", contact_threshold)
    else:
        penetration_threshold = contact_threshold

    # Generate synthetic contact/height data if not available
    if not contact_frames and len(frame_indices) > 0:
        # Simple heuristic: detect periods of high penetration as potential contact
        contact_frames = (metric_result.frame_average > (max_penetration * 0.1)).tolist()

    if not foot_heights and len(frame_indices) > 0:
        # Generate synthetic height data based on contact frames
        foot_heights = []
        for i, is_contact in enumerate(contact_frames):
            if is_contact:
                # Random height below threshold (potentially penetrating)
                height = contact_threshold * np.random.uniform(-0.5, 0.9)
            else:
                # Random height above threshold
                height = contact_threshold * np.random.uniform(1.2, 3.0)
            foot_heights.append(height)

    if not penetration_frames and len(frame_indices) > 0:
        # Define penetration as frames with significant penetration during contact
        penetration_frames = []
        penetration_threshold = max_penetration * 0.2  # 20% of max as threshold
        for i, (pen, contact) in enumerate(zip(metric_result.frame_average, contact_frames)):
            penetration_frames.append(contact and pen > penetration_threshold)

    # Add contact zone overlay (background for reference)
    if contact_frames:
        # Show contact frames as blue background for reference - lower height to not interfere with penetration
        contact_y = [max_penetration * 0.4 if c else 0 for c in contact_frames]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=contact_y,
                mode="lines",
                name="Near Ground Zone",
                line=dict(color="rgba(0, 100, 200, 0.4)", width=0),
                fill="tozeroy",
                fillcolor="rgba(0, 100, 200, 0.15)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Add penetration events overlay (any negative distance)
    if penetration_events:
        events_y = [max_penetration * 0.7 if e else 0 for e in penetration_events]  # 70% height
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=events_y,
                mode="lines",
                name="Below Ground (Events)",
                line=dict(color="rgba(255, 165, 0, 0.5)", width=0),
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.2)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Add penetration detection overlay (detected penetration based on threshold)
    if penetration_frames:
        penetration_y = [max_penetration if p else 0 for p in penetration_frames]  # Full height for visibility
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=penetration_y,
                mode="lines",
                name="⚠️ PENETRATION DETECTED",
                line=dict(color="rgba(255, 0, 0, 0.6)", width=0),
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.3)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Distance to ground subplot with threshold
    if foot_heights:
        # Convert to mm for consistency
        heights_mm = [h * 1000 if h < 1 else h for h in foot_heights]
        threshold_mm = contact_threshold * 1000 if contact_threshold < 1 else contact_threshold

        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=heights_mm,
                mode="lines+markers",
                name="Foot Height",
                line=dict(color="green", width=2),
                marker=dict(size=3),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add threshold line (ground level)
        fig.add_hline(
            y=0,
            line=dict(color="black", width=2, dash="solid"),
            annotation_text="Ground Level (0mm)",
            annotation_position="top right",
            row=2,
            col=1,
        )

        # Add contact threshold line (above ground)
        if contact_threshold > 0:
            fig.add_hline(
                y=threshold_mm,
                line=dict(color="blue", width=2, dash="dot"),
                annotation_text=f"Contact Zone (+{threshold_mm:.1f}mm)",
                annotation_position="top left",
                row=2,
                col=1,
            )

        # Add penetration threshold line (below ground)
        if "penetration_threshold" in locals() and penetration_threshold < 0:
            penetration_mm = (
                abs(penetration_threshold * 1000) if abs(penetration_threshold) < 1 else abs(penetration_threshold)
            )
            fig.add_hline(
                y=-penetration_mm,
                line=dict(color="red", width=2, dash="dash"),
                annotation_text=f"Penetration Threshold (-{penetration_mm:.1f}mm)",
                annotation_position="bottom right",
                row=2,
                col=1,
            )

    # Update layout and axes
    fig.update_layout(
        height=700,  # Reduced height for 2-row layout
        template="plotly_white",
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    # Update axis labels and ranges
    fig.update_yaxes(title_text=f"Penetration ({penetration_unit})", range=[0, max_penetration * 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Height (mm)", row=2, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)

    return fig


def create_enhanced_jerk_plot(
    metric_name: str, metric_result: MetricResult, frame_indices: np.ndarray, color: str, joint_names: list[str] = None
) -> go.Figure:
    """Create enhanced jerk plot with threshold detection, jerk events, and per-joint analysis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create 3-row layout for jerk with threshold detection, jerk events, and per-joint analysis
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            f"{metric_name.replace('_', ' ').title()} & High Jerk Detection",
            "Jerk Events & Threshold Analysis",
            f"Per-Joint Average {metric_name.replace('_', ' ').title()}",
        ),
        vertical_spacing=0.12,  # Increased spacing for 3-row layout
        row_heights=[0.4, 0.3, 0.3],  # Balanced heights for all three plots
    )

    # Determine units and extract data
    jerk_unit = "m/s³"
    max_jerk = max(np.max(metric_result.frame_average), 1.0) if len(metric_result.frame_average) > 0 else 1.0

    # Add main jerk data
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=metric_result.frame_average,
            mode="lines+markers",
            name=f"Jerk ({jerk_unit})",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Extract jerk detection information from metric metadata
    high_jerk_frames = []
    jerk_threshold = 10.0  # Default threshold
    jerk_events = []

    if hasattr(metric_result, "metadata") and metric_result.metadata:
        # Get available metadata
        metadata = metric_result.metadata

        # Get jerk threshold
        for key in ["high_jerk_threshold", "jerk_threshold", "threshold"]:
            if key in metadata:
                jerk_threshold = metadata[key]
                break

        # Get high jerk frames
        for key in ["high_jerk_frames", "frames_with_high_jerk", "jerk_frames"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                high_jerk_frames = metadata[key]
                break

        # Get jerk events (frames where jerk exceeds threshold)
        for key in ["jerk_events", "high_jerk_events"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                jerk_events = metadata[key]
                break

    # Generate synthetic jerk detection data if not available
    if not high_jerk_frames and len(frame_indices) > 0:
        # Detect frames with jerk above threshold
        high_jerk_frames = (metric_result.frame_average > jerk_threshold).tolist()

    if not jerk_events and len(frame_indices) > 0:
        # Define jerk events as frames with significant jerk spikes
        jerk_events = []
        jerk_spike_threshold = jerk_threshold * 1.5  # 50% above threshold for events
        for i, jerk_val in enumerate(metric_result.frame_average):
            jerk_events.append(jerk_val > jerk_spike_threshold)

    # Add high jerk detection overlay
    if high_jerk_frames:
        high_jerk_y = [max_jerk * 0.9 if h else 0 for h in high_jerk_frames]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=high_jerk_y,
                mode="lines",
                name="High Jerk Detected",
                line=dict(color="rgba(255, 0, 0, 0.3)", width=0),
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.2)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Add jerk events overlay (significant spikes)
    if jerk_events:
        events_y = [max_jerk if e else 0 for e in jerk_events]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=events_y,
                mode="lines",
                name="⚠️ JERK SPIKES",
                line=dict(color="rgba(255, 100, 0, 0.6)", width=0),
                fill="tozeroy",
                fillcolor="rgba(255, 100, 0, 0.3)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Add threshold line
    fig.add_hline(
        y=jerk_threshold,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"High Jerk Threshold ({jerk_threshold:.1f} {jerk_unit})",
        annotation_position="top right",
        row=1,
        col=1,
    )

    # Jerk events analysis subplot
    if jerk_events:
        # Create binary jerk events plot
        events_binary = [1 if e else 0 for e in jerk_events]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=events_binary,
                mode="lines+markers",
                name="Jerk Events",
                line=dict(color="orange", width=2),
                marker=dict(size=3),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add jerk ratio information
        total_events = sum(jerk_events)
        total_frames = len(jerk_events)
        jerk_ratio = total_events / total_frames if total_frames > 0 else 0.0

        fig.add_annotation(
            x=0.02,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Jerk Events: {total_events}/{total_frames} frames ({jerk_ratio:.1%})",
            showarrow=False,
            font=dict(size=12, color="orange"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="orange",
            borderwidth=1,
            row=2,
            col=1,
        )

    # Per-joint jerk analysis (third subplot)
    if joint_names and hasattr(metric_result, "per_joint") and len(metric_result.per_joint) > 0:
        joint_values = metric_result.per_joint
        fig.add_trace(
            go.Bar(
                x=joint_names,
                y=joint_values,
                name=f"{metric_name} (per joint)",
                marker_color=color,
                hovertemplate="<b>%{x}</b><br>" + f"{metric_name}: %{{y:.4f}}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # Update layout and axes
    fig.update_layout(
        height=900,  # Increased height for 3-row layout
        template="plotly_white",
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    # Update axis labels and ranges
    fig.update_yaxes(title_text=f"Jerk ({jerk_unit})", range=[0, max_jerk * 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Jerk Events (Binary)", range=[-0.1, 1.1], row=2, col=1)
    if joint_names and hasattr(metric_result, "per_joint") and len(metric_result.per_joint) > 0:
        fig.update_yaxes(title_text=f"Average {metric_name.replace('_', ' ').title()} ({jerk_unit})", row=3, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)
    if joint_names and hasattr(metric_result, "per_joint") and len(metric_result.per_joint) > 0:
        fig.update_xaxes(title_text="Joint", tickangle=45, row=3, col=1)

    return fig


def create_enhanced_floating_plot(
    metric_name: str, metric_result: MetricResult, frame_indices: np.ndarray, color: str
) -> go.Figure:
    """Create enhanced floating plot with contact detection and floating visualization."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create 2-row layout for floating with contact overlay and distance-to-ground
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{metric_name.replace('_', ' ').title()} & Contact Detection",
            "Distance to Ground (with floating threshold)",
        ),
        vertical_spacing=0.15,  # Increased spacing for 2-row layout
        row_heights=[0.6, 0.4],  # Give more space to main floating plot
    )

    # Determine units and extract data
    floating_unit = "mm"
    max_floating = max(np.max(metric_result.frame_average), 1.0) if len(metric_result.frame_average) > 0 else 1.0

    # Add main floating data
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=metric_result.frame_average,
            mode="lines+markers",
            name=f"Floating ({floating_unit})",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Extract contact and distance information from metric metadata
    contact_frames = []
    floating_frames = []
    floating_events = []
    foot_heights = []
    contact_threshold = 0.05  # Default 50mm
    floating_threshold = 0.02  # Default 20mm

    if hasattr(metric_result, "metadata") and metric_result.metadata:
        # Get available metadata
        metadata = metric_result.metadata

        # Try different contact data keys
        for key in ["contact_frames_bool", "contact_frames", "is_contact"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                contact_frames = metadata[key]
                break

        for key in ["floating_frames_bool", "floating_frames", "is_floating"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                floating_frames = metadata[key]
                break

        # Get floating events (all positive distance, separate from floating detection)
        for key in ["floating_events_bool", "floating_events"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                floating_events = metadata[key]
                break

        for key in ["foot_heights", "heights", "distance_to_ground"]:
            if key in metadata and isinstance(metadata[key], (list, np.ndarray)):
                foot_heights = metadata[key]
                break

        # Get contact threshold (for near ground zone)
        for key in ["contact_threshold", "margin", "tolerance"]:
            if key in metadata:
                contact_threshold = metadata[key]
                break

        # Get floating threshold (separate from contact threshold)
        floating_threshold = metadata.get("floating_threshold", floating_threshold)
    else:
        floating_threshold = contact_threshold

    # Generate synthetic contact/height data if not available
    if not contact_frames and len(frame_indices) > 0:
        # Simple heuristic: detect periods of low floating as potential contact
        contact_frames = (metric_result.frame_average < (max_floating * 0.3)).tolist()

    if not foot_heights and len(frame_indices) > 0:
        # Generate synthetic height data based on contact frames
        foot_heights = []
        for i, is_contact in enumerate(contact_frames):
            if is_contact:
                # Random height below threshold
                height = contact_threshold * np.random.uniform(0.3, 0.9)
            else:
                # Random height above threshold
                height = contact_threshold * np.random.uniform(1.2, 3.0)
            foot_heights.append(height)

    if not floating_frames and len(frame_indices) > 0:
        # Define floating as frames with high elevation
        floating_frames = []
        floating_threshold_local = max_floating * 0.2  # 20% of max as threshold
        for i, (elev, contact) in enumerate(zip(metric_result.frame_average, contact_frames)):
            floating_frames.append((not contact) and elev > floating_threshold_local)

    if not floating_events and len(frame_indices) > 0:
        # Define floating events as any non-contact frame
        floating_events = [(not c) for c in contact_frames]

    # Add contact zone overlay (background for reference)
    if contact_frames:
        # Show contact frames as green background for reference - lower height to not interfere with floating
        contact_y = [max_floating * 0.4 if c else 0 for c in contact_frames]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=contact_y,
                mode="lines",
                name="Contact Zone",
                line=dict(color="rgba(0, 150, 0, 0.4)", width=0),
                fill="tozeroy",
                fillcolor="rgba(0, 150, 0, 0.15)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Add floating events overlay (any above ground distance)
    if floating_events:
        events_y = [max_floating * 0.7 if e else 0 for e in floating_events]  # 70% height
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=events_y,
                mode="lines",
                name="Above Ground (Events)",
                line=dict(color="rgba(0, 150, 255, 0.5)", width=0),
                fill="tozeroy",
                fillcolor="rgba(0, 150, 255, 0.2)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Add floating detection overlay (detected floating based on threshold)
    if floating_frames:
        floating_y = [max_floating if f else 0 for f in floating_frames]  # Full height for visibility
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=floating_y,
                mode="lines",
                name="⚠️ FLOATING DETECTED",
                line=dict(color="rgba(255, 100, 0, 0.6)", width=0),
                fill="tozeroy",
                fillcolor="rgba(255, 100, 0, 0.3)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Distance to ground subplot with threshold
    if foot_heights:
        # Convert to mm for consistency
        heights_mm = [h * 1000 if h < 1 else h for h in foot_heights]
        contact_threshold_mm = contact_threshold * 1000 if contact_threshold < 1 else contact_threshold
        floating_threshold_mm = floating_threshold * 1000 if floating_threshold < 1 else floating_threshold

        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=heights_mm,
                mode="lines+markers",
                name="Foot Height",
                line=dict(color="purple", width=2),
                marker=dict(size=3),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add threshold line (ground level)
        fig.add_hline(
            y=0,
            line=dict(color="black", width=2, dash="solid"),
            annotation_text="Ground Level (0mm)",
            annotation_position="top right",
            row=2,
            col=1,
        )

        # Add contact threshold line (above ground)
        if contact_threshold > 0:
            fig.add_hline(
                y=contact_threshold_mm,
                line=dict(color="green", width=2, dash="dot"),
                annotation_text=f"Contact Zone (+{contact_threshold_mm:.1f}mm)",
                annotation_position="top left",
                row=2,
                col=1,
            )

        # Add floating threshold line (above ground)
        if floating_threshold > contact_threshold:
            fig.add_hline(
                y=floating_threshold_mm,
                line=dict(color="orange", width=2, dash="dash"),
                annotation_text=f"Floating Threshold (+{floating_threshold_mm:.1f}mm)",
                annotation_position="bottom left",
                row=2,
                col=1,
            )

    # Update layout and axes
    fig.update_layout(
        height=700,  # Reduced height for 2-row layout
        template="plotly_white",
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    # Update axis labels and ranges
    fig.update_yaxes(title_text=f"Floating ({floating_unit})", range=[0, max_floating * 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Height (mm)", row=2, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)

    return fig


def create_metric_plots(report: EvaluationReport) -> dict[str, go.Figure]:
    """Create interactive plots for motion metrics.

    Args:
        report: EvaluationReport with metric results

    Returns:
        Dictionary mapping metric names to plotly figures
    """
    logger = logging.getLogger(__name__)
    logger.info("📊 Creating metric visualization plots...")

    plots = {}
    num_frames = report.motion_info["num_frames"]
    frame_indices = np.arange(num_frames)

    # Color scheme - avoiding yellow/light colors that are hard to see on white background
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#17becf",  # Cyan
        "#bcbd22",  # Olive (darker than yellow)
    ]

    for i, (metric_name, metric_result) in enumerate(report.physical_metrics.items()):
        # Use enhanced visualization for foot skating metrics
        if "skating" in metric_name:
            fig = create_enhanced_foot_skating_plot(
                metric_name=metric_name,
                metric_result=metric_result,
                frame_indices=frame_indices,
                color=colors[i % len(colors)],
            )
            plots[metric_name] = fig
            continue

        # Use enhanced visualization for ground penetration metrics
        if "penetration" in metric_name:
            fig = create_enhanced_ground_penetration_plot(
                metric_name=metric_name,
                metric_result=metric_result,
                frame_indices=frame_indices,
                color=colors[i % len(colors)],
            )
            plots[metric_name] = fig
            continue

        # Use enhanced visualization for floating metrics
        if "floating" in metric_name:
            fig = create_enhanced_floating_plot(
                metric_name=metric_name,
                metric_result=metric_result,
                frame_indices=frame_indices,
                color=colors[i % len(colors)],
            )
            plots[metric_name] = fig
            continue

        # Use enhanced visualization for jerk metrics
        if "jerk" in metric_name:
            joint_names = report.motion_info["joint_names"]
            fig = create_enhanced_jerk_plot(
                metric_name=metric_name,
                metric_result=metric_result,
                frame_indices=frame_indices,
                color=colors[i % len(colors)],
                joint_names=joint_names,
            )
            plots[metric_name] = fig
            continue
        # Determine if we need per-joint plots and which joints to show
        show_per_joint = True
        joint_names = report.motion_info["joint_names"]
        joint_values = metric_result.per_joint
        joint_filter_indices = None

        # Check metadata for single_joint_metric flag
        if hasattr(metric_result, "metadata") and metric_result.metadata:
            if metric_result.metadata.get("single_joint_metric", False):
                show_per_joint = False

        if "penetration" in metric_name:
            # For ground penetration, skip per-joint plot (most joints don't touch ground)
            show_per_joint = False
        elif "skating" in metric_name or "contact" in metric_name:
            # For foot skating and foot contact, the per_joint data contains only foot joints
            # We need to get the appropriate joint names for labeling
            foot_joint_names = ["left_foot", "right_foot"]

            # Check if we have the expected number of foot joints in the data
            if len(joint_values) == len(foot_joint_names):
                # Use foot joint names for proper labeling
                joint_names = foot_joint_names
                # joint_values already contains the correct foot data
            else:
                # If dimensions don't match, try to find foot joints in the full list
                joint_filter_indices = []
                for foot_name in foot_joint_names:
                    if foot_name in joint_names:
                        joint_filter_indices.append(joint_names.index(foot_name))

                if joint_filter_indices and len(joint_filter_indices) == len(joint_values):
                    # Use the found foot joint names
                    joint_names = [joint_names[i] for i in joint_filter_indices]
                    # joint_values should already be correct
                else:
                    # Fallback if we can't match the data structure
                    show_per_joint = False

        # Create subplot layout
        if show_per_joint:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    f"{metric_name.replace('_', ' ').title()} Over Time",
                    f"Per-Joint Average {metric_name.replace('_', ' ').title()}",
                ),
                vertical_spacing=0.12,
            )
        else:
            fig = make_subplots(
                rows=1,
                cols=1,
                subplot_titles=(f"{metric_name.replace('_', ' ').title()} Over Time",),
            )

        # Per-frame metric plot
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=metric_result.frame_average,
                mode="lines+markers",
                name=f"{metric_name} (per frame)",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                hovertemplate=f"<b>Frame %{{x}}</b><br>{metric_name}: %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add sequence average line
        fig.add_hline(
            y=metric_result.sequence_average,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {metric_result.sequence_average:.4f}",
            row=1,
            col=1,
        )

        # Per-joint metric bar plot (only if show_per_joint is True)
        if show_per_joint:
            fig.add_trace(
                go.Bar(
                    x=joint_names,
                    y=joint_values,
                    name=f"{metric_name} (per joint)",
                    marker_color=colors[i % len(colors)],
                    hovertemplate="<b>%{x}</b><br>" + f"{metric_name}: %{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title_text=f"{metric_name.replace('_', ' ').title()} Analysis",
            showlegend=True,
            height=700 if show_per_joint else 400,  # Shorter height for single plots
            template="plotly_white",
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Frame", row=1, col=1)
        if show_per_joint:
            fig.update_xaxes(title_text="Joint", tickangle=45, row=2, col=1)

        # Update y-axis labels and set reasonable limits for ground penetration
        unit = (
            "degrees"
            if "root_orientation" in metric_name
            else "m/s"
            if "skating" in metric_name
            else "m/s³"
            if "jerk" in metric_name
            else "m"
            if "penetration" in metric_name
            else "score"
            if "dynamic" in metric_name
            else "contact"
            if "contact" in metric_name
            else "m/s²"
        )

        # Set reasonable y-axis limits for small-scale metrics
        if "penetration" in metric_name:
            # Ground penetration: max at 1 cm (0.01 m) or slightly above actual max
            max_value = np.max(metric_result.frame_average)  # Only frame values since no per-joint plot
            y_max = max(0.01, max_value * 1.2)
            fig.update_yaxes(
                title_text=f"{metric_name.replace('_', ' ').title()} ({unit})", range=[0, y_max], row=1, col=1
            )
        elif "contact" in metric_name:
            # Foot contact: binary values (0-1), set y-axis to 0-1.1
            fig.update_yaxes(
                title_text=f"{metric_name.replace('_', ' ').title()} ({unit})", range=[0, 1.1], row=1, col=1
            )
            if show_per_joint:
                fig.update_yaxes(title_text=f"Contact Ratio ({unit})", range=[0, 1.1], row=2, col=1)
        elif "skating" in metric_name:
            # Foot skating: max at 10 cm/s (0.1 m/s) or slightly above actual max
            if show_per_joint:
                max_value = max(np.max(metric_result.frame_average), np.max(joint_values))
            else:
                max_value = np.max(metric_result.frame_average)
            y_max = max(0.1, max_value * 1.2)
            fig.update_yaxes(
                title_text=f"{metric_name.replace('_', ' ').title()} ({unit})", range=[0, y_max], row=1, col=1
            )
            if show_per_joint:
                fig.update_yaxes(
                    title_text=f"Average {metric_name.replace('_', ' ').title()} ({unit})",
                    range=[0, y_max],
                    row=2,
                    col=1,
                )
        else:
            fig.update_yaxes(title_text=f"{metric_name.replace('_', ' ').title()} ({unit})", row=1, col=1)
            if show_per_joint:
                fig.update_yaxes(title_text=f"Average {metric_name.replace('_', ' ').title()} ({unit})", row=2, col=1)

        plots[metric_name] = fig

    # Create summary comparison plot
    summary_fig = go.Figure()

    metric_names = list(report.summary_scores.keys())
    metric_scores = list(report.summary_scores.values())

    # Remove aggregate scores for individual metric visualization
    individual_metrics = [
        (name, score) for name, score in zip(metric_names, metric_scores) if name not in ["physical", "overall"]
    ]

    if individual_metrics:
        names, scores = zip(*individual_metrics)

        # Color code by quality (red < 0.5, yellow < 0.8, green >= 0.8)
        colors_quality = ["red" if s < 0.5 else "orange" if s < 0.8 else "green" for s in scores]

        summary_fig.add_trace(
            go.Bar(
                x=names,
                y=scores,
                marker_color=colors_quality,
                text=[f"{s:.3f}" for s in scores],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Quality Score: %{y:.3f}<extra></extra>",
            )
        )

        summary_fig.update_layout(
            title="Motion Quality Summary (Higher is Better)",
            xaxis_title="Metric",
            yaxis_title="Quality Score (0-1)",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            height=400,
        )

        # Add quality threshold lines
        summary_fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Good (0.8)")
        summary_fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Fair (0.5)")

    plots["summary"] = summary_fig

    logger.info(f"✓ Created {len(plots)} interactive plots")
    return plots


def create_motion_visualization(motion: Motion, output_dir: Path, body_model_path: str | None = None) -> str | None:
    """Create interactive motion visualization using GLB export and HTML viewer.

    Args:
        motion: Motion object to visualize
        output_dir: Directory to save visualization
        body_model_path: Path to SMPL body models for GLB export

    Returns:
        Path to saved HTML visualization or None if failed
    """
    logger = logging.getLogger(__name__)
    logger.info("🎬 Creating interactive motion visualization...")

    if not GLB_AVAILABLE:
        logger.warning("GLB export not available - skipping motion visualization")
        return None

    if not body_model_path or not Path(body_model_path).exists():
        logger.warning("Body models not available - skipping motion visualization")
        return None

    try:
        # Create GLB exporter
        exporter = GLBExporter(model_path=body_model_path, gender="neutral")

        # Export motion to GLB
        glb_path = output_dir / "motion_animation.glb"
        exporter.export(motion, str(glb_path))

        # Create motion visualizer
        motion_visualizer = MotionVisualizer()
        html_path = output_dir / "motion_animation.html"

        motion_visualizer.create_standalone_html(
            glb_path=str(glb_path),
            output_path=str(html_path),
            title=f"Motion Animation ({motion.num_frames} frames)",
            width="100%",
            height="600px",
        )

        logger.info(f"✓ Interactive motion visualization saved: {html_path}")
        return str(html_path)

    except Exception as e:
        logger.error(f"Failed to create motion visualization: {e}")
        logger.info("💡 Tip: Ensure SMPL body models are available and GLB dependencies are installed")
        return None


def get_ratio_color_class(metric_name: str, ratio: float, thresholds: dict[str, dict[str, float]]) -> str:
    """Determine color class for a ratio value based on metric type and thresholds.

    Args:
        metric_name: Name of the metric
        ratio: Ratio value (0-1)
        thresholds: Threshold configuration from config

    Returns:
        Color class string: "excellent", "good", "fair", or "poor"
    """
    # Determine which threshold set to use
    if "skating" in metric_name:
        if "mesh" in metric_name:
            threshold_key = "foot_skating_mesh"
        else:
            threshold_key = "foot_skating_joint"
    elif "jerk" in metric_name:
        threshold_key = "jerk"
    elif "penetration" in metric_name:
        threshold_key = "penetration"
    elif "floating" in metric_name:
        threshold_key = "floating"
    elif "dynamic" in metric_name:
        # Dynamic score is inverted (higher is better)
        dyn_thresholds = thresholds.get("dynamic", {"static": 0.10, "low_activity": 0.50})
        if ratio < dyn_thresholds.get("static", 0.10):
            return "poor"
        elif ratio < dyn_thresholds.get("low_activity", 0.50):
            return "fair"
        else:
            return "excellent"
    else:
        # Fallback to generic thresholds
        return "excellent" if ratio < 0.05 else "good" if ratio < 0.15 else "fair" if ratio < 0.30 else "poor"

    # Get thresholds for this metric type
    metric_thresholds = thresholds.get(threshold_key, {})

    # Determine class based on thresholds (lower ratio is better for most metrics)
    if "excellent" in metric_thresholds:
        # Has excellent/moderate structure
        if ratio < metric_thresholds.get("excellent", 0.15):
            return "excellent"
        elif ratio < metric_thresholds.get("moderate", 0.30):
            return "fair"
        else:
            return "poor"
    elif "good" in metric_thresholds:
        # Has good/moderate structure
        if ratio < metric_thresholds.get("good", 0.40):
            return "good"
        elif ratio < metric_thresholds.get("moderate", 0.60):
            return "fair"
        else:
            return "poor"
    else:
        # Fallback
        return "excellent" if ratio < 0.05 else "good" if ratio < 0.15 else "fair" if ratio < 0.30 else "poor"


def create_comprehensive_html_report(
    motion: Motion,
    report: EvaluationReport,
    plots: dict[str, go.Figure],
    motion_viz_path: str | None,
    output_path: Path,
    motion_name: str = "GVHMR Motion",
    ratio_thresholds: dict[str, dict[str, float]] | None = None,
) -> None:
    """Create comprehensive HTML report with motion visualization and metrics.

    Args:
        motion: Motion object
        report: EvaluationReport with metrics
        plots: Dictionary of plotly figures
        motion_viz_path: Path to motion visualization image
        output_path: Path to save HTML report
        motion_name: Display name for the motion
        ratio_thresholds: Ratio assessment thresholds from config (optional, will use defaults if not provided)
    """
    logger = logging.getLogger(__name__)
    logger.info("📄 Creating comprehensive HTML report...")

    # Use provided ratio thresholds or fall back to hardcoded defaults
    if ratio_thresholds is None:
        ratio_thresholds = RATIO_THRESHOLDS

    # Convert plots to HTML
    plot_htmls = {}
    for plot_name, fig in plots.items():
        plot_htmls[plot_name] = fig.to_html(include_plotlyjs="cdn", div_id=f"plot_{plot_name}")

    # Check if we need to embed motion viewer script
    model_viewer_script_tag = ""
    if motion_viz_path and Path(motion_viz_path).exists():
        try:
            with open(motion_viz_path, encoding="utf-8") as f:
                motion_html_content = f.read()
            import re

            model_viewer_script = re.search(
                r'<script[^>]*src="https://ajax\.googleapis\.com/ajax/libs/model-viewer/[^"]*"[^>]*></script>',
                motion_html_content,
            )
            if model_viewer_script:
                model_viewer_script_tag = model_viewer_script.group(0)
        except Exception as e:
            logger.warning(f"Could not extract model-viewer script: {e}")

    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Motion Evaluation Report - {motion_name}</title>
    
    <!-- Model Viewer Script -->
    {model_viewer_script_tag}
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 25px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            background: #fff;
        }}
        .section h2 {{
            margin-top: 0;
            color: #495057;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            padding: 20px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background: #f8f9fa;
        }}
        .metric-title {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
            text-transform: capitalize;
        }}
        .score {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .score.excellent {{ color: #28a745; }}
        .score.good {{ color: #17a2b8; }}
        .score.fair {{ color: #ffc107; }}
        .score.poor {{ color: #dc3545; }}
        .ratio-info {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 8px;
            padding: 4px 8px;
            background: #e9ecef;
            border-radius: 4px;
            border-left: 3px solid #007bff;
        }}
        .supporting-info {{
            font-size: 0.85em;
            color: #868e96;
            margin-top: 4px;
            line-height: 1.4;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .info-table th, .info-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .info-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        .motion-viz {{
            text-align: center;
            margin: 20px 0;
        }}
        .motion-viz img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            margin: 30px 0;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
        }}
        .recommendations {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendations h3 {{
            margin-top: 0;
            color: #1976d2;
        }}
        .recommendations ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 5px 0;
        }}
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .alert-info {{
            background-color: #d1ecf1;
            border-color: #bee5eb;
            color: #0c5460;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border-color: #ffeaa7;
            color: #856404;
        }}
        .alert-danger {{
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Motion Evaluation Report</h1>
            <div class="subtitle">{motion_name}</div>
            <div style="margin-top: 20px;">
                <span style="font-size: 1.2em; color: white;">
                    Comprehensive Motion Quality Analysis
                </span>
            </div>
        </div>
        
        <div class="content">
            <!-- Motion Information Section -->
            <div class="section">
                <h2>📊 Motion Information</h2>
                <table class="info-table">
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Duration</td><td>{report.motion_info["duration"]:.2f} seconds</td></tr>
                    <tr><td>Frames</td><td>{report.motion_info["num_frames"]}</td></tr>
                    <tr><td>Frame Rate</td><td>{report.motion_info["framerate"]} fps</td></tr>
                    <tr><td>Joints</td><td>{report.motion_info["num_joints"]}</td></tr>
                    <tr><td>Motion Type</td><td>{report.motion_info["motion_type"]}</td></tr>
                </table>
            </div>
    """

    # Add embedded motion visualization if available
    if motion_viz_path and Path(motion_viz_path).exists():
        try:
            # Find the corresponding GLB file
            viz_path = Path(motion_viz_path)
            glb_path = viz_path.parent / viz_path.name.replace("_animation.html", ".glb").replace(
                "_motion.html", ".glb"
            )

            # Try common GLB file names if the derived name doesn't exist
            if not glb_path.exists():
                for glb_name in ["motion_animation.glb", "test_motion.glb", f"{viz_path.stem}.glb"]:
                    potential_glb = viz_path.parent / glb_name
                    if potential_glb.exists():
                        glb_path = potential_glb
                        break

            if glb_path.exists():
                # Use the MotionVisualizer approach: directly embed the GLB file
                import base64
                import uuid

                # Read and encode GLB file as base64
                with open(glb_path, "rb") as f:
                    glb_data = f.read()
                base64_data = base64.b64encode(glb_data).decode("utf-8")
                glb_base64_url = f"data:model/gltf-binary;base64,{base64_data}"

                # Generate unique IDs for this embedded viewer
                unique_id = str(uuid.uuid4())[:8]

                # Embed the motion viewer using the same approach as MotionVisualizer
                html_content += f"""
            <!-- Embedded Motion Visualization Section -->
            <div class="section">
                <h2>🎬 Interactive Motion Visualization</h2>
                <div class="motion-viz" style="margin: 20px 0;">
                    <p><em>Interactive 3D animation with {motion.num_frames} frames at {motion.framerate}fps</em></p>
                    
                    <div class="motion-viewer-container-{unique_id}" style="position: relative; width: 100%; height: 500px; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; margin: 15px 0; background: #f8f9fa;">
                        <div class="loading-overlay" id="loadingOverlay_{unique_id}" style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(248, 249, 250, 0.9); display: flex; align-items: center; justify-content: center; z-index: 10;">
                            <div class="spinner" style="width: 40px; height: 40px; border: 4px solid #e5e7eb; border-top: 4px solid #667eea; border-radius: 50%; animation: spin_{unique_id} 1s linear infinite;"></div>
                        </div>
                        
                        <model-viewer
                            id="motionViewer_{unique_id}"
                            style="display: block; width: 100%; height: 100%; position: relative;"
                            src="{glb_base64_url}"
                            ar
                            camera-controls
                            camera-orbit="-180deg 75deg 8m"
                            camera-target="0m 1m 0m"
                            min-camera-orbit="auto auto 4m"
                            max-camera-orbit="auto auto 20m"
                            touch-action="pan-y"
                            shadow-intensity="0.5"
                            shadow-softness="1"
                            interaction-prompt="none"
                            autoplay
                        >
                            <!-- Control Panel -->
                            <div class="motion-viewer-controls" id="controlPanel_{unique_id}" style="position: absolute; bottom: 16px; left: 16px; right: 16px; display: flex; align-items: center; gap: 8px; padding: 12px; background-color: rgba(0, 0, 0, 0.7); backdrop-filter: blur(8px); border-radius: 8px; transition: all 0.3s ease;">
                                <!-- Play/Pause Button -->
                                <button id="playPauseBtn_{unique_id}" style="background: none; border: none; color: white; cursor: pointer; padding: 8px; border-radius: 4px; transition: background-color 0.2s; display: flex; align-items: center; justify-content: center;" title="Play/Pause">
                                    <svg id="playIcon_{unique_id}" style="width: 16px; height: 16px; display: none;" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M8 5v14l11-7z"/>
                                    </svg>
                                    <svg id="pauseIcon_{unique_id}" style="width: 16px; height: 16px;" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
                                    </svg>
                                </button>

                                <!-- Stop Button -->
                                <button id="stopBtn_{unique_id}" style="background: none; border: none; color: white; cursor: pointer; padding: 8px; border-radius: 4px; transition: background-color 0.2s; display: flex; align-items: center; justify-content: center;" title="Stop">
                                    <svg style="width: 16px; height: 16px;" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M6 6h12v12H6z"/>
                                    </svg>
                                </button>

                                <!-- Progress Slider -->
                                <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    value="0"
                                    id="progressSlider_{unique_id}"
                                    style="flex: 1; margin: 0 12px; background: transparent; outline: none; -webkit-appearance: none; appearance: none; height: 4px; border-radius: 2px; background: rgba(255, 255, 255, 0.3); cursor: pointer;"
                                />
                            </div>
                        </model-viewer>
                    </div>
                    
                    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <strong>🎮 Animation Controls:</strong>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Use mouse/touch to rotate, zoom, and pan the view</li>
                            <li>Play/pause controls at the bottom of the viewer</li>
                            <li>Progress slider to jump to specific frames</li>
                            <li>Mouse wheel to zoom, right-click and drag to pan</li>
                        </ul>
                    </div>
                </div>
                
                <!-- CSS animations and JavaScript for the embedded viewer -->
                <style>
                    @keyframes spin_{unique_id} {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}
                    #controlPanel_{unique_id} button:hover {{
                        background-color: rgba(255, 255, 255, 0.1) !important;
                    }}
                    #progressSlider_{unique_id}::-webkit-slider-thumb {{
                        -webkit-appearance: none;
                        appearance: none;
                        width: 16px;
                        height: 16px;
                        border-radius: 50%;
                        background: white;
                        cursor: pointer;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    }}
                    #progressSlider_{unique_id}::-moz-range-thumb {{
                        width: 16px;
                        height: 16px;
                        border-radius: 50%;
                        background: white;
                        cursor: pointer;
                        border: none;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    }}
                </style>
                
                <script>
                (function() {{
                    // Get references to elements
                    const modelViewer = document.getElementById('motionViewer_{unique_id}');
                    const controlPanel = document.getElementById('controlPanel_{unique_id}');
                    const playPauseBtn = document.getElementById('playPauseBtn_{unique_id}');
                    const stopBtn = document.getElementById('stopBtn_{unique_id}');
                    const progressSlider = document.getElementById('progressSlider_{unique_id}');
                    const playIcon = document.getElementById('playIcon_{unique_id}');
                    const pauseIcon = document.getElementById('pauseIcon_{unique_id}');
                    const loadingOverlay = document.getElementById('loadingOverlay_{unique_id}');
                    
                    if (!modelViewer) return;
                    
                    let isPlaying = true;
                    let animationDuration = 0;
                    
                    // Hide loading overlay when model loads
                    modelViewer.addEventListener('load', () => {{
                        if (loadingOverlay) loadingOverlay.style.display = 'none';
                        
                        // Get animation duration
                        const animations = modelViewer.availableAnimations;
                        if (animations && animations.length > 0) {{
                            animationDuration = modelViewer.getDuration();
                        }}
                    }});
                    
                    // Play/pause functionality
                    if (playPauseBtn) {{
                        playPauseBtn.addEventListener('click', () => {{
                            if (isPlaying) {{
                                modelViewer.pause();
                                if (playIcon) playIcon.style.display = 'block';
                                if (pauseIcon) pauseIcon.style.display = 'none';
                            }} else {{
                                modelViewer.play();
                                if (playIcon) playIcon.style.display = 'none';
                                if (pauseIcon) pauseIcon.style.display = 'block';
                            }}
                            isPlaying = !isPlaying;
                        }});
                    }}
                    
                    // Stop functionality
                    if (stopBtn) {{
                        stopBtn.addEventListener('click', () => {{
                            modelViewer.currentTime = 0;
                            if (progressSlider) progressSlider.value = 0;
                        }});
                    }}
                    
                    // Progress slider functionality
                    if (progressSlider) {{
                        progressSlider.addEventListener('input', (e) => {{
                            const progress = parseFloat(e.target.value) / 100;
                            modelViewer.currentTime = progress * animationDuration;
                        }});
                        
                        // Update slider position during playback
                        modelViewer.addEventListener('timeupdate', () => {{
                            if (animationDuration > 0) {{
                                const progress = (modelViewer.currentTime / animationDuration) * 100;
                                progressSlider.value = progress;
                            }}
                        }});
                    }}
                }})();
                </script>
            </div>
        """
            else:
                logger.warning(f"GLB file not found for motion visualization: {glb_path}")
        except Exception as e:
            logger.warning(f"Failed to embed motion visualization, using link instead: {e}")
            viz_filename = Path(motion_viz_path).name
            html_content += f"""
            <!-- Motion Visualization Section -->
            <div class="section">
                <h2>🎬 Interactive Motion Visualization</h2>
                <div class="motion-viz">
                    <p><strong>3D Interactive Animation:</strong> <a href="{viz_filename}" target="_blank">Open Motion Animation</a></p>
                    <p><em>Full motion sequence with {motion.num_frames} frames at {motion.framerate}fps</em></p>
                </div>
            </div>
        """

    # Add metrics summary
    html_content += """
            <!-- Metrics Summary Section -->
            <div class="section">
                <h2>📈 Quality Metrics Summary</h2>
                <div class="metrics-grid">
    """

    # Add individual metric cards
    for metric_name, metric_result in report.physical_metrics.items():
        score = report.summary_scores.get(metric_name, 0.0)
        score_class = "excellent" if score >= 0.9 else "good" if score >= 0.8 else "fair" if score >= 0.5 else "poor"

        # For foot skating, ground penetration, floating, jerk, and root orientation metrics, show ratio as main metric
        ratio_based_metrics = (
            "skating" in metric_name
            or "penetration" in metric_name
            or "floating" in metric_name
            or "jerk" in metric_name
        )

        # Special handling for root orientation - show max angle instead of ratio
        if "root_orientation" in metric_name and hasattr(metric_result, "ratio") and metric_result.ratio is not None:
            # For root orientation, show max rotation change as main metric
            max_angle_value = np.max(metric_result.per_frame)
            main_display = f"{max_angle_value:.2f}°"

            # Determine color class based on max angle thresholds
            # Good: < 15°, Moderate: 15-30°, High concern: > 30°
            if max_angle_value < 15.0:
                angle_class = "excellent"
            elif max_angle_value < 30.0:
                angle_class = "good"
            elif max_angle_value < 45.0:
                angle_class = "fair"
            else:
                angle_class = "poor"

            # Show ratio and other stats as supporting info
            mean_angle = f"{metric_result.metadata.get('mean_angle_degrees', 0):.2f}°"
            threshold = metric_result.metadata.get("rotation_threshold_degrees", 45)
            ratio_value = metric_result.ratio
            supporting_info = (
                f'<div class="supporting-info">Frames above {threshold:.0f}° threshold: {ratio_value:.1%}</div>'
            )
            supporting_info += f'<div class="supporting-info">Mean rotation change: {mean_angle}</div>'

            html_content += f"""
                        <div class="metric-card">
                            <div class="metric-title">{metric_name.replace("_", " ")}</div>
                            <div class="score {angle_class}">{main_display}</div>
                            {supporting_info}
                        </div>
            """
        elif hasattr(metric_result, "ratio") and metric_result.ratio is not None and ratio_based_metrics:
            # Use ratio as the main colored metric for skating/penetration/floating/jerk
            main_value = metric_result.ratio
            main_display = f"{main_value:.1%}"

            # Determine color class using config-based thresholds
            ratio_class = get_ratio_color_class(metric_name, main_value, ratio_thresholds)

            # Show displacement statistics as supporting info
            if "skating" in metric_name:
                avg_disp = f"{metric_result.sequence_average:.2f}mm"
                max_disp = f"{np.max(metric_result.per_frame):.2f}mm"
                supporting_info = f'<div class="supporting-info">Avg displacement: {avg_disp}</div>'
                supporting_info += f'<div class="supporting-info">Max displacement: {max_disp}</div>'
                supporting_info += f'<div class="supporting-info">Quality score: {score:.3f}</div>'
            elif "floating" in metric_name:
                avg_height = f"{metric_result.sequence_average:.2f}mm"
                max_height = f"{np.max(metric_result.per_frame):.2f}mm"
                supporting_info = f'<div class="supporting-info">Avg floating height: {avg_height}</div>'
                supporting_info += f'<div class="supporting-info">Max floating height: {max_height}</div>'
                supporting_info += f'<div class="supporting-info">Quality score: {score:.3f}</div>'
            elif "jerk" in metric_name:
                # For jerk metrics, show actual jerk values as supporting info
                jerk_unit = "m/s³"
                avg_jerk = f"{metric_result.sequence_average:.2f} {jerk_unit}"
                max_jerk = f"{np.max(metric_result.per_frame):.2f} {jerk_unit}"
                supporting_info = f'<div class="supporting-info">Avg jerk: {avg_jerk}</div>'
                supporting_info += f'<div class="supporting-info">Max jerk: {max_jerk}</div>'
                supporting_info += f'<div class="supporting-info">Quality score: {score:.3f}</div>'
            else:  # penetration
                avg_pen = f"{metric_result.sequence_average:.2f}mm"
                max_pen = f"{np.max(metric_result.per_frame):.2f}mm"
                supporting_info = f'<div class="supporting-info">Avg penetration: {avg_pen}</div>'
                supporting_info += f'<div class="supporting-info">Max penetration: {max_pen}</div>'
                supporting_info += f'<div class="supporting-info">Quality score: {score:.3f}</div>'

            html_content += f"""
                        <div class="metric-card">
                            <div class="metric-title">{metric_name.replace("_", " ")}</div>
                            <div class="score {ratio_class}">{main_display}</div>
                            {supporting_info}
                        </div>
            """
        else:
            # For other metrics, use the standard display with score as main metric
            ratio_info = ""
            if hasattr(metric_result, "ratio") and metric_result.ratio is not None:
                # Determine ratio meaning based on metric type
                if "jerk" in metric_name:
                    ratio_label = "High Jerk Frames"
                elif "acceleration" in metric_name:
                    ratio_label = "High Acceleration Frames"
                elif "dynamic" in metric_name:
                    ratio_label = "Active Motion Frames"
                elif "contact" in metric_name:
                    ratio_label = "Overall Contact Ratio"
                elif "floating" in metric_name:
                    ratio_label = "Floating Frames"
                elif "root_orientation" in metric_name:
                    # Get threshold from metadata
                    if hasattr(metric_result, "metadata"):
                        threshold = metric_result.metadata.get("rotation_threshold_degrees", 45)
                    else:
                        threshold = 45
                    ratio_label = f"High Rotation Frames (>{threshold:.0f}°)"
                else:
                    ratio_label = "Ratio"

                ratio_info = f'<div class="ratio-info">{ratio_label}: {metric_result.ratio:.1%}</div>'

            html_content += f"""
                        <div class="metric-card">
                            <div class="metric-title">{metric_name.replace("_", " ")}</div>
                            <div class="score {score_class}">{score:.3f}</div>
                            <div>Sequence Average: {metric_result.sequence_average:.4f}</div>
                            <div>Max Value: {np.max(metric_result.per_frame):.4f}</div>
                            {ratio_info}
                        </div>
            """

    html_content += """
                </div>
            </div>
    """

    # Add detailed ratio analysis table
    ratios_found = False
    ratio_table_content = ""

    for metric_name, metric_result in report.physical_metrics.items():
        if hasattr(metric_result, "ratio") and metric_result.ratio is not None:
            if not ratios_found:
                ratios_found = True
                ratio_table_content += """
            <div class="section">
                <h2>📊 Detailed Ratio Analysis</h2>
                <p><em>These ratios provide additional insights into the distribution and severity of metric violations.</em></p>
                <table class="info-table">
                    <tr>
                        <th>Metric</th>
                        <th>Ratio Value</th>
                        <th>Meaning</th>
                        <th>Assessment</th>
                    </tr>
                """

            # Determine ratio meaning and assessment
            if "skating" in metric_name:
                meaning = "Percentage of contact frames where foot skating occurs"
                # Different thresholds for joint-based vs mesh-based
                if "mesh" in metric_name:
                    thresholds = RATIO_THRESHOLDS["foot_skating_mesh"]
                    assessment = (
                        "🟢 Good"
                        if metric_result.ratio < thresholds["good"]
                        else "🟡 Moderate"
                        if metric_result.ratio < thresholds["moderate"]
                        else "🔴 High concern"
                    )
                else:
                    thresholds = RATIO_THRESHOLDS["foot_skating_joint"]
                    assessment = (
                        "🟢 Good"
                        if metric_result.ratio < thresholds["good"]
                        else "🟡 Moderate"
                        if metric_result.ratio < thresholds["moderate"]
                        else "🔴 High concern"
                    )
            elif "penetration" in metric_name:
                meaning = "Percentage of contact frames with ground penetration"
                thresholds = RATIO_THRESHOLDS["penetration"]
                assessment = (
                    "🟢 Excellent"
                    if metric_result.ratio < thresholds["excellent"]
                    else "🟡 Moderate"
                    if metric_result.ratio < thresholds["moderate"]
                    else "🔴 High concern"
                )
            elif "jerk" in metric_name:
                meaning = (
                    f"Percentage of frames with jerk > {metric_result.metadata.get('high_jerk_threshold', 10)} m/s³"
                )
                thresholds = RATIO_THRESHOLDS["jerk"]
                assessment = (
                    "🟢 Good"
                    if metric_result.ratio < thresholds["good"]
                    else "🟡 Moderate"
                    if metric_result.ratio < thresholds["moderate"]
                    else "🔴 High concern"
                )
            elif "acceleration" in metric_name:
                meaning = "Percentage of frames with high acceleration spikes"
                assessment = (
                    "🟢 Excellent"
                    if metric_result.ratio < 0.15
                    else "🟡 Moderate"
                    if metric_result.ratio < 0.30
                    else "🔴 High concern"
                )
            elif "dynamic" in metric_name:
                meaning = "Percentage of frames with active motion (above threshold)"
                thresholds = RATIO_THRESHOLDS["dynamic"]
                assessment = (
                    "🔴 Static"
                    if metric_result.ratio < thresholds["static"]
                    else "🟡 Low activity"
                    if metric_result.ratio < thresholds["low_activity"]
                    else "🟢 Active"
                )
            elif "contact" in metric_name:
                meaning = "Overall percentage of frames where feet are in contact with ground"
                assessment = (
                    "🔴 Floating/Airborne"
                    if metric_result.ratio < 0.20
                    else "🟡 Mixed Contact"
                    if metric_result.ratio < 0.60
                    else "🟢 Good Ground Contact"
                )
            elif "floating" in metric_name:
                meaning = "Percentage of frames where floating was detected (body hovering above ground)"
                thresholds = RATIO_THRESHOLDS["floating"]
                assessment = (
                    "🟢 Excellent"
                    if metric_result.ratio < thresholds["excellent"]
                    else "🟡 Moderate"
                    if metric_result.ratio < thresholds["moderate"]
                    else "🔴 High concern"
                )
            else:
                meaning = "Metric-specific ratio measurement"
                assessment = "🟡 Review needed"

            ratio_table_content += f"""
                    <tr>
                        <td><strong>{metric_name.replace("_", " ").title()}</strong></td>
                        <td style="font-size: 1.2em; font-weight: bold;">{metric_result.ratio:.1%}</td>
                        <td>{meaning}</td>
                        <td>{assessment}</td>
                    </tr>
            """

    if ratios_found:
        ratio_table_content += """
                </table>
                <div class="alert alert-info">
                    <strong>💡 Interpretation Guide:</strong>
                    <ul>
                        <li><strong>🟢 Excellent:</strong> Minimal issues - motion quality is very good</li>
                        <li><strong>🟡 Moderate:</strong> Some issues present - consider review and optimization</li>
                        <li><strong>🔴 High concern:</strong> Significant issues - intervention recommended</li>
                    </ul>
                </div>
            </div>
        """
        html_content += ratio_table_content

    # Add summary plot
    if "summary" in plot_htmls:
        html_content += f"""
            <!-- Summary Plot Section -->
            <div class="section">
                <h2>📊 Quality Score Overview</h2>
                <div class="plot-container">
                    {plot_htmls["summary"]}
                </div>
            </div>
        """

    # Add detailed metric plots
    html_content += """
            <!-- Detailed Metrics Section -->
            <div class="section">
                <h2>📉 Detailed Metric Analysis</h2>
    """

    for metric_name, plot_html in plot_htmls.items():
        if metric_name != "summary":
            html_content += f"""
                <h3>{metric_name.replace("_", " ").title()}</h3>
                <div class="plot-container">
                    {plot_html}
                </div>
            """

    # Add recommendations
    recommendations = generate_recommendations(report)
    if recommendations:
        html_content += f"""
            </div>
            
            <!-- Recommendations Section -->
            <div class="section">
                <div class="recommendations">
                    <h3>💡 Recommendations for Improvement</h3>
                    <ul>
                        {"".join(f"<li>{rec}</li>" for rec in recommendations)}
                    </ul>
                </div>
            </div>
        """

    # Add metadata and close
    html_content += f"""
            <!-- Technical Details Section -->
            <div class="section">
                <h2>🔧 Technical Details</h2>
                <div class="alert alert-info">
                    <strong>Evaluation Configuration:</strong>
                    <ul>
                        <li>Configuration: Comprehensive Preset</li>
                        <li>Metrics computed: {len(report.physical_metrics)} different quality measures</li>
                        <li>Analysis scope: Per-frame and sequence-level assessment</li>
                    </ul>
                </div>
                
                <div class="alert alert-info">
                    <strong>Metrics Computed:</strong> {", ".join(report.physical_metrics.keys())}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    """

    # Save HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"✓ Comprehensive HTML report saved: {output_path}")


def generate_recommendations(report: EvaluationReport) -> list[str]:
    """Generate recommendations based on evaluation scores.

    Args:
        report: EvaluationReport with scores

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Check individual metrics and provide specific advice
    if report.summary_scores.get("foot_skating", 1.0) < 0.7:
        skating_avg = report.physical_metrics["foot_skating"].sequence_average
        recommendations.append(
            f"High foot skating detected (avg: {skating_avg:.3f} m/s). Consider adding contact constraints or smoothing foot trajectories during ground contact phases."
        )

    if report.summary_scores.get("jerk", 1.0) < 0.7:
        jerk_avg = report.physical_metrics["jerk"].sequence_average
        recommendations.append(
            f"Motion appears jerky (avg: {jerk_avg:.1f} m/s³). Consider temporal smoothing, increasing frame rate, or applying motion filters."
        )

    if report.summary_scores.get("ground_penetration", 1.0) < 0.9:
        penetration_avg = report.physical_metrics["ground_penetration"].sequence_average
        recommendations.append(
            f"Ground penetration detected (avg: {penetration_avg:.3f} m). Check foot positioning constraints and ensure proper contact modeling."
        )

    # Check skating ratio from ratio field of skating metrics
    if "foot_skating" in report.physical_metrics:
        skating_result = report.physical_metrics["foot_skating"]
        if hasattr(skating_result, "ratio") and skating_result.ratio is not None:
            if skating_result.ratio > 0.20:  # More than 20% skating during contact
                recommendations.append(
                    f"High foot skating ratio during contact detected ({skating_result.ratio:.1%} of contact frames). "
                    f"This indicates significant sliding during foot contact phases. "
                    f"Consider improving contact constraints, motion smoothing, or ground interaction modeling."
                )

    # Check penetration ratio from ratio field of penetration metrics
    if "ground_penetration" in report.physical_metrics:
        penetration_result = report.physical_metrics["ground_penetration"]
        if hasattr(penetration_result, "ratio") and penetration_result.ratio is not None:
            if penetration_result.ratio > 0.25:  # More than 25% penetration during contact
                recommendations.append(
                    f"High penetration ratio during contact detected ({penetration_result.ratio:.1%} of contact frames). "
                    f"This indicates significant ground violations during foot contact phases. "
                    f"Consider improving contact detection or ground constraint modeling."
                )

    if report.summary_scores.get("acceleration_peaks", 1.0) < 0.7:
        accel_avg = report.physical_metrics["acceleration_peaks"].sequence_average
        recommendations.append(
            f"High acceleration peaks detected (avg: {accel_avg:.1f} m/s²). Review motion capture quality, consider outlier removal or temporal filtering."
        )

    # Check dynamic score for motion activity
    if "dynamic_score" in report.physical_metrics:
        dynamic_result = report.physical_metrics["dynamic_score"]
        if dynamic_result.sequence_average < 0.1:  # Very low dynamism
            recommendations.append(
                f"Motion appears very static (dynamic score: {dynamic_result.sequence_average:.3f}). "
                f"This might indicate lack of natural movement, frozen poses, or T-pose sequences. "
                f"Consider adding natural idle motion or checking for motion capture issues."
            )
        elif hasattr(dynamic_result, "ratio") and dynamic_result.ratio is not None:
            if dynamic_result.ratio < 0.10:  # Less than 10% active frames
                recommendations.append(
                    f"Very low motion activity ({dynamic_result.ratio:.1%} active frames). "
                    f"Motion may contain long static periods or minimal movement. "
                    f"Consider reviewing animation timing or adding subtle secondary motion."
                )

    # Check foot contact metrics for debugging insights
    for metric_name, metric_result in report.physical_metrics.items():
        if "contact" in metric_name and hasattr(metric_result, "ratio") and metric_result.ratio is not None:
            if metric_result.ratio < 0.20:  # Very little ground contact
                recommendations.append(
                    f"Low ground contact detected in {metric_name.replace('_', ' ')} ({metric_result.ratio:.1%}). "
                    f"Character appears to be floating or airborne most of the time. "
                    f"This might indicate issues with floor alignment or contact detection thresholds."
                )
            elif metric_result.ratio > 0.95:  # Almost always in contact
                recommendations.append(
                    f"Very high ground contact detected in {metric_name.replace('_', ' ')} ({metric_result.ratio:.1%}). "
                    f"Character appears to be stuck to the ground. "
                    f"This might indicate overly loose contact detection or lack of natural lifting movements."
                )

    # Overall quality recommendations
    overall_score = report.summary_scores["overall"]
    if overall_score >= 0.9:
        recommendations.append("Excellent motion quality! This motion meets high standards for physical plausibility.")
    elif overall_score >= 0.8:
        recommendations.append(
            "Good motion quality with minor issues. Consider fine-tuning based on specific metric feedback."
        )
    elif overall_score >= 0.6:
        recommendations.append("Fair motion quality. Focus on addressing the lowest-scoring metrics for improvement.")
    else:
        recommendations.append(
            "Poor motion quality detected. Significant post-processing or re-generation may be needed."
        )

    return recommendations


def main():
    """Main script function."""
    parser = argparse.ArgumentParser(description="GVHMR Motion Evaluation and Visualization")
    parser.add_argument(
        "--data-file", type=Path, default=Path("sample_data/gvhmr/gvhmr_1.pt"), help="Path to GVHMR data file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/motion_evaluation"),
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames to process (default: process all frames)"
    )
    parser.add_argument(
        "--body-models-path",
        type=Path,
        default=Path.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl",
        help="Path to body models directory",
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 GVHMR Motion Evaluation Pipeline")
    logger.info("=" * 60)

    # Step 1: Load and convert GVHMR data
    if not args.data_file.exists():
        logger.error(f"❌ Data file not found: {args.data_file}")
        return

    motion = load_and_convert_gvhmr(
        str(args.data_file),
        max_frames=args.max_frames,
        body_model_path=str(args.body_models_path) if args.body_models_path.exists() else None,
    )

    if motion is None:
        logger.error("❌ Failed to load motion data")
        return

    # Step 2: Evaluate motion metrics
    report, evaluator = evaluate_motion_metrics(motion)

    # Step 3: Create visualizations
    plots = create_metric_plots(report)
    motion_viz_path = create_motion_visualization(
        motion, args.output_dir, body_model_path=str(args.body_models_path) if args.body_models_path.exists() else None
    )

    # Step 4: Generate comprehensive HTML report
    motion_name = args.data_file.stem
    html_output = args.output_dir / f"{motion_name}_evaluation_report.html"

    # Extract ratio thresholds from evaluator config
    ratio_thresholds = evaluator.evaluation_config.metadata.get("ratio_thresholds", None)

    create_comprehensive_html_report(
        motion=motion,
        report=report,
        plots=plots,
        motion_viz_path=motion_viz_path,
        output_path=html_output,
        motion_name=motion_name,
        ratio_thresholds=ratio_thresholds,
    )

    # Step 5: Save evaluation data
    json_output = args.output_dir / f"{motion_name}_evaluation_data.json"
    report.save_json(json_output)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Motion evaluation completed successfully!")
    logger.info(f"📁 Results saved to: {args.output_dir}")
    logger.info(f"🌐 Open HTML report: {html_output}")

    # Print summary of findings
    logger.info("\n📋 Quality Summary:")
    for metric_name, score in report.summary_scores.items():
        if metric_name not in ["physical", "overall"]:
            status = "✓" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
            logger.info(f"   {status} {metric_name}: {score:.3f}")


if __name__ == "__main__":
    main()
