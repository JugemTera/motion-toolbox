"""Evaluation framework for comprehensive motion assessment.

This module provides a unified evaluation framework that aggregates multiple
physical and quality metrics to assess motion data comprehensively.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.motion import Motion
from .physical_metrics import (
    MetricResult,
    calculate_foot_skating,
    calculate_ground_penetration,
    calculate_jerk,
)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for motion data.

    Contains aggregated results from multiple evaluation metrics along with
    metadata and summary statistics.

    Args:
        motion_info: Basic information about the motion (frames, joints, duration)
        physical_metrics: Results from physical plausibility metrics
        summary_scores: Aggregated scores for quick comparison
        metadata: Additional context about the evaluation

    Examples:
        >>> report = evaluator.evaluate_motion(motion)
        >>> print(f"Overall quality score: {report.summary_scores['overall']:.3f}")
        >>> print(f"Physical plausibility: {report.summary_scores['physical']:.3f}")
    """

    motion_info: dict[str, Any]
    physical_metrics: dict[str, MetricResult]
    summary_scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        import numpy as np

        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        return {
            "motion_info": convert_numpy_types(self.motion_info),
            "physical_metrics": {
                name: {
                    "per_frame": result.per_frame.tolist(),
                    "per_joint": result.per_joint.tolist(),
                    "frame_average": result.frame_average.tolist(),
                    "sequence_average": float(result.sequence_average),
                    "ratio": float(result.ratio) if result.ratio is not None else None,
                    "metadata": convert_numpy_types(result.metadata),
                }
                for name, result in self.physical_metrics.items()
            },
            "summary_scores": convert_numpy_types(self.summary_scores),
            "metadata": convert_numpy_types(self.metadata),
        }

    def save_json(self, filepath: str | Path) -> None:
        """Save evaluation report as JSON file.

        Args:
            filepath: Path to save the JSON report
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str | Path) -> "EvaluationReport":
        """Load evaluation report from JSON file.

        Args:
            filepath: Path to the JSON report file

        Returns:
            EvaluationReport instance
        """
        with open(filepath) as f:
            data = json.load(f)

        # Reconstruct MetricResult objects
        physical_metrics = {}
        for name, metric_data in data["physical_metrics"].items():
            physical_metrics[name] = MetricResult(
                per_frame=np.array(metric_data["per_frame"]),
                per_joint=np.array(metric_data["per_joint"]),
                frame_average=np.array(metric_data["frame_average"]),
                sequence_average=metric_data["sequence_average"],
            )

        return cls(
            motion_info=data["motion_info"],
            physical_metrics=physical_metrics,
            summary_scores=data["summary_scores"],
            metadata=data.get("metadata", {}),
        )


class MotionEvaluator:
    """Comprehensive motion evaluation system.

    This class provides a unified interface for evaluating motion quality using
    multiple physical and semantic metrics. It aggregates results into summary
    scores and generates comprehensive reports.

    Args:
        foot_joint_names: Names of foot joints for skating analysis
        velocity_threshold: Skating detection sensitivity (m/s)
        contact_height_threshold: Ground contact detection threshold (m)
        ground_height: Y-coordinate of ground plane
        use_stablemotion_contact: Use StableMotion-style displacement-based contact detection

    Examples:
        >>> evaluator = MotionEvaluator()
        >>> report = evaluator.evaluate_motion(motion)
        >>>
        >>> # Save detailed report
        >>> report.save_json('motion_evaluation.json')
        >>>
        >>> # Quick quality check
        >>> if report.summary_scores['physical'] < 0.8:
        ...     print("Motion may have physical plausibility issues")
    """

    def __init__(
        self,
        foot_joint_names: list[str] | None = None,
        velocity_threshold: float = 0.02,
        contact_height_threshold: float = 0.05,
        ground_height: float = 0.0,
        use_stablemotion_contact: bool = False,
    ):
        """Initialize evaluator with configuration parameters."""
        self.foot_joint_names = foot_joint_names or ["left_foot", "right_foot"]
        self.velocity_threshold = velocity_threshold
        self.contact_height_threshold = contact_height_threshold
        self.ground_height = ground_height
        self.use_stablemotion_contact = use_stablemotion_contact

    def evaluate_motion(
        self,
        motion: Motion,
        include_metrics: list[str] | None = None,
    ) -> EvaluationReport:
        """Perform comprehensive evaluation of motion data.

        Args:
            motion: Motion object to evaluate
            include_metrics: List of metrics to compute. If None, computes all available
                           Options: ['foot_skating', 'jerk', 'ground_penetration']

        Returns:
            EvaluationReport with comprehensive analysis results

        Examples:
            >>> # Full evaluation
            >>> report = evaluator.evaluate_motion(motion)
            >>>
            >>> # Evaluate only specific metrics
            >>> report = evaluator.evaluate_motion(motion, include_metrics=['foot_skating', 'jerk'])
        """
        if include_metrics is None:
            include_metrics = ["foot_skating", "jerk", "ground_penetration"]

        # Extract basic motion information
        motion_info = self._extract_motion_info(motion)

        # Compute physical metrics
        physical_metrics = {}

        if "foot_skating" in include_metrics:
            physical_metrics["foot_skating"] = calculate_foot_skating(
                motion,
                foot_joint_names=self.foot_joint_names,
                velocity_threshold=self.velocity_threshold,
                contact_height_threshold=self.contact_height_threshold,
                use_displacement_contact=self.use_stablemotion_contact,
            )

        if "jerk" in include_metrics:
            physical_metrics["jerk"] = calculate_jerk(motion)

        if "ground_penetration" in include_metrics:
            physical_metrics["ground_penetration"] = calculate_ground_penetration(
                motion,
                ground_height=self.ground_height,
                foot_joint_names=self.foot_joint_names,
            )

        # Compute summary scores
        summary_scores = self._compute_summary_scores(physical_metrics)

        # Create metadata
        metadata = {
            "evaluator_config": {
                "foot_joint_names": self.foot_joint_names,
                "velocity_threshold": self.velocity_threshold,
                "contact_height_threshold": self.contact_height_threshold,
                "ground_height": self.ground_height,
            },
            "metrics_computed": list(physical_metrics.keys()),
        }

        return EvaluationReport(
            motion_info=motion_info,
            physical_metrics=physical_metrics,
            summary_scores=summary_scores,
            metadata=metadata,
        )

    def _extract_motion_info(self, motion: Motion) -> dict[str, Any]:
        """Extract basic information about the motion."""
        return {
            "num_frames": motion.num_frames,
            "num_joints": motion.num_joints,
            "duration": motion.duration,
            "framerate": motion.framerate,
            "joint_names": motion.skeleton.joint_names,
            "motion_type": "Motion",
        }

    def _compute_summary_scores(self, physical_metrics: dict[str, MetricResult]) -> dict[str, float]:
        """Compute aggregated summary scores from individual metrics.

        Summary scores are normalized to [0, 1] range where 1.0 indicates
        perfect quality and 0.0 indicates poor quality.
        """
        scores = {}

        # Individual metric scores (normalized and inverted so higher = better)
        if "foot_skating" in physical_metrics:
            # Convert skating to quality score (lower skating = higher quality)
            skating_avg = physical_metrics["foot_skating"].sequence_average
            scores["foot_skating"] = max(0.0, 1.0 - min(skating_avg / 0.1, 1.0))  # Cap at 0.1 m/s

        if "jerk" in physical_metrics:
            # Convert jerk to smoothness score (lower jerk = higher smoothness)
            jerk_avg = physical_metrics["jerk"].sequence_average
            scores["jerk"] = max(0.0, 1.0 - min(jerk_avg / 100.0, 1.0))  # Cap at 100 m/s³

        if "ground_penetration" in physical_metrics:
            # Convert penetration to plausibility score
            penetration_avg = physical_metrics["ground_penetration"].sequence_average
            scores["ground_penetration"] = max(0.0, 1.0 - min(penetration_avg / 0.05, 1.0))  # Cap at 5cm

        # Aggregate scores
        if scores:
            scores["physical"] = np.mean(list(scores.values()))
            scores["overall"] = scores["physical"]  # Can be extended with semantic metrics later
        else:
            scores["physical"] = 1.0
            scores["overall"] = 1.0

        return scores

    def compare_motions(
        self,
        motions: dict[str, Motion],
        include_metrics: list[str] | None = None,
    ) -> dict[str, EvaluationReport]:
        """Evaluate and compare multiple motions.

        Args:
            motions: Dictionary mapping motion names to motion objects
            include_metrics: List of metrics to compute for each motion

        Returns:
            Dictionary mapping motion names to their evaluation reports

        Examples:
            >>> motions = {
            ...     'original': original_motion,
            ...     'generated': generated_motion,
            ...     'refined': refined_motion,
            ... }
            >>> reports = evaluator.compare_motions(motions)
            >>>
            >>> # Compare overall quality
            >>> for name, report in reports.items():
            ...     print(f"{name}: {report.summary_scores['overall']:.3f}")
        """
        return {name: self.evaluate_motion(motion, include_metrics=include_metrics) for name, motion in motions.items()}

    def generate_html_report(
        self,
        report: EvaluationReport,
        output_path: str | Path,
        motion_name: str = "Motion",
    ) -> None:
        """Generate an HTML visualization report.

        Args:
            report: EvaluationReport to visualize
            output_path: Path to save the HTML file
            motion_name: Display name for the motion

        Note:
            This creates a basic HTML report. For advanced visualizations,
            consider integrating with plotly or similar libraries.
        """
        html_content = self._create_html_report(report, motion_name)

        with open(output_path, "w") as f:
            f.write(html_content)

    def _create_html_report(self, report: EvaluationReport, motion_name: str) -> str:
        """Create HTML content for the evaluation report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Motion Evaluation Report - {motion_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 8px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .score {{ font-weight: bold; color: #2e7d32; }}
        .warning {{ color: #f57c00; }}
        .error {{ color: #d32f2f; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Motion Evaluation Report</h1>
        <h2>{motion_name}</h2>
        <p><strong>Overall Quality Score:</strong> <span class="score">{report.summary_scores["overall"]:.3f}</span></p>
    </div>
    
    <div class="section">
        <h3>Motion Information</h3>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Frames</td><td>{report.motion_info["num_frames"]}</td></tr>
            <tr><td>Joints</td><td>{report.motion_info["num_joints"]}</td></tr>
            <tr><td>Duration</td><td>{report.motion_info["duration"]:.2f}s</td></tr>
            <tr><td>Framerate</td><td>{report.motion_info["framerate"]}fps</td></tr>
            <tr><td>Type</td><td>{report.motion_info["motion_type"]}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h3>Physical Metrics</h3>
        """

        for metric_name, metric_result in report.physical_metrics.items():
            score = report.summary_scores.get(metric_name, 0.0)
            score_class = "score" if score > 0.8 else "warning" if score > 0.5 else "error"

            html += f"""
        <div class="metric">
            <h4>{metric_name.replace("_", " ").title()}</h4>
            <p><strong>Quality Score:</strong> <span class="{score_class}">{score:.3f}</span></p>
            <p><strong>Sequence Average:</strong> {metric_result.sequence_average:.4f}</p>
            <p><strong>Max Per Frame:</strong> {np.max(metric_result.per_frame):.4f}</p>
            <p><strong>Problematic Joints:</strong> {len([x for x in metric_result.per_joint if x > metric_result.sequence_average * 1.5])}</p>
        </div>
            """

        html += """
    </div>
    
    <div class="section">
        <h3>Recommendations</h3>
        <ul>
        """

        # Generate recommendations based on scores
        if report.summary_scores.get("foot_skating", 1.0) < 0.7:
            html += "<li>Consider reducing foot sliding artifacts during ground contact phases</li>"
        if report.summary_scores.get("jerk", 1.0) < 0.7:
            html += "<li>Motion appears jerky - consider smoothing or increasing temporal resolution</li>"
        if report.summary_scores.get("ground_penetration", 1.0) < 0.9:
            html += "<li>Ground penetration detected - check foot positioning and contact constraints</li>"

        html += (
            """
        </ul>
    </div>
    
    <div class="section">
        <h3>Metadata</h3>
        <p><strong>Metrics Computed:</strong> """
            + ", ".join(report.metadata.get("metrics_computed", []))
            + """</p>
        <p><strong>Evaluation Configuration:</strong></p>
        <ul>
        """
        )

        config = report.metadata.get("evaluator_config", {})
        for key, value in config.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"

        html += """
        </ul>
    </div>
</body>
</html>
        """

        return html
