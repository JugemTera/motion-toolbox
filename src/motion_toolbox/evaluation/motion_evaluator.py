"""Clean metric-based motion evaluator.

This module provides a simple, configurable evaluator that instantiates
specific metric implementations based on configuration and computes them
on motion data. Supports both programmatic configuration and YAML config files.
"""

from pathlib import Path
from typing import Any

from ..core.motion import Motion
from .base_metric import MetricRegistry, MetricResult
from .config_loader import ConfigLoader, EvaluationConfig
from .evaluator import EvaluationReport


class MotionEvaluator:
    """Modern, configurable motion evaluator with YAML configuration support.

    This is the primary motion evaluation system. It supports YAML configuration files,
    multiple metric conventions, and provides comprehensive motion quality assessment.
    Each metric is instantiated once and reused for evaluations.

    Examples:
        >>> # Define metric configurations
        >>> metric_configs = [
        ...     {
        ...         "name": "foot_skating_height",
        ...         "metric": "foot_skating",
        ...         "convention": "height_based_contact",
        ...         "params": {
        ...             "velocity_threshold": 0.02,
        ...             "contact_height_threshold": 0.05
        ...         }
        ...     },
        ...     {
        ...         "name": "foot_skating_stablemotion",
        ...         "metric": "foot_skating",
        ...         "convention": "displacement_based_contact",
        ...         "params": {
        ...             "velocity_threshold": 0.02,
        ...             "displacement_threshold": 0.01
        ...         }
        ...     },
        ...     {
        ...         "name": "jerk_standard",
        ...         "metric": "jerk",
        ...         "convention": "standard_jerk",
        ...         "params": {}
        ...     }
        ... ]
        >>>
        >>> # Create evaluator
        >>> evaluator = MotionEvaluator(metric_configs)
        >>>
        >>> # Evaluate motion
        >>> report = evaluator.evaluate(motion)
        >>> print(f"Height-based foot skating: {report.physical_metrics['foot_skating_height'].sequence_average:.4f}")
        >>> print(f"StableMotion foot skating: {report.physical_metrics['foot_skating_stablemotion'].sequence_average:.4f}")
    """

    def __init__(self, config: list[dict[str, Any]] | str | Path | EvaluationConfig | None = None):
        """Initialize evaluator with metric configurations.

        Args:
            config: Configuration for the evaluator. Can be:
                   - List of metric config dictionaries (legacy format)
                   - Path to YAML configuration file (str or Path)
                   - EvaluationConfig object
                   - None (creates basic default configuration)

        Raises:
            ValueError: If configurations are invalid or names are not unique
            FileNotFoundError: If config file doesn't exist
        """
        self.config_loader = ConfigLoader()

        # Handle different input types
        if config is None:
            # Create default basic configuration
            eval_config = self.config_loader.create_default_config("basic")
            self.metric_configs = eval_config.get_metric_configs()
            self.evaluation_config = eval_config
        elif isinstance(config, (str, Path)):
            # Load from YAML file
            eval_config = self.config_loader.load_config(config)
            self.metric_configs = eval_config.get_metric_configs()
            self.evaluation_config = eval_config
        elif isinstance(config, EvaluationConfig):
            # Use provided EvaluationConfig
            self.metric_configs = config.get_metric_configs()
            self.evaluation_config = config
        elif isinstance(config, list):
            # Legacy format: list of metric config dictionaries
            self.metric_configs = config
            # Create a minimal EvaluationConfig for compatibility
            self.evaluation_config = EvaluationConfig(
                name="Legacy Configuration",
                metrics=[],  # Will be empty since we're using dict format
            )
        else:
            raise ValueError(
                f"Invalid config type: {type(config)}. Expected list, str, Path, EvaluationConfig, or None"
            )

        self.metrics = {}
        self._validate_and_initialize_metrics()

    def _validate_and_initialize_metrics(self) -> None:
        """Validate configurations and initialize metric instances."""
        names = []

        for config in self.metric_configs:
            # Validate required fields
            required = ["name", "metric", "convention", "params"]
            missing = [field for field in required if field not in config]
            if missing:
                raise ValueError(f"Missing required fields in metric config: {missing}")

            name = config["name"]
            metric_type = config["metric"]
            convention_str = config["convention"]
            params = config["params"]

            # Check for duplicate names
            if name in names:
                raise ValueError(f"Duplicate metric name: {name}")
            names.append(name)

            # Validate metric type exists
            if metric_type not in MetricRegistry.list_metrics():
                available = MetricRegistry.list_metrics()
                raise ValueError(f"Unknown metric type '{metric_type}'. Available: {available}")

            # Get convention enum
            available_conventions = MetricRegistry.get_conventions(metric_type)
            convention = None
            for conv in available_conventions:
                if conv.value == convention_str:
                    convention = conv
                    break

            if convention is None:
                available_strs = [conv.value for conv in available_conventions]
                raise ValueError(
                    f"Unknown convention '{convention_str}' for metric '{metric_type}'. Available: {available_strs}"
                )

            # Create metric instance
            try:
                metric_instance = MetricRegistry.create_metric(metric_type, convention, **params)
                self.metrics[name] = metric_instance
            except Exception as e:
                raise ValueError(f"Failed to create metric '{name}': {e}")

    def evaluate(self, motion: Motion, include_metrics: list[str] | None = None) -> EvaluationReport:
        """Evaluate motion using configured metrics.

        Args:
            motion: Motion object to evaluate
            include_metrics: List of metric names to compute. If None, computes all configured metrics.

        Returns:
            EvaluationReport with results

        Raises:
            ValueError: If requested metric names are not configured
        """
        if include_metrics is None:
            include_metrics = list(self.metrics.keys())

        # Validate requested metrics
        invalid = set(include_metrics) - set(self.metrics.keys())
        if invalid:
            available = list(self.metrics.keys())
            raise ValueError(f"Requested metrics not configured: {invalid}. Available: {available}")

        # Extract motion info
        motion_info = self._extract_motion_info(motion)

        # Compute metrics
        results = {}
        for metric_name in include_metrics:
            try:
                results[metric_name] = self.metrics[metric_name].compute(motion)
            except Exception as e:
                print(f"Warning: Failed to compute metric '{metric_name}': {e}")
                continue

        # Compute summary scores
        summary_scores = self._compute_summary_scores(results)

        return EvaluationReport(
            motion_info=motion_info,
            physical_metrics=results,
            summary_scores=summary_scores,
        )

    def _extract_motion_info(self, motion: Motion) -> dict[str, Any]:
        """Extract basic motion information."""
        return {
            "num_frames": motion.num_frames,
            "num_joints": motion.num_joints,
            "duration": motion.duration,
            "framerate": motion.framerate,
            "joint_names": motion.skeleton.joint_names,
            "motion_type": "Motion",
        }

    def _compute_summary_scores(self, results: dict[str, MetricResult]) -> dict[str, float]:
        """Compute summary scores from metric results."""
        if not results:
            return {"overall": 0.0, "physical": 0.0}

        # Convert metric values to quality scores (0-1, higher is better)
        quality_scores = {}

        for metric_name, result in results.items():
            # Get the base metric type to determine scoring
            config = next(c for c in self.metric_configs if c["name"] == metric_name)
            metric_type = config["metric"]

            if metric_type == "foot_skating":
                # Lower skating is better; use sigmoid to map [0, inf) to [1, 0)
                quality_scores[metric_name] = 1.0 / (1.0 + result.sequence_average * 50)
            elif metric_type == "jerk":
                # Lower jerk is better
                quality_scores[metric_name] = 1.0 / (1.0 + result.sequence_average / 100)
            elif metric_type == "ground_penetration":
                # Lower penetration is better
                quality_scores[metric_name] = 1.0 / (1.0 + result.sequence_average * 100)
            elif metric_type == "acceleration":
                # Lower acceleration peaks are better
                quality_scores[metric_name] = 1.0 / (1.0 + result.sequence_average / 50)
            else:
                # Default: assume lower is better
                quality_scores[metric_name] = 1.0 / (1.0 + result.sequence_average)

        # Compute overall scores
        physical_score = float(sum(quality_scores.values()) / len(quality_scores))
        overall_score = physical_score  # For now, only physical metrics

        return {
            "overall": overall_score,
            "physical": physical_score,
            **quality_scores,
        }

    def get_metric_info(self) -> dict[str, dict[str, Any]]:
        """Get information about configured metrics.

        Returns:
            Dictionary mapping metric names to their information
        """
        info = {}
        for name, metric in self.metrics.items():
            config = next(c for c in self.metric_configs if c["name"] == name)
            info[name] = {
                **metric.get_info(),
                "base_metric_type": config["metric"],
            }
        return info

    def compare_motions(
        self, motions: dict[str, Motion], include_metrics: list[str] | None = None
    ) -> dict[str, EvaluationReport]:
        """Compare multiple motions using the same metrics.

        Args:
            motions: Dictionary mapping motion names to Motion objects
            include_metrics: List of metric names to compute

        Returns:
            Dictionary mapping motion names to their evaluation reports
        """
        reports = {}
        for motion_name, motion in motions.items():
            try:
                reports[motion_name] = self.evaluate(motion, include_metrics)
            except Exception as e:
                print(f"Warning: Failed to evaluate motion '{motion_name}': {e}")
                continue
        return reports

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "MotionEvaluator":
        """Create evaluator from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            MotionEvaluator configured from file

        Examples:
            >>> # Load from preset config
            >>> evaluator = MotionEvaluator.from_config_file("configs/evaluation/stablemotion_compatible.yaml")
            >>>
            >>> # Load from custom config
            >>> evaluator = MotionEvaluator.from_config_file("my_custom_config.yaml")
        """
        return cls(config_path)

    @classmethod
    def from_preset(cls, preset_name: str) -> "MotionEvaluator":
        """Create evaluator from preset configuration.

        Args:
            preset_name: Name of preset configuration
                        Options: "basic", "comprehensive", "stablemotion"

        Returns:
            MotionEvaluator with preset configuration

        Examples:
            >>> # Basic configuration
            >>> evaluator = MotionEvaluator.from_preset("basic")
            >>>
            >>> # StableMotion-compatible
            >>> evaluator = MotionEvaluator.from_preset("stablemotion")
        """
        loader = ConfigLoader()
        eval_config = loader.create_default_config(preset_name)
        return cls(eval_config)

    @classmethod
    def create_stablemotion_compatible(cls) -> "MotionEvaluator":
        """Create evaluator with StableMotion-compatible metric configurations.

        Returns:
            MotionEvaluator configured for StableMotion compatibility
        """
        return cls.from_preset("stablemotion")

    @classmethod
    def create_comparison_evaluator(cls) -> "MotionEvaluator":
        """Create evaluator that computes the same metrics with different conventions for comparison.

        Returns:
            MotionEvaluator with multiple conventions for comparison
        """
        return cls.from_preset("comprehensive")
