"""Configuration loader for motion evaluation metrics.

This module provides functionality to load and validate YAML configuration files
for the MetricEvaluator system, enabling flexible, file-based configuration
of evaluation metrics and their parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .base_metric import MetricRegistry


@dataclass
class MetricConfig:
    """Configuration for a single metric instance.

    Args:
        name: Unique name for this metric instance
        metric: Metric type (must be registered in MetricRegistry)
        convention: Convention to use for this metric
        params: Parameters specific to this metric/convention combination
        enabled: Whether this metric should be computed (default: True)
        description: Optional description of this metric configuration
    """

    name: str
    metric: str
    convention: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by MetricEvaluator."""
        return {"name": self.name, "metric": self.metric, "convention": self.convention, "params": self.params}


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration.

    Args:
        name: Name/description of this configuration
        version: Configuration format version
        metrics: List of metric configurations
        global_params: Global parameters that apply to all metrics
        metadata: Additional metadata about this configuration
    """

    name: str
    version: str = "1.0"
    metrics: list[MetricConfig] = field(default_factory=list)
    global_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled_metrics(self) -> list[MetricConfig]:
        """Get only enabled metric configurations."""
        return [metric for metric in self.metrics if metric.enabled]

    def get_metric_configs(self) -> list[dict[str, Any]]:
        """Get metric configurations in format expected by MetricEvaluator."""
        return [metric.to_dict() for metric in self.enabled_metrics]


class ConfigLoader:
    """Loader and validator for evaluation configuration files.

    Supports loading YAML configuration files and validating them against
    the available metrics and conventions in the MetricRegistry.

    Examples:
        >>> # Load configuration from file
        >>> loader = ConfigLoader()
        >>> config = loader.load_config("path/to/config.yaml")
        >>>
        >>> # Create evaluator from config
        >>> evaluator = MetricEvaluator(config.get_metric_configs())
        >>>
        >>> # Validate config without loading
        >>> is_valid, errors = loader.validate_config_file("config.yaml")
    """

    def __init__(self):
        """Initialize configuration loader."""
        self._available_metrics = MetricRegistry.list_metrics()
        self._metric_conventions = {
            metric: MetricRegistry.get_conventions(metric) for metric in self._available_metrics
        }

    def load_config(self, config_path: str | Path) -> EvaluationConfig:
        """Load and validate configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            EvaluationConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If configuration is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML in {config_path}: {e}")

        # Validate and convert to EvaluationConfig
        return self._parse_config(config_data, config_path)

    def load_config_from_string(self, config_yaml: str) -> EvaluationConfig:
        """Load configuration from YAML string.

        Args:
            config_yaml: YAML configuration as string

        Returns:
            EvaluationConfig object

        Raises:
            yaml.YAMLError: If YAML is malformed
            ValueError: If configuration is invalid
        """
        try:
            config_data = yaml.safe_load(config_yaml)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML: {e}")

        return self._parse_config(config_data, "<string>")

    def validate_config_file(self, config_path: str | Path) -> tuple[bool, list[str]]:
        """Validate configuration file without loading it.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            self.load_config(config_path)
            return True, []
        except Exception as e:
            return False, [str(e)]

    def _parse_config(self, config_data: dict[str, Any], source: str | Path) -> EvaluationConfig:
        """Parse and validate configuration data.

        Args:
            config_data: Parsed YAML data
            source: Source file/string for error messages

        Returns:
            EvaluationConfig object

        Raises:
            ValueError: If configuration is invalid
        """
        errors = []

        # Validate required fields
        if not isinstance(config_data, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config_data)}")

        # Extract basic fields
        name = config_data.get("name", f"Config from {source}")
        version = config_data.get("version", "1.0")
        global_params = config_data.get("global_params", {})
        metadata = config_data.get("metadata", {})

        # Extract ratio_thresholds if present and add to metadata
        if "ratio_thresholds" in config_data:
            metadata["ratio_thresholds"] = config_data["ratio_thresholds"]

        # Parse metrics
        metrics_data = config_data.get("metrics", [])
        if not isinstance(metrics_data, list):
            raise ValueError("'metrics' must be a list")

        metrics = []
        for i, metric_data in enumerate(metrics_data):
            try:
                metric_config = self._parse_metric_config(metric_data, global_params)
                metrics.append(metric_config)
            except ValueError as e:
                errors.append(f"Metric {i}: {e}")

        if errors:
            error_msg = f"Configuration validation failed in {source}:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

        return EvaluationConfig(
            name=name, version=version, metrics=metrics, global_params=global_params, metadata=metadata
        )

    def _parse_metric_config(self, metric_data: dict[str, Any], global_params: dict[str, Any]) -> MetricConfig:
        """Parse and validate a single metric configuration.

        Args:
            metric_data: Metric configuration data
            global_params: Global parameters to merge with metric params

        Returns:
            MetricConfig object

        Raises:
            ValueError: If metric configuration is invalid
        """
        # Check required fields
        required_fields = ["name", "metric", "convention"]
        missing = [field for field in required_fields if field not in metric_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        name = metric_data["name"]
        metric_type = metric_data["metric"]
        convention_str = metric_data["convention"]
        enabled = metric_data.get("enabled", True)
        description = metric_data.get("description")

        # Validate metric type
        if metric_type not in self._available_metrics:
            raise ValueError(f"Unknown metric type '{metric_type}'. Available: {self._available_metrics}")

        # Validate convention
        available_conventions = self._metric_conventions[metric_type]
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

        # Merge parameters (global params + metric-specific params)
        params = {}
        params.update(global_params)
        params.update(metric_data.get("params", {}))

        # Validate parameters by trying to create the metric
        try:
            MetricRegistry.create_metric(metric_type, convention, **params)
        except Exception as e:
            raise ValueError(f"Invalid parameters for {metric_type}/{convention_str}: {e}")

        return MetricConfig(
            name=name,
            metric=metric_type,
            convention=convention_str,
            params=params,
            enabled=enabled,
            description=description,
        )

    def create_default_config(self, config_type: str = "comprehensive") -> EvaluationConfig:
        """Create a default configuration.

        Args:
            config_type: Type of default config to create
                       - "comprehensive": All metrics with default settings
                       - "stablemotion": StableMotion-compatible settings
                       - "basic": Basic set of commonly used metrics

        Returns:
            EvaluationConfig object
        """
        if config_type == "comprehensive":
            return self._create_comprehensive_config()
        elif config_type == "stablemotion":
            return self._create_stablemotion_config()
        elif config_type == "basic":
            return self._create_basic_config()
        else:
            raise ValueError(f"Unknown config type: {config_type}. Available: comprehensive, stablemotion, basic")

    def _create_comprehensive_config(self) -> EvaluationConfig:
        """Create comprehensive configuration with all available metrics."""
        metrics = [
            MetricConfig(
                name="foot_skating_height",
                metric="foot_skating",
                convention="height_based_contact",
                params={"velocity_threshold": 0.02, "contact_height_threshold": 0.05},
                description="Height-based foot contact detection",
            ),
            MetricConfig(
                name="foot_skating_displacement",
                metric="foot_skating",
                convention="displacement_based_contact",
                params={"velocity_threshold": 0.02, "displacement_threshold": 0.01},
                description="StableMotion-style displacement-based contact detection",
            ),
            MetricConfig(
                name="foot_skating_velocity",
                metric="foot_skating",
                convention="velocity_based_contact",
                params={"velocity_threshold": 0.02, "contact_velocity_threshold": 0.01},
                description="Velocity-based contact detection",
            ),
            MetricConfig(
                name="jerk_standard",
                metric="jerk",
                convention="standard_jerk",
                description="Standard jerk calculation (3rd derivative)",
            ),
            MetricConfig(
                name="jerk_rms",
                metric="jerk",
                convention="rms_jerk",
                description="RMS jerk for better statistical properties",
            ),
            MetricConfig(
                name="ground_penetration",
                metric="ground_penetration",
                convention="simple_height_check",
                params={"ground_height": 0.0},
                description="Simple Y-coordinate ground penetration check",
            ),
        ]

        return EvaluationConfig(
            name="Comprehensive Evaluation Configuration",
            version="1.0",
            metrics=metrics,
            metadata={
                "description": "Comprehensive evaluation with all available metric variants",
                "use_case": "Research and detailed motion analysis",
            },
        )

    def _create_stablemotion_config(self) -> EvaluationConfig:
        """Create StableMotion-compatible configuration."""
        metrics = [
            MetricConfig(
                name="foot_skating",
                metric="foot_skating",
                convention="displacement_based_contact",
                params={"velocity_threshold": 0.02, "displacement_threshold": 0.01},
                description="StableMotion-compatible foot skating detection",
            ),
            MetricConfig(
                name="jerk",
                metric="jerk",
                convention="standard_jerk",
                description="Standard jerk calculation matching StableMotion",
            ),
            MetricConfig(
                name="ground_penetration",
                metric="ground_penetration",
                convention="simple_height_check",
                params={"ground_height": 0.0},
                description="Ground penetration detection",
            ),
            MetricConfig(
                name="acceleration",
                metric="acceleration",
                convention="peak_detection",
                params={"peak_threshold": 50.0},
                description="Acceleration peak detection",
            ),
        ]

        return EvaluationConfig(
            name="StableMotion Compatible Configuration",
            version="1.0",
            metrics=metrics,
            metadata={
                "description": "Configuration compatible with StableMotion evaluation standards",
                "reference": "StableMotion paper evaluation methodology",
                "use_case": "Research comparison and benchmarking",
            },
        )

    def _create_basic_config(self) -> EvaluationConfig:
        """Create basic configuration with essential metrics."""
        metrics = [
            MetricConfig(
                name="foot_skating",
                metric="foot_skating",
                convention="height_based_contact",
                params={"velocity_threshold": 0.02, "contact_height_threshold": 0.05},
                description="Basic foot skating detection",
            ),
            MetricConfig(
                name="jerk", metric="jerk", convention="standard_jerk", description="Motion smoothness analysis"
            ),
            MetricConfig(
                name="ground_penetration",
                metric="ground_penetration",
                convention="simple_height_check",
                params={"ground_height": 0.0},
                description="Ground collision detection",
            ),
        ]

        return EvaluationConfig(
            name="Basic Evaluation Configuration",
            version="1.0",
            metrics=metrics,
            metadata={
                "description": "Basic set of essential motion quality metrics",
                "use_case": "Quick motion quality assessment",
            },
        )

    def save_config(self, config: EvaluationConfig, output_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration to save
            output_path: Output file path
        """
        output_path = Path(output_path)

        # Convert to serializable format
        config_dict = {
            "name": config.name,
            "version": config.version,
            "metadata": config.metadata,
            "global_params": config.global_params,
            "metrics": [],
        }

        for metric in config.metrics:
            metric_dict = {
                "name": metric.name,
                "metric": metric.metric,
                "convention": metric.convention,
                "enabled": metric.enabled,
                "params": metric.params,
            }
            if metric.description:
                metric_dict["description"] = metric.description
            config_dict["metrics"].append(metric_dict)

        # Save to YAML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
