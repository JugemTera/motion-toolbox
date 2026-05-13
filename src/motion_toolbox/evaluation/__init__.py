"""Motion evaluation metrics and assessment tools.

This module provides comprehensive evaluation capabilities for motion data,
including physical plausibility metrics, motion quality assessment, and
semantic evaluation metrics for text-to-motion generation.

The primary interfaces are:
    - MotionEvaluator: Per-motion physical metrics with YAML configuration
    - SemanticEvaluator: Distribution-level semantic metrics (FID, R-Precision, etc.)
    - EvaluationReportGenerator: Multi-format report generation (HTML, LaTeX, PDF)
"""

# Import metric implementations to ensure registration
from . import metrics  # noqa: F401

# Core metric system components
from .base_metric import EvalMetric, MetricConvention, MetricRegistry
from .config_loader import ConfigLoader, EvaluationConfig, MetricConfig
from .evaluator import EvaluationReport

# Primary motion evaluation interface
from .motion_evaluator import MotionEvaluator

# Legacy API (backward compatibility)
from .physical_metrics import (
    MetricResult,
    calculate_foot_skating,
    calculate_ground_penetration,
    calculate_jerk,
)

# Report generation
from .report import EvaluationReportGenerator

# Semantic evaluation (lazy imports to avoid torch dependency when not needed)
from .semantic import (
    SemanticEvaluationReport,
    SemanticEvaluator,
    SemanticMetricResult,
)

__all__ = [
    # Primary motion evaluation interface
    "MotionEvaluator",
    "EvaluationReport",
    # Semantic evaluation interface
    "SemanticEvaluator",
    "SemanticEvaluationReport",
    "SemanticMetricResult",
    # Report generation
    "EvaluationReportGenerator",
    # Configuration system
    "ConfigLoader",
    "EvaluationConfig",
    "MetricConfig",
    # Metric system components
    "EvalMetric",
    "MetricConvention",
    "MetricRegistry",
    "MetricResult",
    # Legacy API (backward compatibility)
    "calculate_foot_skating",
    "calculate_jerk",
    "calculate_ground_penetration",
]
