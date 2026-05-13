"""Semantic evaluation metrics for text-to-motion generation.

This module implements the standard HumanML3D/T2M evaluation protocol used by
MoMask, MDM, T2M-GPT, and other text-to-motion generation works. It provides
distribution-level metrics that measure how well generated motions match ground-truth
in a learned text-motion co-embedding space.

Metrics:
    - FID (Frechet Inception Distance): Distribution similarity (lower is better)
    - R-Precision (Top-1/2/3): Text-motion retrieval accuracy (higher is better)
    - Matching Score: Text-motion alignment quality (lower is better)
    - Diversity: Variation across generated motions (higher is better)
    - MultiModality: Variation per text prompt (higher is better)

Usage:
    >>> from motion_toolbox.evaluation.semantic import SemanticEvaluator
    >>> evaluator = SemanticEvaluator(
    ...     dataset_name="humanml",
    ...     checkpoint_dir="./checkpoints",
    ...     glove_dir="./glove",
    ... )
    >>> report = evaluator.evaluate(gen_motions, gen_lens, gt_motions, gt_lens, texts)
    >>> print(report.summary())
"""

from .evaluator_wrapper import BaseEvaluatorWrapper, T2MEvaluatorWrapper
from .metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_matching_score,
    calculate_multimodality,
    calculate_R_precision,
    euclidean_distance_matrix,
)
from .semantic_evaluator import (
    SemanticEvaluationReport,
    SemanticEvaluator,
    SemanticMetricResult,
)
from .word_vectorizer import WordVectorizer, texts_to_embeddings, tokenize_text

__all__ = [
    # Main evaluator
    "SemanticEvaluator",
    "SemanticEvaluationReport",
    "SemanticMetricResult",
    # Evaluator wrappers
    "BaseEvaluatorWrapper",
    "T2MEvaluatorWrapper",
    # Metric functions
    "calculate_frechet_distance",
    "calculate_R_precision",
    "calculate_matching_score",
    "calculate_diversity",
    "calculate_multimodality",
    "calculate_activation_statistics",
    "euclidean_distance_matrix",
    # Text processing
    "WordVectorizer",
    "texts_to_embeddings",
    "tokenize_text",
]
