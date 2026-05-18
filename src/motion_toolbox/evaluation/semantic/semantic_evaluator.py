"""High-level semantic evaluation orchestrator for text-to-motion generation.

The SemanticEvaluator coordinates the full evaluation pipeline:
    1. Load pre-trained text-motion co-embedding model
    2. Extract embeddings from generated and ground-truth motions + text
    3. Compute semantic metrics (FID, R-Precision, Matching Score, Diversity, MultiModality)
    4. Optionally repeat N times and report mean + 95% confidence intervals

This follows the standard evaluation protocol used by MoMask, MDM, T2M-GPT,
and other text-to-motion generation works.

Input format:
    Currently supports HumanML3D 263-d features (normalized). A future
    extension can accept :class:`~motion_toolbox.core.motion.Motion`
    instances and convert internally via the HumanML3D converters in
    ``motion_toolbox.converters.humanml3d_*``.

References:
    - Guo et al., "Generating Diverse and Natural 3D Human Motions from Text" (CVPR 2022)
    - MoMask (CVPR 2024): https://github.com/EricGuo5513/momask-codes
    - MDM (ICLR 2023): https://github.com/GuyTevet/motion-diffusion-model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .evaluator_wrapper import T2MEvaluatorWrapper
from .metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    calculate_R_precision,
    euclidean_distance_matrix,
)
from .word_vectorizer import WordVectorizer, texts_to_embeddings


@dataclass
class SemanticMetricResult:
    """Container for a single semantic metric result across multiple evaluation runs.

    Args:
        name: Metric name (e.g., "FID", "R_precision_top1").
        values: List of values from each evaluation run.
        mean: Mean across runs.
        std: Standard deviation across runs.
        conf_interval_95: 95% confidence interval half-width.
        direction: Whether "lower" or "higher" is better.
    """

    name: str
    values: list[float]
    mean: float
    std: float
    conf_interval_95: float
    direction: str  # "lower" or "higher"

    def __repr__(self) -> str:
        arrow = "↓" if self.direction == "lower" else "↑"
        return f"{self.name}: {self.mean:.4f} ± {self.conf_interval_95:.4f} {arrow}"


@dataclass
class SemanticEvaluationReport:
    """Complete semantic evaluation report.

    Args:
        metrics: Dictionary mapping metric names to SemanticMetricResult objects.
        num_replications: Number of evaluation runs performed.
        num_samples: Total number of motion samples evaluated.
        dataset_name: Name of the dataset used for evaluation.
        metadata: Additional metadata about the evaluation.
    """

    metrics: dict[str, SemanticMetricResult]
    num_replications: int
    num_samples: int
    dataset_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "metrics": {
                name: {
                    "mean": result.mean,
                    "std": result.std,
                    "conf_interval_95": result.conf_interval_95,
                    "values": result.values,
                    "direction": result.direction,
                }
                for name, result in self.metrics.items()
            },
            "num_replications": self.num_replications,
            "num_samples": self.num_samples,
            "dataset_name": self.dataset_name,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Return a formatted string summary of all metrics."""
        lines = [
            f"Semantic Evaluation Report ({self.dataset_name})",
            f"  Replications: {self.num_replications}, Samples: {self.num_samples}",
            "  " + "-" * 60,
        ]
        for name, result in self.metrics.items():
            lines.append(f"  {result}")
        return "\n".join(lines)


class SemanticEvaluator:
    """High-level evaluator for text-to-motion semantic metrics.

    Orchestrates the full evaluation pipeline: loading models, extracting
    embeddings, computing metrics, and aggregating results across replications.

    Args:
        dataset_name: Dataset name ("humanml" or "kit").
        checkpoint_dir: Path to directory containing T2M evaluator checkpoints.
        glove_dir: Path to directory containing GloVe data files.
        device: Compute device (default "cpu").
        batch_size: Batch size for embedding extraction (default 32, matches
            standard T2M protocol where R-Precision operates on batch-sized
            retrieval pools).
        num_replications: Number of evaluation runs for CI estimation (default 20).
        seed: Random seed for reproducibility (default 0).

    Example:
        >>> evaluator = SemanticEvaluator(
        ...     dataset_name="humanml",
        ...     checkpoint_dir="./checkpoints",
        ...     glove_dir="./glove",
        ...     device="cuda",
        ... )
        >>> report = evaluator.evaluate(
        ...     gen_motions=generated_motions,       # (N, T, 263)
        ...     gen_m_lens=generated_lengths,         # (N,)
        ...     gt_motions=ground_truth_motions,      # (N, T, 263)
        ...     gt_m_lens=ground_truth_lengths,       # (N,)
        ...     texts=text_descriptions,              # list of N strings
        ... )
        >>> print(report.summary())
    """

    def __init__(
        self,
        dataset_name: str = "humanml",
        checkpoint_dir: str = "./checkpoints",
        glove_dir: str = "./glove",
        device: str | torch.device = "cpu",
        batch_size: int = 32,
        num_replications: int = 20,
        seed: int = 0,
        # Future extension: accept a custom evaluator_wrapper instance
        # evaluator_wrapper: BaseEvaluatorWrapper | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = batch_size
        self.num_replications = num_replications
        self.seed = seed

        # Load the evaluator wrapper (pre-trained networks)
        # Future extension: allow passing a custom BaseEvaluatorWrapper instance
        # to support alternative evaluator architectures
        self.eval_wrapper = T2MEvaluatorWrapper(
            dataset_name=dataset_name,
            checkpoint_dir=checkpoint_dir,
            device=str(self.device),
        )

        # Load GloVe vectorizer for text encoding
        self.word_vectorizer = WordVectorizer(glove_dir, prefix="our_vab")
        print(f"Loaded GloVe vectorizer ({len(self.word_vectorizer)} words)")

    def evaluate(
        self,
        gen_motions: np.ndarray | torch.Tensor,
        gen_m_lens: np.ndarray | torch.Tensor,
        gt_motions: np.ndarray | torch.Tensor,
        gt_m_lens: np.ndarray | torch.Tensor,
        texts: list[str],
        compute_multimodality: bool = False,
        mm_motions: np.ndarray | torch.Tensor | None = None,
        mm_m_lens: np.ndarray | torch.Tensor | None = None,
        mm_num_repeats: int = 30,
        mm_num_times: int = 10,
        diversity_times: int = 300,
    ) -> SemanticEvaluationReport:
        """Run the full semantic evaluation pipeline.

        Args:
            gen_motions: Generated motions, shape (N, T, 263) for HumanML3D.
            gen_m_lens: Generated motion lengths in frames, shape (N,).
            gt_motions: Ground-truth motions, shape (N, T, 263).
            gt_m_lens: Ground-truth motion lengths in frames, shape (N,).
            texts: Text descriptions, list of N strings.
            compute_multimodality: Whether to compute multimodality metric.
            mm_motions: Multiple generated motions for multimodality,
                shape (N_mm, num_repeats, T, 263). Required if compute_multimodality=True.
            mm_m_lens: Motion lengths for mm_motions, shape (N_mm, num_repeats).
            mm_num_repeats: Number of repeats per prompt for multimodality.
            mm_num_times: Number of random pairs for multimodality sampling.
            diversity_times: Number of random pairs for diversity sampling.

        Returns:
            SemanticEvaluationReport with aggregated metrics.
        """
        # Convert to tensors if needed
        gen_motions = self._to_tensor(gen_motions)
        gen_m_lens = self._to_long_tensor(gen_m_lens)
        gt_motions = self._to_tensor(gt_motions)
        gt_m_lens = self._to_long_tensor(gt_m_lens)

        num_samples = len(texts)
        assert gen_motions.shape[0] == num_samples, (
            f"Number of generated motions ({gen_motions.shape[0]}) doesn't match number of texts ({num_samples})"
        )
        assert gt_motions.shape[0] == num_samples

        # Pre-compute text embeddings (only needs to happen once)
        word_embs, pos_ohots, cap_lens = texts_to_embeddings(texts, self.word_vectorizer)
        word_embs_t = torch.from_numpy(word_embs).float()
        pos_ohots_t = torch.from_numpy(pos_ohots).float()
        cap_lens_t = torch.from_numpy(cap_lens).long()

        # Run replications
        all_metrics: dict[str, list[float]] = {
            "FID": [],
            "matching_score": [],
            "R_precision_top1": [],
            "R_precision_top2": [],
            "R_precision_top3": [],
            "diversity": [],
            "diversity_gt": [],
        }
        if compute_multimodality:
            all_metrics["multimodality"] = []

        for rep in range(self.num_replications):
            np.random.seed(self.seed + rep)
            torch.manual_seed(self.seed + rep)

            result = self._evaluate_single_run(
                gen_motions=gen_motions,
                gen_m_lens=gen_m_lens,
                gt_motions=gt_motions,
                gt_m_lens=gt_m_lens,
                word_embs=word_embs_t,
                pos_ohots=pos_ohots_t,
                cap_lens=cap_lens_t,
                diversity_times=diversity_times,
            )

            all_metrics["FID"].append(result["fid"])
            all_metrics["matching_score"].append(result["matching_score"])
            all_metrics["R_precision_top1"].append(result["r_precision"][0])
            all_metrics["R_precision_top2"].append(result["r_precision"][1])
            all_metrics["R_precision_top3"].append(result["r_precision"][2])
            all_metrics["diversity"].append(result["diversity"])
            all_metrics["diversity_gt"].append(result["diversity_gt"])

            print(
                f"  Replication {rep + 1}/{self.num_replications}: "
                f"FID={result['fid']:.4f}, "
                f"R@1={result['r_precision'][0]:.4f}, "
                f"MS={result['matching_score']:.4f}, "
                f"Div={result['diversity']:.4f}"
            )

        # Compute multimodality if requested
        if compute_multimodality and mm_motions is not None:
            mm_result = self._compute_multimodality(mm_motions, mm_m_lens, mm_num_repeats, mm_num_times)
            all_metrics["multimodality"] = mm_result

        # Aggregate results
        metric_directions = {
            "FID": "lower",
            "matching_score": "lower",
            "R_precision_top1": "higher",
            "R_precision_top2": "higher",
            "R_precision_top3": "higher",
            "diversity": "closer_to_gt",  # Special: closer to GT diversity is better
            "diversity_gt": "higher",
            "multimodality": "higher",
        }

        report_metrics = {}
        for name, values in all_metrics.items():
            if len(values) == 0:
                continue
            values_arr = np.array(values)
            mean = float(np.mean(values_arr))
            std = float(np.std(values_arr))
            # 95% CI = 1.96 * std / sqrt(n)
            conf_95 = 1.96 * std / np.sqrt(len(values_arr)) if len(values_arr) > 1 else 0.0

            direction = metric_directions.get(name, "lower")
            if direction == "closer_to_gt":
                direction = "higher"  # For display purposes

            report_metrics[name] = SemanticMetricResult(
                name=name,
                values=[float(v) for v in values],
                mean=mean,
                std=std,
                conf_interval_95=float(conf_95),
                direction=direction,
            )

        return SemanticEvaluationReport(
            metrics=report_metrics,
            num_replications=self.num_replications,
            num_samples=num_samples,
            dataset_name=self.dataset_name,
            metadata={
                "batch_size": self.batch_size,
                "seed": self.seed,
                "diversity_times": diversity_times,
                "compute_multimodality": compute_multimodality,
            },
        )

    def _evaluate_single_run(
        self,
        gen_motions: torch.Tensor,
        gen_m_lens: torch.Tensor,
        gt_motions: torch.Tensor,
        gt_m_lens: torch.Tensor,
        word_embs: torch.Tensor,
        pos_ohots: torch.Tensor,
        cap_lens: torch.Tensor,
        diversity_times: int = 300,
    ) -> dict[str, Any]:
        """Run a single evaluation pass over all data.

        Returns dict with: fid, matching_score, r_precision, diversity, diversity_gt.
        """
        num_samples = gen_motions.shape[0]

        # Accumulate embeddings and scores across batches
        all_gt_motion_embs = []
        all_gen_motion_embs = []
        r_precision_sum = np.zeros(3)
        matching_score_sum = 0.0
        num_processed = 0

        for start in range(0, num_samples, self.batch_size):
            end = min(start + self.batch_size, num_samples)
            batch_size = end - start

            # Get batch data
            batch_word_embs = word_embs[start:end]
            batch_pos_ohots = pos_ohots[start:end]
            batch_cap_lens = cap_lens[start:end]
            batch_gt_motions = gt_motions[start:end]
            batch_gt_m_lens = gt_m_lens[start:end]
            batch_gen_motions = gen_motions[start:end]
            batch_gen_m_lens = gen_m_lens[start:end]

            # Extract GT embeddings
            et_gt, em_gt = self.eval_wrapper.get_co_embeddings(
                batch_word_embs,
                batch_pos_ohots,
                batch_cap_lens,
                batch_gt_motions,
                batch_gt_m_lens,
            )

            # Extract generated embeddings
            et_gen, em_gen = self.eval_wrapper.get_co_embeddings(
                batch_word_embs,
                batch_pos_ohots,
                batch_cap_lens,
                batch_gen_motions,
                batch_gen_m_lens,
            )

            all_gt_motion_embs.append(em_gt.cpu().numpy())
            all_gen_motion_embs.append(em_gen.cpu().numpy())

            # Batch R-Precision and Matching Score (on generated)
            et_gen_np = et_gen.cpu().numpy()
            em_gen_np = em_gen.cpu().numpy()

            batch_r_precision = calculate_R_precision(et_gen_np, em_gen_np, top_k=3, sum_all=True)
            r_precision_sum += batch_r_precision

            batch_matching = euclidean_distance_matrix(et_gen_np, em_gen_np).trace()
            matching_score_sum += batch_matching

            num_processed += batch_size

        # Concatenate all embeddings
        all_gt_motion_embs_np = np.concatenate(all_gt_motion_embs, axis=0)
        all_gen_motion_embs_np = np.concatenate(all_gen_motion_embs, axis=0)

        # Compute FID
        gt_mu, gt_cov = calculate_activation_statistics(all_gt_motion_embs_np)
        gen_mu, gen_cov = calculate_activation_statistics(all_gen_motion_embs_np)
        fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)

        # Normalize accumulated metrics
        r_precision = r_precision_sum / num_processed
        matching_score = matching_score_sum / num_processed

        # Diversity
        div_times = min(diversity_times, num_processed - 1)
        diversity_gt = calculate_diversity(all_gt_motion_embs_np, div_times)
        diversity_gen = calculate_diversity(all_gen_motion_embs_np, div_times)

        return {
            "fid": fid,
            "matching_score": matching_score,
            "r_precision": r_precision,
            "diversity": diversity_gen,
            "diversity_gt": diversity_gt,
        }

    def _compute_multimodality(
        self,
        mm_motions: np.ndarray | torch.Tensor,
        mm_m_lens: np.ndarray | torch.Tensor | None,
        mm_num_repeats: int,
        mm_num_times: int,
    ) -> list[float]:
        """Compute multimodality across replications.

        Args:
            mm_motions: shape (num_prompts, num_repeats, T, motion_dim)
            mm_m_lens: shape (num_prompts, num_repeats)
            mm_num_repeats: Number of repeats to use
            mm_num_times: Number of pairs to sample

        Returns:
            List of multimodality values, one per replication.
        """
        mm_motions_t = self._to_tensor(mm_motions)
        num_prompts, num_repeats = mm_motions_t.shape[:2]

        if mm_m_lens is None:
            # Assume all motions are full length
            seq_len = mm_motions_t.shape[2]
            mm_m_lens_t = torch.full((num_prompts, num_repeats), seq_len, dtype=torch.long)
        else:
            mm_m_lens_t = self._to_long_tensor(mm_m_lens)

        # Extract embeddings for each prompt's repeats
        all_mm_embeddings = []
        for i in range(num_prompts):
            prompt_motions = mm_motions_t[i]  # (num_repeats, T, D)
            prompt_lens = mm_m_lens_t[i]  # (num_repeats,)
            emb = self.eval_wrapper.get_motion_embeddings(prompt_motions, prompt_lens)
            all_mm_embeddings.append(emb.cpu().numpy())

        # Shape: (num_prompts, num_repeats, embed_dim)
        mm_emb_array = np.stack(all_mm_embeddings, axis=0)

        # Compute multimodality across replications
        mm_values = []
        for rep in range(self.num_replications):
            np.random.seed(self.seed + rep + 10000)
            mm_val = calculate_multimodality(mm_emb_array, mm_num_times)
            mm_values.append(mm_val)

        return mm_values

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert to float tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float()

    @staticmethod
    def _to_long_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert to long tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).long()
        return x.long()
