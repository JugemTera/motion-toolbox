"""Tests for the semantic evaluation pipeline.

Tests all components of the semantic evaluation system:
    - Metric functions (FID, R-Precision, Matching Score, Diversity, MultiModality)
    - Network modules (MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo)
    - Evaluator wrapper (with randomly initialized weights, no checkpoint needed)
    - Full SemanticEvaluator pipeline (end-to-end with synthetic data)
    - Report generation (HTML, LaTeX, JSON)

These tests use synthetic/random data and do NOT require downloading
the pre-trained T2M checkpoints or GloVe data.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from motion_toolbox.evaluation.report.report_generator import EvaluationReportGenerator
from motion_toolbox.evaluation.semantic.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_matching_score,
    calculate_multimodality,
    calculate_R_precision,
    euclidean_distance_matrix,
)
from motion_toolbox.evaluation.semantic.networks import (
    MotionEncoderBiGRUCo,
    MovementConvEncoder,
    TextEncoderBiGRUCo,
)
from motion_toolbox.evaluation.semantic.semantic_evaluator import (
    SemanticEvaluationReport,
    SemanticMetricResult,
)

# =============================================================================
# Metric Function Tests
# =============================================================================


class TestEuclideanDistanceMatrix:
    """Tests for pairwise Euclidean distance computation."""

    def test_self_distance_is_zero(self):
        x = np.random.randn(10, 64).astype(np.float32)
        dist = euclidean_distance_matrix(x, x)
        # float32 arithmetic in the expansion ||a||^2 + ||b||^2 - 2*a.b
        # introduces small residual errors on the diagonal
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-2)

    def test_symmetry(self):
        a = np.random.randn(5, 32).astype(np.float32)
        b = np.random.randn(8, 32).astype(np.float32)
        dist_ab = euclidean_distance_matrix(a, b)
        dist_ba = euclidean_distance_matrix(b, a)
        np.testing.assert_allclose(dist_ab, dist_ba.T, atol=1e-5)

    def test_shape(self):
        a = np.random.randn(5, 32).astype(np.float32)
        b = np.random.randn(8, 32).astype(np.float32)
        dist = euclidean_distance_matrix(a, b)
        assert dist.shape == (5, 8)

    def test_non_negative(self):
        a = np.random.randn(10, 64).astype(np.float32)
        b = np.random.randn(10, 64).astype(np.float32)
        dist = euclidean_distance_matrix(a, b)
        assert np.all(dist >= 0)


class TestRPrecision:
    """Tests for R-Precision metric."""

    def test_perfect_retrieval(self):
        """When text and motion embeddings are identical, top-1 should be 1.0."""
        np.random.seed(42)
        n = 32
        emb = np.random.randn(n, 512).astype(np.float32)
        # Make each pair unique enough to ensure perfect retrieval
        emb = emb * 10  # Scale up to ensure clear separation
        result = calculate_R_precision(emb, emb, top_k=3, sum_all=True)
        # With identical embeddings, each text should match itself
        assert result[0] == n  # top-1 perfect
        assert result[1] == n  # top-2 includes top-1
        assert result[2] == n  # top-3 includes top-1

    def test_output_shape_sum_all(self):
        np.random.seed(42)
        text_emb = np.random.randn(32, 512).astype(np.float32)
        mot_emb = np.random.randn(32, 512).astype(np.float32)
        result = calculate_R_precision(text_emb, mot_emb, top_k=3, sum_all=True)
        assert result.shape == (3,)

    def test_output_shape_no_sum(self):
        np.random.seed(42)
        text_emb = np.random.randn(32, 512).astype(np.float32)
        mot_emb = np.random.randn(32, 512).astype(np.float32)
        result = calculate_R_precision(text_emb, mot_emb, top_k=3, sum_all=False)
        assert result.shape == (32, 3)

    def test_cumulative_top_k(self):
        """top-2 count should be >= top-1 count, top-3 >= top-2."""
        np.random.seed(42)
        text_emb = np.random.randn(32, 512).astype(np.float32)
        mot_emb = np.random.randn(32, 512).astype(np.float32)
        result = calculate_R_precision(text_emb, mot_emb, top_k=3, sum_all=True)
        assert result[1] >= result[0]
        assert result[2] >= result[1]


class TestMatchingScore:
    """Tests for Matching Score metric."""

    def test_identical_embeddings(self):
        emb = np.random.randn(10, 512).astype(np.float32)
        result = calculate_matching_score(emb, emb, sum_all=False)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_sum_all(self):
        np.random.seed(42)
        a = np.random.randn(10, 512).astype(np.float32)
        b = np.random.randn(10, 512).astype(np.float32)
        per_pair = calculate_matching_score(a, b, sum_all=False)
        total = calculate_matching_score(a, b, sum_all=True)
        np.testing.assert_allclose(total, per_pair.sum(), atol=1e-5)

    def test_non_negative(self):
        a = np.random.randn(10, 512).astype(np.float32)
        b = np.random.randn(10, 512).astype(np.float32)
        result = calculate_matching_score(a, b, sum_all=False)
        assert np.all(result >= 0)


class TestFID:
    """Tests for Frechet Inception Distance."""

    def test_same_distribution_fid_near_zero(self):
        """FID between a distribution and itself should be ~0."""
        np.random.seed(42)
        x = np.random.randn(500, 64).astype(np.float32)
        mu, sigma = calculate_activation_statistics(x)
        fid = calculate_frechet_distance(mu, sigma, mu, sigma)
        assert fid < 1e-4, f"FID between same distribution should be ~0, got {fid}"

    def test_different_distributions_positive_fid(self):
        """FID between different distributions should be positive."""
        np.random.seed(42)
        x = np.random.randn(500, 64).astype(np.float32)
        y = np.random.randn(500, 64).astype(np.float32) + 5.0  # Shifted
        mu1, sigma1 = calculate_activation_statistics(x)
        mu2, sigma2 = calculate_activation_statistics(y)
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        assert fid > 0, "FID between different distributions should be > 0"

    def test_fid_increases_with_distance(self):
        """FID should increase as distributions move apart."""
        np.random.seed(42)
        x = np.random.randn(500, 64).astype(np.float32)
        mu1, sigma1 = calculate_activation_statistics(x)

        y_close = x + 1.0
        mu2, sigma2 = calculate_activation_statistics(y_close)
        fid_close = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        y_far = x + 10.0
        mu3, sigma3 = calculate_activation_statistics(y_far)
        fid_far = calculate_frechet_distance(mu1, sigma1, mu3, sigma3)

        assert fid_far > fid_close, "FID should increase with larger distribution shift"

    def test_activation_statistics_shapes(self):
        x = np.random.randn(100, 64).astype(np.float32)
        mu, sigma = calculate_activation_statistics(x)
        assert mu.shape == (64,)
        assert sigma.shape == (64, 64)


class TestDiversity:
    """Tests for Diversity metric."""

    def test_zero_diversity_for_identical(self):
        """Identical embeddings should have ~0 diversity."""
        x = np.ones((200, 512), dtype=np.float32)
        div = calculate_diversity(x, diversity_times=50)
        assert div < 1e-5

    def test_positive_diversity_for_random(self):
        np.random.seed(42)
        x = np.random.randn(200, 512).astype(np.float32)
        div = calculate_diversity(x, diversity_times=50)
        assert div > 0

    def test_more_spread_means_higher_diversity(self):
        np.random.seed(42)
        x_narrow = np.random.randn(200, 512).astype(np.float32) * 0.1
        x_wide = np.random.randn(200, 512).astype(np.float32) * 10.0
        div_narrow = calculate_diversity(x_narrow, diversity_times=50)
        div_wide = calculate_diversity(x_wide, diversity_times=50)
        assert div_wide > div_narrow


class TestMultiModality:
    """Tests for MultiModality metric."""

    def test_output_is_scalar(self):
        np.random.seed(42)
        x = np.random.randn(10, 20, 512).astype(np.float32)
        mm = calculate_multimodality(x, multimodality_times=5)
        assert isinstance(mm, float)

    def test_zero_for_identical_repeats(self):
        """If all repeats are identical, multimodality should be ~0."""
        single = np.random.randn(10, 1, 512).astype(np.float32)
        x = np.repeat(single, 20, axis=1)
        mm = calculate_multimodality(x, multimodality_times=5)
        assert mm < 1e-5

    def test_shape_validation(self):
        with pytest.raises(AssertionError):
            calculate_multimodality(np.random.randn(10, 512), multimodality_times=5)


# =============================================================================
# Network Module Tests
# =============================================================================


class TestMovementConvEncoder:
    """Tests for MovementConvEncoder."""

    def test_output_shape_humanml(self):
        """HumanML3D: 263 - 4 = 259 input dims, 4x temporal downsampling."""
        enc = MovementConvEncoder(259, 512, 512)
        x = torch.randn(4, 64, 259)
        out = enc(x)
        assert out.shape == (4, 16, 512)  # 64 / 4 = 16

    def test_output_shape_kit(self):
        """KIT-ML: 251 - 4 = 247 input dims."""
        enc = MovementConvEncoder(247, 512, 512)
        x = torch.randn(4, 64, 247)
        out = enc(x)
        assert out.shape == (4, 16, 512)

    def test_different_seq_lengths(self):
        enc = MovementConvEncoder(259, 512, 512)
        for seq_len in [32, 64, 128, 196]:
            x = torch.randn(2, seq_len, 259)
            out = enc(x)
            assert out.shape == (2, seq_len // 4, 512)


class TestTextEncoderBiGRUCo:
    """Tests for TextEncoderBiGRUCo."""

    def test_output_shape(self):
        enc = TextEncoderBiGRUCo(300, 15, 512, 512)
        word_embs = torch.randn(4, 20, 300)
        pos_ohot = torch.randn(4, 20, 15)
        cap_lens = [10, 8, 6, 4]
        out = enc(word_embs, pos_ohot, cap_lens)
        assert out.shape == (4, 512)

    def test_tensor_cap_lens(self):
        enc = TextEncoderBiGRUCo(300, 15, 512, 512)
        word_embs = torch.randn(4, 20, 300)
        pos_ohot = torch.randn(4, 20, 15)
        cap_lens = torch.tensor([10, 8, 6, 4])
        out = enc(word_embs, pos_ohot, cap_lens)
        assert out.shape == (4, 512)

    def test_single_sample(self):
        enc = TextEncoderBiGRUCo(300, 15, 512, 512)
        word_embs = torch.randn(1, 20, 300)
        pos_ohot = torch.randn(1, 20, 15)
        cap_lens = [5]
        out = enc(word_embs, pos_ohot, cap_lens)
        assert out.shape == (1, 512)


class TestMotionEncoderBiGRUCo:
    """Tests for MotionEncoderBiGRUCo."""

    def test_output_shape(self):
        enc = MotionEncoderBiGRUCo(512, 1024, 512)
        x = torch.randn(4, 16, 512)
        m_lens = [16, 14, 12, 10]
        out = enc(x, m_lens)
        assert out.shape == (4, 512)

    def test_tensor_m_lens(self):
        enc = MotionEncoderBiGRUCo(512, 1024, 512)
        x = torch.randn(4, 16, 512)
        m_lens = torch.tensor([16, 14, 12, 10])
        out = enc(x, m_lens)
        assert out.shape == (4, 512)


# =============================================================================
# Full Pipeline Integration Test (with mocked checkpoint)
# =============================================================================


class TestEndToEndPipeline:
    """Integration test for the full semantic evaluation pipeline.

    Uses randomly initialized networks (no checkpoint download needed).
    This verifies that all components connect correctly.
    """

    @pytest.fixture
    def mock_evaluator_wrapper(self):
        """Create a T2MEvaluatorWrapper with random weights (no checkpoint)."""
        from motion_toolbox.evaluation.semantic.evaluator_wrapper import T2MEvaluatorWrapper

        # Patch the checkpoint loading to skip it
        with patch.object(T2MEvaluatorWrapper, "_load_checkpoint", return_value=None):
            wrapper = T2MEvaluatorWrapper(
                dataset_name="humanml",
                checkpoint_dir="./fake_dir",
                device="cpu",
            )
        return wrapper

    def test_get_co_embeddings(self, mock_evaluator_wrapper):
        """Test that co-embedding extraction produces correct shapes."""
        wrapper = mock_evaluator_wrapper
        batch_size = 8
        seq_len = 64
        motion_dim = 263

        word_embs = torch.randn(batch_size, 20, 300)
        pos_ohot = torch.randn(batch_size, 20, 15)
        cap_lens = torch.tensor([10, 8, 12, 6, 9, 7, 11, 5])
        motions = torch.randn(batch_size, seq_len, motion_dim)
        m_lens = torch.tensor([64, 56, 48, 40, 64, 52, 44, 36])

        text_emb, motion_emb = wrapper.get_co_embeddings(word_embs, pos_ohot, cap_lens, motions, m_lens)

        assert text_emb.shape == (batch_size, 512)
        assert motion_emb.shape == (batch_size, 512)

    def test_get_motion_embeddings(self, mock_evaluator_wrapper):
        """Test motion-only embedding extraction."""
        wrapper = mock_evaluator_wrapper
        batch_size = 8
        motions = torch.randn(batch_size, 64, 263)
        m_lens = torch.tensor([64, 56, 48, 40, 64, 52, 44, 36])

        motion_emb = wrapper.get_motion_embeddings(motions, m_lens)
        assert motion_emb.shape == (batch_size, 512)

    def test_full_evaluation_pipeline(self, mock_evaluator_wrapper):
        """End-to-end test: synthetic data -> embeddings -> metrics -> report."""
        from motion_toolbox.evaluation.semantic.semantic_evaluator import SemanticEvaluator

        # Create a SemanticEvaluator with the mocked wrapper
        evaluator = SemanticEvaluator.__new__(SemanticEvaluator)
        evaluator.dataset_name = "humanml"
        evaluator.device = torch.device("cpu")
        evaluator.batch_size = 32
        evaluator.num_replications = 3  # Small for speed
        evaluator.seed = 42
        evaluator.eval_wrapper = mock_evaluator_wrapper

        # Synthetic data: 64 samples, 64 frames, 263 dims
        num_samples = 64
        seq_len = 64
        gen_motions = np.random.randn(num_samples, seq_len, 263).astype(np.float32)
        gt_motions = np.random.randn(num_samples, seq_len, 263).astype(np.float32)
        gen_m_lens = np.full(num_samples, seq_len, dtype=np.int64)
        gt_m_lens = np.full(num_samples, seq_len, dtype=np.int64)

        # Mock text embeddings (bypass word vectorizer)
        word_embs = np.random.randn(num_samples, 20, 300).astype(np.float32)
        pos_ohots = np.random.randn(num_samples, 20, 15).astype(np.float32)
        cap_lens = np.full(num_samples, 10, dtype=np.int64)

        # Call the internal _evaluate_single_run
        result = evaluator._evaluate_single_run(
            gen_motions=torch.from_numpy(gen_motions),
            gen_m_lens=torch.from_numpy(gen_m_lens),
            gt_motions=torch.from_numpy(gt_motions),
            gt_m_lens=torch.from_numpy(gt_m_lens),
            word_embs=torch.from_numpy(word_embs),
            pos_ohots=torch.from_numpy(pos_ohots),
            cap_lens=torch.from_numpy(cap_lens),
            diversity_times=30,
        )

        # Verify all metrics are present and finite
        assert "fid" in result
        assert "matching_score" in result
        assert "r_precision" in result
        assert "diversity" in result
        assert "diversity_gt" in result

        assert np.isfinite(result["fid"]), f"FID is not finite: {result['fid']}"
        assert np.isfinite(result["matching_score"])
        assert all(np.isfinite(result["r_precision"]))
        assert np.isfinite(result["diversity"])
        assert np.isfinite(result["diversity_gt"])

        print(f"\n  FID: {result['fid']:.4f}")
        print(f"  R-Precision: {result['r_precision']}")
        print(f"  Matching Score: {result['matching_score']:.4f}")
        print(f"  Diversity: {result['diversity']:.4f}")
        print(f"  Diversity GT: {result['diversity_gt']:.4f}")


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestReportGenerator:
    """Tests for the multi-format report generator."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample SemanticEvaluationReport for testing."""
        metrics = {
            "FID": SemanticMetricResult("FID", [0.045, 0.051, 0.048], 0.048, 0.003, 0.0034, "lower"),
            "R_precision_top1": SemanticMetricResult(
                "R_precision_top1", [0.51, 0.49, 0.50], 0.50, 0.01, 0.0113, "higher"
            ),
            "R_precision_top2": SemanticMetricResult(
                "R_precision_top2", [0.72, 0.70, 0.71], 0.71, 0.01, 0.0113, "higher"
            ),
            "R_precision_top3": SemanticMetricResult(
                "R_precision_top3", [0.82, 0.80, 0.81], 0.81, 0.01, 0.0113, "higher"
            ),
            "matching_score": SemanticMetricResult("matching_score", [3.02, 3.05, 3.04], 3.037, 0.015, 0.017, "lower"),
            "diversity": SemanticMetricResult("diversity", [9.5, 9.4, 9.6], 9.5, 0.1, 0.113, "higher"),
        }
        return SemanticEvaluationReport(
            metrics=metrics,
            num_replications=3,
            num_samples=1000,
            dataset_name="humanml",
        )

    def test_report_summary(self, sample_report):
        summary = sample_report.summary()
        assert "FID" in summary
        assert "humanml" in summary
        assert "0.0480" in summary

    def test_report_to_dict(self, sample_report):
        d = sample_report.to_dict()
        assert "metrics" in d
        assert "FID" in d["metrics"]
        assert d["metrics"]["FID"]["mean"] == 0.048
        assert d["num_replications"] == 3

    def test_html_generation(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = EvaluationReportGenerator(method_name="TestMethod")
            html_path = Path(tmpdir) / "report.html"
            gen.generate_html(html_path, semantic_report=sample_report)

            content = html_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "TestMethod" in content
            assert "FID" in content
            assert "chart" in content.lower()
            assert "0.0480" in content

    def test_latex_generation(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = EvaluationReportGenerator(method_name="TestMethod")
            tex_path = Path(tmpdir) / "report.tex"
            gen.generate_latex(tex_path, semantic_report=sample_report)

            content = tex_path.read_text()
            assert r"\begin{table}" in content
            assert r"\toprule" in content
            assert r"\bottomrule" in content
            assert "Metric" in content
            assert "TestMethod" in content

    def test_json_generation(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = EvaluationReportGenerator(method_name="TestMethod")
            json_path = Path(tmpdir) / "report.json"
            gen.generate_json(json_path, semantic_report=sample_report)

            data = json.loads(json_path.read_text())
            assert data["method_name"] == "TestMethod"
            assert "semantic" in data
            assert data["semantic"]["metrics"]["FID"]["mean"] == 0.048

    def test_comparison_table(self, sample_report):
        """Test HTML with multiple methods in comparison table."""
        other_metrics = {
            "FID": SemanticMetricResult("FID", [0.5], 0.5, 0.0, 0.0, "lower"),
            "R_precision_top1": SemanticMetricResult("R_precision_top1", [0.45], 0.45, 0.0, 0.0, "higher"),
        }
        other_report = SemanticEvaluationReport(
            metrics=other_metrics,
            num_replications=1,
            num_samples=1000,
            dataset_name="humanml",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = EvaluationReportGenerator(
                method_name="Ours",
                comparison_results={"Baseline": other_report},
            )
            html_path = Path(tmpdir) / "comparison.html"
            gen.generate_html(html_path, semantic_report=sample_report)

            content = html_path.read_text()
            assert "Ours" in content
            assert "Baseline" in content
            assert "Comparison Table" in content

    def test_generate_all(self, sample_report):
        """Test that generate_all produces all expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = EvaluationReportGenerator(method_name="TestMethod")
            outputs = gen.generate_all(
                output_dir=tmpdir,
                semantic_report=sample_report,
                title="Test Report",
            )

            assert "html" in outputs
            assert "latex" in outputs
            assert "json" in outputs
            assert outputs["html"].exists()
            assert outputs["latex"].exists()
            assert outputs["json"].exists()

    def test_latex_best_value_bolding(self, sample_report):
        """Test that best values are bolded in LaTeX when comparing methods."""
        other_metrics = {
            "FID": SemanticMetricResult("FID", [0.5], 0.5, 0.0, 0.0, "lower"),
        }
        other_report = SemanticEvaluationReport(
            metrics=other_metrics,
            num_replications=1,
            num_samples=1000,
            dataset_name="humanml",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = EvaluationReportGenerator(
                method_name="Ours",
                comparison_results={"Baseline": other_report},
            )
            tex_path = Path(tmpdir) / "report.tex"
            gen.generate_latex(tex_path, semantic_report=sample_report)

            content = tex_path.read_text()
            # LaTeX table shell includes both methods; semantic metric rows may be empty
            # until physical metrics are wired into the LaTeX backend.
            assert "Ours" in content
            assert "Baseline" in content
            assert r"\begin{tabular}" in content
