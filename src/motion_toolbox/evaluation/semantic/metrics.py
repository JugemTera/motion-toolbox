"""Pure numpy/scipy implementations of semantic evaluation metrics.

These metrics follow the standard T2M evaluation protocol used by MoMask, MDM,
T2M-GPT, and other text-to-motion generation works. The protocol originates
from Guo et al.'s "Generating Diverse and Natural 3D Human Motions from Text"
(CVPR 2022).

All functions operate on numpy arrays of embeddings (typically 512-d).

References:
    - MoMask: https://github.com/EricGuo5513/momask-codes/blob/main/utils/metrics.py
    - MDM: https://github.com/GuyTevet/motion-diffusion-model
    - T2M: https://github.com/EricGuo5513/text-to-motion
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def euclidean_distance_matrix(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between two sets of vectors.

    Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b

    Args:
        matrix1: First set of vectors, shape (N1, D).
        matrix2: Second set of vectors, shape (N2, D).

    Returns:
        Distance matrix of shape (N1, N2) where entry [i, j] is the
        Euclidean distance between matrix1[i] and matrix2[j].
    """
    assert matrix1.shape[1] == matrix2.shape[1], f"Dimension mismatch: {matrix1.shape[1]} vs {matrix2.shape[1]}"
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(np.maximum(d1 + d2 + d3, 0.0))
    return dists


def _calculate_top_k(sorted_indices: np.ndarray, top_k: int) -> np.ndarray:
    """Check if the ground-truth index appears in the top-k retrieved indices.

    For each row i, the ground-truth is assumed to be index i itself (diagonal).

    Args:
        sorted_indices: Argsorted distance matrix, shape (N, N).
        top_k: Maximum rank to check (e.g., 3 for top-1/2/3).

    Returns:
        Boolean matrix of shape (N, top_k) where entry [i, k] indicates
        whether item i's ground-truth was retrieved within the top-(k+1).
    """
    size = sorted_indices.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = sorted_indices == gt_mat
    correct_vec = np.zeros(size, dtype=bool)
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(
    text_embeddings: np.ndarray,
    motion_embeddings: np.ndarray,
    top_k: int = 3,
    sum_all: bool = False,
) -> np.ndarray:
    """Compute R-Precision (text-to-motion retrieval accuracy) at top-1/2/3.

    For each text embedding, motions are ranked by L2 distance and we check
    whether the paired ground-truth motion appears within the top-k results.

    Args:
        text_embeddings: Text embeddings, shape (N, D).
        motion_embeddings: Motion embeddings, shape (N, D). Entry i corresponds
            to text_embeddings[i] (paired data).
        top_k: Number of top-k levels to evaluate (default 3 for top-1/2/3).
        sum_all: If True, return the sum of correct retrievals (for accumulation
            across batches). If False, return the boolean matrix.

    Returns:
        If sum_all: array of shape (top_k,) with counts of correct retrievals.
        If not sum_all: boolean matrix of shape (N, top_k).
    """
    dist_mat = euclidean_distance_matrix(text_embeddings, motion_embeddings)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = _calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(
    text_embeddings: np.ndarray,
    motion_embeddings: np.ndarray,
    sum_all: bool = False,
) -> np.ndarray | float:
    """Compute matching score (mean L2 distance between paired embeddings).

    Lower is better -- indicates tighter text-motion alignment.

    Args:
        text_embeddings: Text embeddings, shape (N, D).
        motion_embeddings: Motion embeddings, shape (N, D).
        sum_all: If True, return the sum (for accumulation across batches).
            If False, return per-pair distances.

    Returns:
        If sum_all: scalar sum of distances.
        If not sum_all: array of shape (N,) with per-pair distances.
    """
    assert text_embeddings.shape == motion_embeddings.shape
    dist = linalg.norm(text_embeddings - motion_embeddings, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist


def calculate_activation_statistics(
    activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of activation vectors (for FID).

    Args:
        activations: Embedding matrix, shape (N, D).

    Returns:
        Tuple of (mu, sigma) where mu has shape (D,) and sigma has shape (D, D).
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute the Frechet Distance between two multivariate Gaussians.

    The Frechet distance between X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    This is the standard FID metric used in motion generation evaluation.
    Implementation follows the numerically stable version by Dougal J. Sutherland.

    Args:
        mu1: Mean of the first Gaussian, shape (D,).
        sigma1: Covariance of the first Gaussian, shape (D, D).
        mu2: Mean of the second Gaussian, shape (D,).
        sigma2: Covariance of the second Gaussian, shape (D, D).
        eps: Small value for numerical stability.

    Returns:
        The Frechet Distance (scalar). Lower is better.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, f"Mean vectors have different lengths: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, (
        f"Covariance matrices have different dimensions: {sigma1.shape} vs {sigma2.shape}"
    )

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component in FID calculation: {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_diversity(
    activations: np.ndarray,
    diversity_times: int = 300,
) -> float:
    """Compute diversity of motion embeddings.

    Randomly samples pairs of motion embeddings and computes mean L2 distance.
    Higher diversity indicates more varied generated motions.

    Args:
        activations: Motion embeddings, shape (N, D).
        diversity_times: Number of random pairs to sample.

    Returns:
        Mean L2 distance between sampled pairs (scalar). Higher is better.
    """
    assert len(activations.shape) == 2
    assert activations.shape[0] > diversity_times, (
        f"Need more samples ({activations.shape[0]}) than diversity_times ({diversity_times})"
    )
    num_samples = activations.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activations[first_indices] - activations[second_indices], axis=1)
    return float(dist.mean())


def calculate_multimodality(
    activations: np.ndarray,
    multimodality_times: int = 10,
) -> float:
    """Compute multimodality of generated motions.

    Given multiple generations per text prompt, measures variation across
    different outputs for the same prompt. Higher indicates more diverse
    generation for each individual prompt.

    Args:
        activations: Motion embeddings grouped by text prompt,
            shape (num_prompts, num_repeats, D).
        multimodality_times: Number of random pairs to sample per prompt.

    Returns:
        Mean L2 distance across sampled pairs (scalar). Higher is better.
    """
    assert len(activations.shape) == 3, (
        f"Expected 3D array (num_prompts, num_repeats, D), got shape {activations.shape}"
    )
    assert activations.shape[1] > multimodality_times, (
        f"Need more repeats ({activations.shape[1]}) than multimodality_times ({multimodality_times})"
    )
    num_per_sent = activations.shape[1]

    first_indices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_indices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activations[:, first_indices] - activations[:, second_indices], axis=2)
    return float(dist.mean())
