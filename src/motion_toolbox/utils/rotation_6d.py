"""6D rotation representation (Zhou et al., CVPR 2019).

This module implements the 6D continuous rotation representation which provides
better continuity properties than other representations like Euler angles or
quaternions for neural network training.

Reference:
    Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
    (CVPR 2019) http://arxiv.org/abs/1812.07035
"""

import numpy as np
import torch


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation.

    Takes the first 2 columns of the 3x3 rotation matrix as a compact
    representation. The third column can be reconstructed via cross product.

    Args:
        matrix: [..., 3, 3] rotation matrices

    Returns:
        d6: [..., 6] 6D representation (first 2 columns flattened)

    Examples:
        >>> import torch
        >>> # Identity rotation
        >>> R = torch.eye(3).unsqueeze(0)  # [1, 3, 3]
        >>> d6 = matrix_to_rotation_6d(R)
        >>> d6.shape
        torch.Size([1, 6])
        >>> # First column [1, 0, 0], second column [0, 1, 0]
        >>> torch.allclose(d6, torch.tensor([[1., 0., 0., 0., 1., 0.]]))
        True

    Reference:
        Zhou et al. "On the Continuity of Rotation Representations
        in Neural Networks" (CVPR 2019)
        http://arxiv.org/abs/1812.07035
    """
    # Extract first 2 columns and flatten
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D representation to rotation matrix.

    Uses Gram-Schmidt orthogonalization to ensure the output is a valid
    rotation matrix (orthonormal columns, determinant = 1).

    Args:
        d6: [..., 6] 6D representation (first 2 columns of rotation matrix)

    Returns:
        matrix: [..., 3, 3] rotation matrices

    Examples:
        >>> import torch
        >>> # 6D representation of identity rotation
        >>> d6 = torch.tensor([[1., 0., 0., 0., 1., 0.]])
        >>> R = rotation_6d_to_matrix(d6)
        >>> R.shape
        torch.Size([1, 3, 3])
        >>> torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)
        True

        >>> # Round-trip test
        >>> R_orig = torch.randn(5, 3, 3)
        >>> # Make it a valid rotation matrix via QR decomposition
        >>> Q, _ = torch.linalg.qr(R_orig)
        >>> d6 = matrix_to_rotation_6d(Q)
        >>> R_reconstructed = rotation_6d_to_matrix(d6)
        >>> torch.allclose(Q, R_reconstructed, atol=1e-5)
        True
    """
    # Reshape to get 2 column vectors
    a1 = d6[..., :3]  # First column
    a2 = d6[..., 3:]  # Second column

    # Gram-Schmidt orthogonalization
    # Normalize first vector
    b1 = torch.nn.functional.normalize(a1, dim=-1)

    # Make second vector orthogonal to first
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)

    # Third vector via cross product (ensures right-handed coordinate system)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as ROWS (dim=-2), not columns
    # This matches StableMotion's convention
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """NumPy version of rotation_6d_to_matrix.

    Args:
        d6: [..., 6] 6D representation

    Returns:
        matrix: [..., 3, 3] rotation matrices
    """
    d6_torch = torch.from_numpy(d6).float()
    matrix_torch = rotation_6d_to_matrix(d6_torch)
    return matrix_torch.numpy()


def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    """NumPy version of matrix_to_rotation_6d.

    Args:
        matrix: [..., 3, 3] rotation matrices

    Returns:
        d6: [..., 6] 6D representation
    """
    matrix_torch = torch.from_numpy(matrix).float()
    d6_torch = matrix_to_rotation_6d(matrix_torch)
    return d6_torch.numpy()
