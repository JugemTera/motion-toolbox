"""Euler angle utilities for ZYX decomposition.

This module provides utilities for working with Euler angles, specifically
the ZYX decomposition convention used in ALIGN representation.

Convention: R = Rz(θz) * Ry(θy) * Rx(θx)
"""

import numpy as np
import torch


def matrix_to_euler_zyx(matrix: torch.Tensor) -> torch.Tensor:
    """Extract ZYX Euler angles from rotation matrix.

    Decomposes a rotation matrix into Z, Y, X Euler angles following the
    convention: R = Rz(θz) * Ry(θy) * Rx(θx)

    Handles gimbal lock singularities gracefully.

    Args:
        matrix: [..., 3, 3] rotation matrices

    Returns:
        euler: [..., 3] angles [θz, θy, θx] in radians

    Examples:
        >>> import torch
        >>> # Identity rotation should give zero angles
        >>> R = torch.eye(3).unsqueeze(0)
        >>> angles = matrix_to_euler_zyx(R)
        >>> torch.allclose(angles, torch.zeros(1, 3), atol=1e-6)
        True

        >>> # 90 degree rotation around Z axis
        >>> theta = torch.tensor([torch.pi / 2])
        >>> Rz = axis_angle_rotation('Z', theta)
        >>> angles = matrix_to_euler_zyx(Rz)
        >>> torch.allclose(angles[0, 0], theta[0], atol=1e-6)  # θz should be π/2
        True
    """
    # Handle singularities with clipping
    sy = torch.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    singular = sy < 1e-6

    # Initialize output
    theta_x = torch.zeros_like(matrix[..., 0, 0])
    theta_y = torch.zeros_like(matrix[..., 0, 0])
    theta_z = torch.zeros_like(matrix[..., 0, 0])

    # Normal case (no gimbal lock)
    if not singular.all():
        theta_x = torch.where(
            singular,
            torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1]),
            torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2]),
        )
        theta_y = torch.where(singular, torch.atan2(-matrix[..., 2, 0], sy), torch.atan2(-matrix[..., 2, 0], sy))
        theta_z = torch.where(singular, torch.zeros_like(theta_z), torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0]))
    else:
        # All singular case
        theta_x = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
        theta_y = torch.atan2(-matrix[..., 2, 0], sy)
        theta_z = torch.zeros_like(theta_z)

    return torch.stack([theta_z, theta_y, theta_x], dim=-1)


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Create rotation matrix for rotation around a single axis.

    Generates a 3x3 rotation matrix for rotation by the specified angle
    around the X, Y, or Z axis.

    Args:
        axis: 'X', 'Y', or 'Z' - the axis to rotate around
        angle: [...] rotation angles in radians

    Returns:
        matrix: [..., 3, 3] rotation matrices

    Examples:
        >>> import torch
        >>> # 90 degree rotation around Z axis
        >>> angle = torch.tensor([torch.pi / 2])
        >>> Rz = axis_angle_rotation('Z', angle)
        >>> # Should rotate (1, 0, 0) to (0, 1, 0)
        >>> v = torch.tensor([1., 0., 0.])
        >>> v_rotated = torch.matmul(Rz[0], v)
        >>> torch.allclose(v_rotated, torch.tensor([0., 1., 0.]), atol=1e-6)
        True

        >>> # Identity rotation (0 degrees)
        >>> angle = torch.tensor([0.])
        >>> R = axis_angle_rotation('X', angle)
        >>> torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)
        True
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis.upper() == "X":
        # Rx = [1,  0,    0   ]
        #      [0,  cos, -sin ]
        #      [0,  sin,  cos ]
        R = torch.stack([one, zero, zero, zero, cos, -sin, zero, sin, cos], dim=-1).reshape(*angle.shape, 3, 3)
    elif axis.upper() == "Y":
        # Ry = [ cos, 0, sin ]
        #      [ 0,   1, 0   ]
        #      [-sin, 0, cos ]
        R = torch.stack([cos, zero, sin, zero, one, zero, -sin, zero, cos], dim=-1).reshape(*angle.shape, 3, 3)
    elif axis.upper() == "Z":
        # Rz = [ cos, -sin, 0 ]
        #      [ sin,  cos, 0 ]
        #      [ 0,    0,   1 ]
        R = torch.stack([cos, -sin, zero, sin, cos, zero, zero, zero, one], dim=-1).reshape(*angle.shape, 3, 3)
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'X', 'Y', or 'Z'")

    return R


def euler_zyx_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert ZYX Euler angles to rotation matrix.

    Constructs rotation matrix from Z, Y, X Euler angles:
    R = Rz(θz) * Ry(θy) * Rx(θx)

    Args:
        euler: [..., 3] angles [θz, θy, θx] in radians

    Returns:
        matrix: [..., 3, 3] rotation matrices

    Examples:
        >>> import torch
        >>> # Zero angles should give identity
        >>> euler = torch.zeros(1, 3)
        >>> R = euler_zyx_to_matrix(euler)
        >>> torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)
        True

        >>> # Round-trip test
        >>> angles_orig = torch.tensor([[0.5, 0.3, 0.7]])
        >>> R = euler_zyx_to_matrix(angles_orig)
        >>> angles_recovered = matrix_to_euler_zyx(R)
        >>> torch.allclose(angles_orig, angles_recovered, atol=1e-5)
        True
    """
    theta_z = euler[..., 0]
    theta_y = euler[..., 1]
    theta_x = euler[..., 2]

    Rz = axis_angle_rotation("Z", theta_z)
    Ry = axis_angle_rotation("Y", theta_y)
    Rx = axis_angle_rotation("X", theta_x)

    # Matrix multiplication: R = Rz * Ry * Rx
    R = torch.matmul(Rz, torch.matmul(Ry, Rx))

    return R


def matrix_to_euler_zyx_np(matrix: np.ndarray) -> np.ndarray:
    """NumPy version of matrix_to_euler_zyx.

    Args:
        matrix: [..., 3, 3] rotation matrices

    Returns:
        euler: [..., 3] angles [θz, θy, θx] in radians
    """
    matrix_torch = torch.from_numpy(matrix).float()
    euler_torch = matrix_to_euler_zyx(matrix_torch)
    return euler_torch.numpy()


def axis_angle_rotation_np(axis: str, angle: np.ndarray) -> np.ndarray:
    """NumPy version of axis_angle_rotation.

    Args:
        axis: 'X', 'Y', or 'Z'
        angle: [...] rotation angles in radians

    Returns:
        matrix: [..., 3, 3] rotation matrices
    """
    angle_torch = torch.from_numpy(angle).float()
    matrix_torch = axis_angle_rotation(axis, angle_torch)
    return matrix_torch.numpy()


def euler_zyx_to_matrix_np(euler: np.ndarray) -> np.ndarray:
    """NumPy version of euler_zyx_to_matrix.

    Args:
        euler: [..., 3] angles [θz, θy, θx] in radians

    Returns:
        matrix: [..., 3, 3] rotation matrices
    """
    euler_torch = torch.from_numpy(euler).float()
    matrix_torch = euler_zyx_to_matrix(euler_torch)
    return matrix_torch.numpy()
