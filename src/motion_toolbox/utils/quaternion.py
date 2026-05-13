"""Quaternion and rotation mathematics utilities.

This module provides utility functions for quaternion operations and conversions
between different rotation representations (quaternions, axis-angles, Euler angles).
All quaternions use the (w, x, y, z) convention for consistency with motion data.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.

    Args:
        q1: First quaternion(s), shape (..., 4) in (w, x, y, z) format
        q2: Second quaternion(s), shape (..., 4) in (w, x, y, z) format

    Returns:
        Product quaternion(s), shape (..., 4) in (w, x, y, z) format

    Examples:
        >>> import numpy as np
        >>> q1 = np.array([1, 0, 0, 0])  # Identity
        >>> q2 = np.array([0.7071, 0, 0, 0.7071])  # 90 degrees around Z
        >>> result = q_mul(q1, q2)
        >>> np.allclose(result, q2, atol=1e-4)
        True
    """
    # Convert to scipy format and multiply
    if q1.ndim == 1:
        q1_scipy = np.array([q1[1], q1[2], q1[3], q1[0]])
        q2_scipy = np.array([q2[1], q2[2], q2[3], q2[0]])
    else:
        q1_scipy = np.column_stack([q1[..., 1], q1[..., 2], q1[..., 3], q1[..., 0]])
        q2_scipy = np.column_stack([q2[..., 1], q2[..., 2], q2[..., 3], q2[..., 0]])

    # Create rotation objects and multiply
    rot1 = R.from_quat(q1_scipy)
    rot2 = R.from_quat(q2_scipy)
    result_rot = rot1 * rot2

    # Convert back to (w, x, y, z) format
    result_scipy = result_rot.as_quat()
    if result_scipy.ndim == 1:
        return np.array([result_scipy[3], result_scipy[0], result_scipy[1], result_scipy[2]])
    else:
        return np.column_stack([result_scipy[..., 3], result_scipy[..., 0], result_scipy[..., 1], result_scipy[..., 2]])


def q_inv(q: np.ndarray) -> np.ndarray:
    """Compute the inverse (conjugate for unit quaternions) of quaternion(s).

    Args:
        q: Quaternion(s) to invert, shape (..., 4) in (w, x, y, z) format

    Returns:
        Inverted quaternion(s), shape (..., 4) in (w, x, y, z) format

    Examples:
        >>> import numpy as np
        >>> q = np.array([0.7071, 0, 0, 0.7071])  # 90 degrees around Z
        >>> q_inverse = q_inv(q)
        >>> # Multiplying by inverse should give identity
        >>> identity = q_mul(q, q_inverse)
        >>> np.allclose(identity, [1, 0, 0, 0], atol=1e-4)
        True
    """
    # Convert to scipy format
    if q.ndim == 1:
        q_scipy = np.array([q[1], q[2], q[3], q[0]])
    else:
        q_scipy = np.column_stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]])

    # Invert rotation
    rot = R.from_quat(q_scipy)
    inv_rot = rot.inv()

    # Convert back to (w, x, y, z) format
    result_scipy = inv_rot.as_quat()
    if result_scipy.ndim == 1:
        return np.array([result_scipy[3], result_scipy[0], result_scipy[1], result_scipy[2]])
    else:
        return np.column_stack([result_scipy[..., 3], result_scipy[..., 0], result_scipy[..., 1], result_scipy[..., 2]])


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector or array of vectors by a quaternion.

    Args:
        q: Quaternion(s) to rotate by, shape (..., 4) in (w, x, y, z) format
        v: Vector(s) to rotate, shape (..., 3)

    Returns:
        Rotated vector(s), shape (..., 3)

    Examples:
        >>> import numpy as np
        >>> q = np.array([0.7071, 0, 0, 0.7071])  # 90 degree rotation around Z (w,x,y,z)
        >>> v = np.array([1, 0, 0])  # X-axis vector
        >>> rotated = q_rotate(q, v)
        >>> np.allclose(rotated, [0, 1, 0], atol=1e-4)
        True
    """
    # Ensure inputs are numpy arrays
    q = np.asarray(q)
    v = np.asarray(v)

    # Convert from (w, x, y, z) to (x, y, z, w) format for scipy
    if q.ndim == 1:
        # Single quaternion
        q_scipy = np.array([q[1], q[2], q[3], q[0]])
    else:
        # Multiple quaternions
        q_scipy = np.column_stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]])

    # Convert to scipy Rotation object
    rot = R.from_quat(q_scipy)

    # Apply rotation to vector(s)
    return rot.apply(v)


def q_slerp(q1: np.ndarray, q2: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: Start quaternion(s), shape (..., 4) in (x, y, z, w) format
        q2: End quaternion(s), shape (..., 4) in (x, y, z, w) format
        t: Interpolation parameter(s), 0.0 returns q1, 1.0 returns q2

    Returns:
        Interpolated quaternion(s), shape (..., 4) in (x, y, z, w) format

    Examples:
        >>> import numpy as np
        >>> q1 = np.array([0, 0, 0, 1])  # Identity
        >>> q2 = np.array([0, 0, 0.7071, 0.7071])  # 90 degrees around Z
        >>> mid = q_slerp(q1, q2, 0.5)  # Should be 45 degrees
        >>> # Check that it's approximately 45 degrees
        >>> angle = 2 * np.arccos(np.abs(mid[3]))  # Extract angle from w component
        >>> np.allclose(angle, np.pi/4, atol=1e-4)
        True
    """
    # Use scipy's Slerp class for interpolation
    if np.isscalar(t):
        # Single interpolation
        times = np.array([0.0, 1.0])
        rots = R.from_quat(np.array([q1, q2]))
        slerp = Slerp(times, rots)
        result_rot = slerp([t])[0]
    else:
        # Multiple interpolations
        times = np.array([0.0, 1.0])
        rots = R.from_quat(np.array([q1, q2]))
        slerp = Slerp(times, rots)
        result_rot = slerp(t)

    return result_rot.as_quat()


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion or array of quaternions.

    Args:
        q: Quaternion(s) to normalize, shape (..., 4) in (x, y, z, w) format

    Returns:
        Normalized quaternion(s), shape (..., 4) in (x, y, z, w) format

    Examples:
        >>> import numpy as np
        >>> q = np.array([1, 1, 1, 1])  # Un-normalized quaternion
        >>> q_norm = normalize_quaternion(q)
        >>> np.allclose(np.linalg.norm(q_norm), 1.0)
        True
    """
    # Calculate norms
    if q.ndim == 1:
        norm = np.linalg.norm(q)
    else:
        norm = np.linalg.norm(q, axis=-1, keepdims=True)

    # Avoid division by zero
    norm = np.where(norm == 0, 1.0, norm)

    return q / norm


def quaternion_to_euler(q: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Convert quaternion(s) to Euler angles.

    Args:
        q: Quaternion(s), shape (..., 4) in (w, x, y, z) format
        order: Euler angle order (e.g., 'XYZ', 'ZYX'), default 'XYZ'

    Returns:
        Euler angles in radians, shape (..., 3)
    """
    q = np.asarray(q)
    if q.ndim == 1:
        q_scipy = np.array([q[1], q[2], q[3], q[0]])
    else:
        q_scipy = np.column_stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]])
    rot = R.from_quat(q_scipy)
    return rot.as_euler(order.lower())


def euler_to_quaternion(angles: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Convert Euler angles to quaternion(s).

    Args:
        angles: Euler angles in radians, shape (..., 3)
        order: Euler angle order (e.g., 'XYZ', 'ZYX'), default 'XYZ'

    Returns:
        Quaternion(s), shape (..., 4) in (w, x, y, z) format
    """
    rot = R.from_euler(order.lower(), np.asarray(angles))
    s = rot.as_quat()  # SciPy: (x, y, z, w)
    if s.ndim == 1:
        return np.array([s[3], s[0], s[1], s[2]])
    return np.column_stack([s[..., 3], s[..., 0], s[..., 1], s[..., 2]])


def identity_quaternion(shape: tuple = ()) -> np.ndarray:
    """Create identity quaternion(s).

    Args:
        shape: Shape for array of identity quaternions, default is single quaternion

    Returns:
        Identity quaternion(s) [1, 0, 0, 0] (w, x, y, z), shape (*shape, 4)

    Examples:
        >>> identity = identity_quaternion()
        >>> identity.tolist()
        [1.0, 0.0, 0.0, 0.0]
        >>> batch_identity = identity_quaternion((3,))
        >>> batch_identity.shape
        (3, 4)
    """
    if shape == ():
        return np.array([1.0, 0.0, 0.0, 0.0])
    result = np.zeros(shape + (4,))
    result[..., 0] = 1.0
    return result


# === AXIS-ANGLE CONVERSION FUNCTIONS ===


def quaternion_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to axis-angle representation.

    Args:
        q: Quaternion(s), shape (..., 4) in (w, x, y, z) format

    Returns:
        Axis-angle representation, shape (..., 3)
        The magnitude represents the rotation angle in radians,
        the direction represents the rotation axis

    Examples:
        >>> import numpy as np
        >>> # 90 degree rotation around Z axis
        >>> q = np.array([0.7071, 0, 0, 0.7071])  # (w, x, y, z)
        >>> axis_angle = quaternion_to_axis_angle(q)
        >>> np.allclose(axis_angle, [0, 0, np.pi/2], atol=1e-4)
        True
    """
    # Ensure input is numpy array
    q = np.asarray(q)

    # Convert from (w, x, y, z) to (x, y, z, w) format for scipy
    if q.ndim == 1:
        q_scipy = np.array([q[1], q[2], q[3], q[0]])
    else:
        q_scipy = np.column_stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]])

    # Convert to scipy Rotation and extract axis-angle
    rot = R.from_quat(q_scipy)
    return rot.as_rotvec()


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to quaternion(s).

    Args:
        axis_angle: Axis-angle representation, shape (..., 3)
                   The magnitude represents the rotation angle in radians,
                   the direction represents the rotation axis

    Returns:
        Quaternion(s), shape (..., 4) in (w, x, y, z) format

    Examples:
        >>> import numpy as np
        >>> # 90 degree rotation around Z axis
        >>> axis_angle = np.array([0, 0, np.pi/2])
        >>> q = axis_angle_to_quaternion(axis_angle)
        >>> np.allclose(q, [0.7071, 0, 0, 0.7071], atol=1e-4)
        True
    """
    # Ensure input is numpy array and writable
    axis_angle = np.asarray(axis_angle)
    if not axis_angle.flags.writeable:
        axis_angle = axis_angle.copy()

    # Convert to scipy Rotation from axis-angle
    rot = R.from_rotvec(axis_angle)

    # Convert to quaternion and reformat to (w, x, y, z)
    q_scipy = rot.as_quat()
    if q_scipy.ndim == 1:
        return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
    else:
        return np.column_stack([q_scipy[..., 3], q_scipy[..., 0], q_scipy[..., 1], q_scipy[..., 2]])


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle: Axis-angle representation, shape (..., 3)

    Returns:
        Rotation matrix, shape (..., 3, 3)

    Examples:
        >>> import numpy as np
        >>> # 90 degree rotation around Z axis
        >>> axis_angle = np.array([0, 0, np.pi/2])
        >>> rot_mat = axis_angle_to_rotation_matrix(axis_angle)
        >>> expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> np.allclose(rot_mat, expected, atol=1e-4)
        True
    """
    # Ensure input is numpy array
    axis_angle = np.asarray(axis_angle)

    # Convert to scipy Rotation and extract matrix
    rot = R.from_rotvec(axis_angle)
    return rot.as_matrix()


def rotation_matrix_to_axis_angle(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation.

    Args:
        rot_mat: Rotation matrix, shape (..., 3, 3)

    Returns:
        Axis-angle representation, shape (..., 3)

    Examples:
        >>> import numpy as np
        >>> # 90 degree rotation around Z axis matrix
        >>> rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> axis_angle = rotation_matrix_to_axis_angle(rot_mat)
        >>> np.allclose(axis_angle, [0, 0, np.pi/2], atol=1e-4)
        True
    """
    # Ensure input is numpy array
    rot_mat = np.asarray(rot_mat)

    # Convert to scipy Rotation from matrix
    rot = R.from_matrix(rot_mat)
    return rot.as_rotvec()


def extract_angle_from_rotation_matrix(rot_mat: np.ndarray) -> float | np.ndarray:
    """Extract rotation angle from rotation matrix using trace formula.

    Uses the formula: angle = arccos((trace(R) - 1) / 2)
    This is more direct and numerically stable than converting to axis-angle
    representation when only the angle magnitude is needed.

    Args:
        rot_mat: Rotation matrix, shape (..., 3, 3) or (3, 3)

    Returns:
        Rotation angle(s) in degrees, scalar or array depending on input shape

    Examples:
        >>> import numpy as np
        >>> # 90 degree rotation around Z axis matrix
        >>> rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> angle = extract_angle_from_rotation_matrix(rot_mat)
        >>> np.allclose(angle, np.pi/2, atol=1e-4)
        True
        >>>
        >>> # Batch of rotation matrices
        >>> rot_mats = np.array([
        ...     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity (0 degrees)
        ...     [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 degrees around Z
        ... ])
        >>> angles = extract_angle_from_rotation_matrix(rot_mats)
        >>> np.allclose(angles, [0, np.pi/2], atol=1e-4)
        True
    """
    # Ensure input is numpy array
    rot_mat = np.asarray(rot_mat)

    # Compute trace: sum of diagonal elements
    # For batched matrices (..., 3, 3), trace along last two dimensions
    trace = np.trace(rot_mat, axis1=-2, axis2=-1)

    # Apply formula: angle = arccos((trace - 1) / 2)
    # Clamp to valid range [-1, 1] to handle numerical errors
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)

    # Return angle in radians
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float | np.ndarray:
    """Compute the angle between two vectors or sets of vectors.

    Args:
        v1: First vector(s), shape (..., 3)
        v2: Second vector(s), shape (..., 3)

    Returns:
        Angle(s) between vectors in radians

    Examples:
        >>> import numpy as np
        >>> v1 = np.array([1, 0, 0])  # X axis
        >>> v2 = np.array([0, 1, 0])  # Y axis
        >>> angle = angle_between_vectors(v1, v2)
        >>> np.allclose(angle, np.pi/2, atol=1e-6)
        True
    """
    # Ensure inputs are numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

    # Compute dot product and clamp to valid range for arccos
    dot_product = np.sum(v1_norm * v2_norm, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Return angle in radians
    return np.arccos(dot_product)


# Aliases for compatibility with different naming conventions
axis_angle_to_matrix = axis_angle_to_rotation_matrix
matrix_to_axis_angle = rotation_matrix_to_axis_angle
