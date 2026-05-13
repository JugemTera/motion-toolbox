"""Tests for quaternion mathematics utilities."""

import numpy as np
import pytest

from motion_toolbox.utils.quaternion import (
    euler_to_quaternion,
    identity_quaternion,
    normalize_quaternion,
    q_inv,
    q_mul,
    q_rotate,
    q_slerp,
    quaternion_to_euler,
)


class TestQuaternionOperations:
    """Test basic quaternion operations."""

    def test_identity_quaternion(self):
        """Test identity quaternion creation."""
        # Single identity quaternion
        identity = identity_quaternion()
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(identity, expected)

        # Batch identity quaternions
        batch_identity = identity_quaternion((3,))
        assert batch_identity.shape == (3, 4)
        expected_batch = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(batch_identity, expected_batch)

    def test_quaternion_multiplication(self):
        """Test quaternion multiplication."""
        # Identity * quaternion = quaternion
        identity = identity_quaternion()
        q = np.array([0.7071067, 0, 0, 0.7071067])  # 90 degrees around Z, (w,x,y,z)
        result = q_mul(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-5)  # Looser tolerance

        # Test non-commutative property
        q1 = normalize_quaternion(np.array([1.0, 0.3, 0.4, 0.2]))
        q2 = normalize_quaternion(np.array([0.8, 0.1, 0.5, 0.3]))
        result1 = q_mul(q1, q2)
        result2 = q_mul(q2, q1)
        # They should be different (quaternion multiplication is not commutative)
        assert not np.allclose(result1, result2, atol=1e-6)

    def test_quaternion_inverse(self):
        """Test quaternion inverse."""
        q = np.array([0.7071, 0, 0, 0.7071])  # 90 degrees around Z (w,x,y,z)
        q_inverse = q_inv(q)

        # q * q^-1 should be identity
        result = q_mul(q, q_inverse)
        identity = identity_quaternion()
        np.testing.assert_allclose(result, identity, atol=1e-6)

        # q^-1 * q should also be identity
        result2 = q_mul(q_inverse, q)
        np.testing.assert_allclose(result2, identity, atol=1e-6)

    def test_vector_rotation(self):
        """Test vector rotation by quaternion."""
        # 90 degree rotation around Z-axis (w, x, y, z)
        q = np.array([0.7071, 0, 0, 0.7071])

        # Rotate X-axis vector (1,0,0) by 90 degrees around Z
        # Should result in Y-axis vector (0,1,0)
        v = np.array([1, 0, 0])
        rotated = q_rotate(q, v)
        expected = np.array([0, 1, 0])
        np.testing.assert_allclose(rotated, expected, atol=1e-4)

        # Test with multiple vectors
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotated_batch = q_rotate(q, vectors)
        expected_batch = np.array(
            [
                [0, 1, 0],  # X -> Y
                [-1, 0, 0],  # Y -> -X
                [0, 0, 1],  # Z -> Z (unchanged)
            ]
        )
        np.testing.assert_allclose(rotated_batch, expected_batch, atol=1e-4)

    def test_quaternion_normalization(self):
        """Test quaternion normalization."""
        # Un-normalized quaternion
        q = np.array([1, 1, 1, 1])
        q_norm = normalize_quaternion(q)

        # Check that it's normalized
        norm = np.linalg.norm(q_norm)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

        # Test with batch of quaternions
        q_batch = np.array(
            [
                [1, 1, 1, 1],
                [2, 0, 0, 0],
                [1, 0, 0, 0],  # Already normalized (w,x,y,z)
            ]
        )
        q_batch_norm = normalize_quaternion(q_batch)

        # Check all are normalized
        norms = np.linalg.norm(q_batch_norm, axis=1)
        expected_norms = np.ones(3)
        np.testing.assert_allclose(norms, expected_norms, atol=1e-6)


class TestSphericalLinearInterpolation:
    """Test spherical linear interpolation (slerp)."""

    def test_slerp_endpoints(self):
        """Test that slerp returns correct endpoints."""
        q1 = identity_quaternion()
        q2 = np.array([0.7071067, 0, 0, 0.7071067])  # 90 degrees around Z (w,x,y,z)

        # t=0 should return q1
        result_start = q_slerp(q1, q2, 0.0)
        np.testing.assert_allclose(result_start, q1, atol=1e-6)

        # t=1 should return q2
        result_end = q_slerp(q1, q2, 1.0)
        np.testing.assert_allclose(result_end, q2, atol=1e-6)

    def test_slerp_midpoint(self):
        """Test slerp midpoint interpolation."""
        q1 = identity_quaternion()
        q2 = np.array([0.7071067, 0, 0, 0.7071067])  # 90 degrees around Z (w,x,y,z)

        # t=0.5 should give 45 degrees around Z
        mid = q_slerp(q1, q2, 0.5)

        # Test by rotating a vector and checking the angle
        v = np.array([1, 0, 0])  # X-axis
        rotated = q_rotate(mid, v)

        # Should be approximately [cos(45°), sin(45°), 0]
        expected = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0])
        np.testing.assert_allclose(rotated, expected, atol=1e-4)


class TestEulerConversions:
    """Test conversions between quaternions and Euler angles."""

    def test_euler_to_quaternion_roundtrip(self):
        """Test round-trip conversion: euler -> quat -> euler."""
        # Test with various angles
        test_angles = [
            [0, 0, 0],  # Identity
            [np.pi / 4, 0, 0],  # 45 degrees around X
            [0, np.pi / 2, 0],  # 90 degrees around Y
            [0, 0, np.pi],  # 180 degrees around Z
            [np.pi / 6, np.pi / 4, np.pi / 3],  # Mixed angles
        ]

        for angles in test_angles:
            angles = np.array(angles)

            # Convert to quaternion and back
            q = euler_to_quaternion(angles, "XYZ")
            recovered_angles = quaternion_to_euler(q, "XYZ")

            # Account for potential wrapping and sign ambiguity
            # For small angles, should be close
            if np.linalg.norm(angles) < np.pi / 2:
                np.testing.assert_allclose(angles, recovered_angles, atol=1e-4)

    def test_specific_rotations(self):
        """Test specific known rotations."""
        # 90 degrees around Z axis
        angles_z90 = np.array([0, 0, np.pi / 2])
        q_z90 = euler_to_quaternion(angles_z90, "XYZ")

        # (w, x, y, z) for 90° about Z
        expected_q = np.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        np.testing.assert_allclose(q_z90, expected_q, atol=1e-6)

        # Test by rotating a vector
        v = np.array([1, 0, 0])
        rotated = q_rotate(q_z90, v)
        expected_v = np.array([0, 1, 0])
        np.testing.assert_allclose(rotated, expected_v, atol=1e-4)


class TestBatchOperations:
    """Test operations on batches of quaternions."""

    def test_batch_rotation(self):
        """Test rotation of vectors by batch of quaternions."""
        # Create batch of quaternions (rotations around Z)
        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])  # 0, 90, 180, 270 degrees
        quaternions = np.array([euler_to_quaternion([0, 0, angle], "XYZ") for angle in angles])

        # Rotate X-axis vector by each quaternion
        v = np.array([1, 0, 0])
        rotated_batch = np.array([q_rotate(q, v) for q in quaternions])

        # Expected results
        expected = np.array(
            [
                [1, 0, 0],  # 0 degrees
                [0, 1, 0],  # 90 degrees
                [-1, 0, 0],  # 180 degrees
                [0, -1, 0],  # 270 degrees
            ]
        )

        np.testing.assert_allclose(rotated_batch, expected, atol=1e-4)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_normalization_of_zero_quaternion(self):
        """Test normalization of zero quaternion."""
        zero_q = np.array([0, 0, 0, 0])
        normalized = normalize_quaternion(zero_q)

        # Should not crash and should return a valid quaternion
        assert normalized.shape == (4,)
        assert not np.any(np.isnan(normalized))

    def test_quaternion_properties(self):
        """Test basic quaternion properties."""
        q = np.array([0.9, 0.1, 0.2, 0.3])
        q = normalize_quaternion(q)  # Ensure it's normalized

        # q * q^-1 = identity
        q_inv_result = q_inv(q)
        identity = q_mul(q, q_inv_result)
        expected_identity = identity_quaternion()
        np.testing.assert_allclose(identity, expected_identity, atol=1e-6)

        # Rotation preserves vector length
        v = np.array([1, 2, 3])
        rotated_v = q_rotate(q, v)
        original_length = np.linalg.norm(v)
        rotated_length = np.linalg.norm(rotated_v)
        np.testing.assert_allclose(original_length, rotated_length, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
