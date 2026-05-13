"""Tests for Motion class."""

import numpy as np
import pytest

from motion_toolbox.core.motion import Motion
from motion_toolbox.core.skeleton import Skeleton


class TestMotion:
    """Test cases for Motion class."""

    @pytest.fixture
    def simple_skeleton(self):
        """Create a simple 2-joint skeleton for testing."""
        joint_names = ("root", "child")
        parent_indices = (None, 0)
        offsets = np.array([[0, 0, 0], [0, 1, 0]])  # child is 1 unit up from root
        return Skeleton(joint_names, parent_indices, offsets)

    @pytest.fixture
    def simple_motion(self, simple_skeleton):
        """Create a simple motion with identity rotations."""
        num_frames = 10
        num_joints = 2

        # Identity quaternions [w=1, x=0, y=0, z=0]
        rotations = np.zeros((num_frames, num_joints, 4))
        rotations[:, :, 0] = 1.0

        # Zero translation
        translation = np.zeros((num_frames, 3))

        return Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none", framerate=30.0)

    def test_motion_creation(self, simple_skeleton):
        """Test basic motion creation and validation."""
        num_frames, num_joints = 5, 2

        # Valid motion data
        rotations = np.zeros((num_frames, num_joints, 4))
        rotations[:, :, 0] = 1.0  # Identity quaternions
        translation = np.zeros((num_frames, 3))

        motion = Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none")

        assert motion.num_frames == num_frames
        assert motion.num_joints == num_joints
        assert motion.framerate == 30.0
        assert motion.duration == num_frames / 30.0

    def test_motion_validation(self, simple_skeleton):
        """Test motion data validation."""
        # Test wrong rotation shape (axis-angle must be (..., 3))
        with pytest.raises(ValueError, match="joint_rotations must have shape"):
            rotations = np.zeros((5, 2, 2))  # Wrong last dimension
            translation = np.zeros((5, 3))
            Motion(rotations, translation, simple_skeleton)

        # Test wrong translation shape
        with pytest.raises(ValueError, match="root_translation must have shape"):
            rotations = np.zeros((5, 2, 4))
            rotations[:, :, 0] = 1.0
            translation = np.zeros((5, 2))  # Wrong last dimension
            Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none")

        # Test frame mismatch
        with pytest.raises(ValueError, match="Number of frames must match"):
            rotations = np.zeros((5, 2, 4))
            rotations[:, :, 0] = 1.0
            translation = np.zeros((3, 3))  # Different number of frames
            Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none")

        # Test joint mismatch
        with pytest.raises(ValueError, match="Number of joints must match"):
            rotations = np.zeros((5, 3, 4))  # Wrong number of joints
            rotations[:, :, 0] = 1.0
            translation = np.zeros((5, 3))
            Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none")

        # Test non-unit quaternions
        with pytest.raises(ValueError, match="All quaternions must be unit quaternions"):
            rotations = np.zeros((5, 2, 4))
            rotations[:, :, 0] = 2.0  # Non-unit quaternions
            translation = np.zeros((5, 3))
            Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none")

        # Test negative framerate
        with pytest.raises(ValueError, match="Framerate must be positive"):
            rotations = np.zeros((5, 2, 4))
            rotations[:, :, 0] = 1.0
            translation = np.zeros((5, 3))
            Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none", framerate=-1.0)

    def test_get_frame(self, simple_motion):
        """Test getting individual frame data."""
        rotations, translation = simple_motion.get_frame(0)

        assert rotations.shape == (2, 3)
        assert translation.shape == (3,)
        assert np.allclose(rotations, 0.0)
        assert np.allclose(translation, [0, 0, 0])

        # Test invalid frame index
        with pytest.raises(IndexError):
            simple_motion.get_frame(100)

    def test_forward_kinematics_identity(self, simple_motion):
        """Test forward kinematics with identity rotations."""
        positions = simple_motion.positions

        assert positions.shape == (10, 2, 3)

        # With identity rotations, joints should be at their T-pose positions
        expected_root = [0, 0, 0]  # Root at origin
        expected_child = [0, 1, 0]  # Child 1 unit up from root

        for frame in range(10):
            assert np.allclose(positions[frame, 0], expected_root)
            assert np.allclose(positions[frame, 1], expected_child)

    def test_forward_kinematics_with_rotation(self, simple_skeleton):
        """Test forward kinematics with actual rotations."""
        num_frames = 1

        # 90-degree rotation around Z-axis for root joint
        # Quaternion for 90° rotation around Z: [cos(π/4), 0, 0, sin(π/4)]
        cos_45 = np.cos(np.pi / 4)
        sin_45 = np.sin(np.pi / 4)

        rotations = np.zeros((num_frames, 2, 4))
        rotations[0, 0] = [cos_45, 0, 0, sin_45]  # Root rotation
        rotations[0, 1] = [1, 0, 0, 0]  # Child identity

        translation = np.zeros((num_frames, 3))

        motion = Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none")
        positions = motion.positions

        # Root should still be at origin
        assert np.allclose(positions[0, 0], [0, 0, 0])

        # Child should be rotated: originally [0, 1, 0], now should be [-1, 0, 0]
        assert np.allclose(positions[0, 1], [-1, 0, 0], atol=1e-6)

    def test_forward_kinematics_with_translation(self, simple_skeleton):
        """Test forward kinematics with root translation."""
        num_frames = 1

        rotations = np.zeros((num_frames, 2, 4))
        rotations[:, :, 0] = 1.0  # Identity rotations

        translation = np.array([[1, 2, 3]])  # Move root

        motion = Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none")
        positions = motion.positions

        # Root should be at translated position
        assert np.allclose(positions[0, 0], [1, 2, 3])

        # Child should be offset from new root position
        assert np.allclose(positions[0, 1], [1, 3, 3])

    def test_slice_frames(self, simple_motion):
        """Test frame slicing functionality."""
        # Slice middle frames
        sliced = simple_motion.slice_frames(2, 7)

        assert sliced.num_frames == 5
        assert sliced.num_joints == 2
        assert sliced.framerate == 30.0
        assert sliced.skeleton is simple_motion.skeleton

        # Test slice to end
        sliced_to_end = simple_motion.slice_frames(5)
        assert sliced_to_end.num_frames == 5

        # Test invalid slice
        with pytest.raises(ValueError, match="Invalid slice range"):
            simple_motion.slice_frames(8, 5)  # start > end

    def test_downsample(self, simple_motion):
        """Test downsampling functionality."""
        downsampled = simple_motion.downsample(2)

        assert downsampled.num_frames == 5  # 10 frames / 2
        assert downsampled.num_joints == 2
        assert downsampled.framerate == 15.0  # 30 fps / 2

        # Test invalid downsampling factor
        with pytest.raises(ValueError):
            simple_motion.downsample(0)

    def test_normalize_quaternions(self, simple_skeleton):
        """Test quaternion normalization."""
        # Create motion with slightly non-unit quaternions
        rotations = np.zeros((5, 2, 4))
        rotations[:, :] = [1.1, 0, 0, 0]  # Slightly off from unit length
        translation = np.zeros((5, 3))

        # This should fail validation
        with pytest.raises(ValueError):
            Motion.from_quaternions(rotations, translation, simple_skeleton, floor_alignment="none")

        # But we can normalize first
        normalized_rotations = rotations / np.linalg.norm(rotations, axis=2, keepdims=True)
        motion = Motion.from_quaternions(normalized_rotations, translation, simple_skeleton, floor_alignment="none")

        # Test the normalize_quaternions method
        normalized_motion = motion.normalize_quaternions()
        assert isinstance(normalized_motion, Motion)
        assert normalized_motion.num_frames == motion.num_frames

    def test_create_identity(self, simple_skeleton):
        """Test identity motion creation."""
        motion = Motion.create_identity(simple_skeleton, num_frames=20, framerate=60.0)

        assert motion.num_frames == 20
        assert motion.num_joints == 2
        assert motion.framerate == 60.0

        # All rotations should be identity (zero axis-angle)
        for frame in range(20):
            for joint in range(2):
                assert np.allclose(motion.joint_rotations[frame, joint], [0.0, 0.0, 0.0])

        # All translations should be zero
        assert np.allclose(motion.root_translation, 0)

    def test_serialization(self, simple_motion):
        """Test dictionary serialization/deserialization."""
        motion_dict = simple_motion.to_dict()

        assert "joint_rotations" in motion_dict
        assert "root_translation" in motion_dict
        assert "skeleton" in motion_dict
        assert "framerate" in motion_dict
        assert motion_dict.get("up_axis") == simple_motion.up_axis
        assert motion_dict.get("floor_alignment") == simple_motion.floor_alignment

        # Reconstruct motion
        reconstructed = Motion.from_dict(motion_dict)

        assert reconstructed.num_frames == simple_motion.num_frames
        assert reconstructed.num_joints == simple_motion.num_joints
        assert reconstructed.framerate == simple_motion.framerate
        assert reconstructed.up_axis == simple_motion.up_axis
        assert reconstructed.floor_alignment == simple_motion.floor_alignment
        assert np.allclose(reconstructed.joint_rotations, simple_motion.joint_rotations)
        assert np.allclose(reconstructed.root_translation, simple_motion.root_translation)


class TestMotionMLInterop:
    """Test ML framework interoperability."""

    @pytest.fixture
    def simple_motion(self):
        """Create a simple motion for ML testing."""
        joint_names = ("root", "child")
        parent_indices = (None, 0)
        offsets = np.array([[0, 0, 0], [0, 1, 0]])
        skeleton = Skeleton(joint_names, parent_indices, offsets)

        return Motion.create_identity(skeleton, num_frames=5, framerate=30.0)

    @pytest.mark.parametrize("framework", ["torch", "jax"])
    def test_ml_framework_conversion(self, simple_motion, framework):
        """Test conversion to/from ML frameworks."""
        if framework == "torch":
            try:
                import torch
            except ImportError:
                pytest.skip("PyTorch not available")

            # Test to_torch
            rot_tensor, trans_tensor = simple_motion.to_torch()
            assert hasattr(rot_tensor, "shape")
            assert rot_tensor.shape == (5, 2, 3)
            assert trans_tensor.shape == (5, 3)

            # Test from_torch
            reconstructed = Motion.from_torch(rot_tensor, trans_tensor, simple_motion.skeleton, floor_alignment="none")
            assert np.allclose(reconstructed.joint_rotations, simple_motion.joint_rotations)

        elif framework == "jax":
            try:
                import jax  # noqa: F401
            except ImportError:
                pytest.skip("JAX not available")

            # Test to_jax
            rot_array, trans_array = simple_motion.to_jax()
            assert hasattr(rot_array, "shape")
            assert rot_array.shape == (5, 2, 3)
            assert trans_array.shape == (5, 3)

            # Test from_jax
            reconstructed = Motion.from_jax(rot_array, trans_array, simple_motion.skeleton, floor_alignment="none")
            assert np.allclose(reconstructed.joint_rotations, simple_motion.joint_rotations)

    def test_ml_import_errors(self, simple_motion):
        """Test that appropriate errors are raised when ML frameworks are not available."""
        # Mock the import to fail
        import sys

        # Test torch import error
        if "torch" in sys.modules:
            # Skip if torch is actually available
            pass
        else:
            with pytest.raises(ImportError, match="PyTorch is required"):
                simple_motion.to_torch()

        # Test jax import error
        if "jax" in sys.modules:
            # Skip if jax is actually available
            pass
        else:
            with pytest.raises(ImportError, match="JAX is required"):
                simple_motion.to_jax()
