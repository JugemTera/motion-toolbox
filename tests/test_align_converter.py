"""Unit tests for ALIGN converter."""

import numpy as np
import pytest
import torch

from motion_toolbox.converters.align_converter import ALIGNConverter
from motion_toolbox.core.motion import Motion
from motion_toolbox.core.skeleton import Skeleton


class TestALIGNConverter:
    """Test suite for ALIGN converter."""

    @pytest.fixture
    def converter(self):
        """Create an ALIGN converter instance."""
        return ALIGNConverter()

    @pytest.fixture
    def sample_motion(self):
        """Create a simple test motion with SMPL (24-joint) skeleton for round-trip tests."""
        skeleton = Skeleton.from_smpl()
        num_frames = 30
        num_joints = skeleton.num_joints

        # Small random rotations
        joint_rotations = 0.1 * np.random.randn(num_frames, num_joints, 3)

        # Simple forward translation (Y-up, consistent with from_align reconstruction)
        root_translation = np.zeros((num_frames, 3))
        root_translation[:, 0] = np.linspace(0, 1, num_frames)  # Move forward
        root_translation[:, 1] = 1.0  # Height above ground

        return Motion(
            joint_rotations=joint_rotations,
            root_translation=root_translation,
            skeleton=skeleton,
            framerate=20.0,
            up_axis="y",
            floor_alignment="none",
        )

    def test_feature_dimensions(self, converter, sample_motion):
        """Test that output has correct dimensions."""
        features = converter.to_align(sample_motion)

        assert features.ndim == 2
        assert features.shape[0] == sample_motion.num_frames
        assert features.shape[1] == 232, f"Expected 232 features, got {features.shape[1]}"

    def test_feature_dtype(self, converter, sample_motion):
        """Test that features are float type."""
        features = converter.to_align(sample_motion)
        assert np.issubdtype(features.dtype, np.floating)

    def test_round_trip_conversion(self, converter, sample_motion):
        """Test that round-trip conversion preserves data."""
        # Forward conversion
        features = converter.to_align(sample_motion)

        # Inverse conversion
        reconstructed = converter.from_align(features, fps=sample_motion.framerate)

        # Check reconstruction accuracy
        # Note: Some precision loss is expected due to the ALIGN representation
        assert reconstructed.num_frames == sample_motion.num_frames
        assert reconstructed.num_joints == sample_motion.num_joints

        # Rotations: global root is re-canonicalized by ALIGN; body joints should be stable.
        assert np.allclose(
            sample_motion.joint_rotations[:, 1:22],
            reconstructed.joint_rotations[:, 1:22],
            atol=0.25,
        ), "Body joint rotations not preserved in round-trip"

        # Translation should be reasonably preserved
        # ALIGN uses canonical alignment, so we expect some differences
        orig_trans_centered = sample_motion.root_translation - sample_motion.root_translation[0]
        recon_trans_centered = reconstructed.root_translation - reconstructed.root_translation[0]

        assert np.allclose(orig_trans_centered, recon_trans_centered, atol=0.1), (
            "Translation not preserved in round-trip"
        )

    def test_canonical_alignment(self, converter, sample_motion):
        """Test that first frame is aligned to canonical direction."""
        features = converter.to_align(sample_motion)

        # First frame trajectory should be at origin
        trajectory = converter.extract_trajectory(features)
        assert np.allclose(trajectory[0], [0, 0], atol=1e-5), f"First frame trajectory not at origin: {trajectory[0]}"

    def test_extract_root_height(self, converter, sample_motion):
        """Test root height extraction."""
        features = converter.to_align(sample_motion)
        root_height = converter.extract_root_height(features)

        assert root_height.shape == (sample_motion.num_frames,)
        assert np.all(root_height >= 0), "Root height should be non-negative after ground removal"

    def test_extract_trajectory(self, converter, sample_motion):
        """Test trajectory extraction."""
        features = converter.to_align(sample_motion)
        trajectory = converter.extract_trajectory(features)

        assert trajectory.shape == (sample_motion.num_frames, 2)
        # Trajectory should start at origin (canonical alignment)
        assert np.allclose(trajectory[0], [0, 0], atol=1e-5)

    def test_extract_heading(self, converter, sample_motion):
        """Test heading extraction."""
        features = converter.to_align(sample_motion)
        heading = converter.extract_heading(features)

        assert heading.shape == (sample_motion.num_frames, 2)
        # Heading is (cos, sin), so magnitude should be ~1
        heading_magnitude = np.linalg.norm(heading, axis=-1)
        assert np.allclose(heading_magnitude, 1.0, atol=0.1), "Heading vectors should have unit magnitude"

    def test_extract_poses_6d(self, converter, sample_motion):
        """Test 6D pose extraction."""
        features = converter.to_align(sample_motion)
        poses_6d = converter.extract_poses_6d(features)

        assert poses_6d.shape == (sample_motion.num_frames, 22, 6)

    def test_extract_joints_local(self, converter, sample_motion):
        """Test local joint position extraction."""
        features = converter.to_align(sample_motion)
        joints_local = converter.extract_joints_local(features)

        assert joints_local.shape == (sample_motion.num_frames, 23, 3)

    def test_extract_foot_positions(self, converter, sample_motion):
        """Test foot position extraction."""
        features = converter.to_align(sample_motion)
        foot_pos = converter.extract_foot_positions(features)

        assert foot_pos.shape == (sample_motion.num_frames, 4, 3)
        # Foot positions should be in global coordinates
        assert np.all(foot_pos[:, :, 2] >= 0), "Foot Z positions should be non-negative (above ground)"

    def test_extract_foot_velocities(self, converter, sample_motion):
        """Test foot velocity extraction."""
        features = converter.to_align(sample_motion)
        foot_vel = converter.extract_foot_velocities(features)

        assert foot_vel.shape == (sample_motion.num_frames, 4, 3)

    def test_extract_trajectory_velocity(self, converter, sample_motion):
        """Test trajectory velocity extraction."""
        features = converter.to_align(sample_motion)
        traj_vel = converter.extract_trajectory_velocity(features)

        assert traj_vel.shape == (sample_motion.num_frames, 2)

    def test_compute_foot_contacts(self, converter, sample_motion):
        """Test foot contact detection."""
        features = converter.to_align(sample_motion)
        contacts = converter.compute_foot_contacts(features)

        assert contacts.shape == (sample_motion.num_frames, 4)
        assert contacts.dtype == bool

    def test_detect_sliding(self, converter, sample_motion):
        """Test foot sliding detection."""
        features = converter.to_align(sample_motion)
        sliding = converter.detect_sliding(features)

        assert sliding.shape == (sample_motion.num_frames, 4)
        assert sliding.dtype == bool

    def test_velocity_computation(self, converter):
        """Test velocity computation preserves length."""
        # Create test data
        data = torch.randn(100, 2, dtype=torch.double)
        vel = converter._compute_velocity(data)

        # Should return same length
        assert vel.shape == data.shape

        # First T-1 differences should match standard diff
        assert torch.allclose(vel[:-1], torch.diff(data, dim=0), atol=1e-6), "Velocity computation incorrect"

    def test_velocity_short_sequence(self, converter):
        """Test velocity computation with short sequences."""
        # Single frame
        data_1 = torch.randn(1, 3, dtype=torch.double)
        vel_1 = converter._compute_velocity(data_1)
        assert vel_1.shape == (1, 3)

        # Two frames
        data_2 = torch.randn(2, 3, dtype=torch.double)
        vel_2 = converter._compute_velocity(data_2)
        assert vel_2.shape == (2, 3)

    def test_convert_auto_detection(self, converter, sample_motion):
        """Test automatic direction detection in convert()."""
        # Motion → ALIGN
        features = converter.convert(sample_motion)
        assert features.shape[-1] == 232

        # ALIGN → Motion
        motion = converter.convert(features)
        assert isinstance(motion, Motion)
        assert motion.num_frames == sample_motion.num_frames

    def test_invalid_input(self, converter):
        """Test that invalid inputs raise appropriate errors."""
        # Wrong shape
        with pytest.raises(ValueError):
            converter.from_align(np.random.randn(10, 100))

        # Wrong type
        with pytest.raises(ValueError):
            converter.convert("invalid_data")

    def test_static_motion(self, converter):
        """Test with static motion (no movement)."""
        skeleton = Skeleton.from_humanml3d()
        num_frames = 20
        num_joints = skeleton.num_joints

        # Zero rotations and translation
        joint_rotations = np.zeros((num_frames, num_joints, 3))
        root_translation = np.zeros((num_frames, 3))
        root_translation[:, 2] = 1.0  # Just height

        motion = Motion(
            joint_rotations=joint_rotations,
            root_translation=root_translation,
            skeleton=skeleton,
            framerate=20.0,
            up_axis="z",
            floor_alignment="none",
        )

        features = converter.to_align(motion)
        assert features.shape == (num_frames, 232)

        # Trajectory velocity should be near zero for static motion
        traj_vel = converter.extract_trajectory_velocity(features)
        assert np.allclose(traj_vel, 0.0, atol=0.1)

    def test_feature_ranges(self, converter, sample_motion):
        """Test that features have reasonable ranges."""
        features = converter.to_align(sample_motion)

        # Features should not have extreme values
        assert np.all(np.isfinite(features)), "Features contain non-finite values"
        assert np.abs(features).max() < 100, "Features have unreasonably large values"

    def test_different_framerates(self, converter, sample_motion):
        """Test conversion with different framerates."""
        features = converter.to_align(sample_motion)

        # Convert back with different FPS
        for fps in [10.0, 20.0, 30.0, 60.0]:
            motion = converter.from_align(features, fps=fps)
            assert motion.framerate == fps
            assert motion.num_frames == sample_motion.num_frames


class TestALIGNIntegration:
    """Integration tests for ALIGN with other components."""

    def test_smpl_skeleton_compatibility(self):
        """Test that ALIGN works with HumanML3D-sized skeleton (22 joints)."""
        converter = ALIGNConverter()
        skeleton = Skeleton.from_humanml3d()

        num_frames = 10
        joint_rotations = np.zeros((num_frames, skeleton.num_joints, 3))
        root_translation = np.zeros((num_frames, 3))

        motion = Motion(
            joint_rotations=joint_rotations,
            root_translation=root_translation,
            skeleton=skeleton,
            framerate=20.0,
            up_axis="z",
            floor_alignment="none",
        )

        features = converter.to_align(motion)
        assert features.shape == (num_frames, 232)

    def test_batch_processing(self):
        """Test that converter can handle batch processing."""
        converter = ALIGNConverter()
        skeleton = Skeleton.from_humanml3d()

        # Create multiple motions
        motions = []
        for i in range(3):
            joint_rotations = 0.1 * np.random.randn(20, skeleton.num_joints, 3)
            root_translation = np.random.randn(20, 3)
            motion = Motion(
                joint_rotations=joint_rotations,
                root_translation=root_translation,
                skeleton=skeleton,
                framerate=20.0,
                up_axis="z",
                floor_alignment="none",
            )
            motions.append(motion)

        # Convert all to ALIGN
        all_features = []
        for motion in motions:
            features = converter.to_align(motion)
            all_features.append(features)

        # All should have same feature dimension
        assert all(f.shape[1] == 232 for f in all_features)
