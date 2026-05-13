"""Unit tests for 272-dimensional motion representation converters.

Tests bidirectional conversion between SMPL and 272-dim format.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

# Test setup
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
SMPL_SAMPLE_DIR = SAMPLE_DATA_DIR / "smpl"
FORMAT_272_SAMPLE_DIR = SAMPLE_DATA_DIR / "format_272"


@pytest.fixture
def body_model_path():
    """Get body model path from environment or skip test."""
    path = os.getenv("SMPL_BODY_MODELS_PATH")
    if not path or not Path(path).exists():
        pytest.skip("SMPL_BODY_MODELS_PATH not set or invalid")
    return path


@pytest.fixture
def sample_smpl_data():
    """Load sample SMPL data for testing."""
    # Try to find any .npz file in SMPL sample directory
    smpl_files = list(SMPL_SAMPLE_DIR.glob("*.npz"))
    if not smpl_files:
        pytest.skip("No SMPL sample data found in sample_data/smpl/")

    # Load first available file
    data = np.load(smpl_files[0], allow_pickle=True)
    return dict(data)


@pytest.fixture
def sample_272_data():
    """Load sample 272-dim data for testing."""
    # Try to find any .npy file in format_272 directory
    format_272_files = list(FORMAT_272_SAMPLE_DIR.glob("[0-9]*.npy"))
    if not format_272_files:
        pytest.skip("No 272-dim sample data found in sample_data/format_272/")

    # Load first available file
    return np.load(format_272_files[0])


class TestSMPLTo272Converter:
    """Test SMPL to 272-dim conversion."""

    def test_converter_initialization(self, body_model_path):
        """Test converter can be initialized."""
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        converter = SMPLTo272Converter(body_model_path=body_model_path)
        assert converter.body_model_path == body_model_path
        assert converter.input_format == "smpl"
        assert converter.output_format == "format_272"

    def test_converter_requires_body_model_path(self):
        """Test that converter requires body_model_path."""
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        with pytest.raises(ValueError, match="body_model_path is required"):
            SMPLTo272Converter(body_model_path=None)

    def test_convert_smpl_to_272(self, body_model_path, sample_smpl_data):
        """Test basic SMPL to 272-dim conversion."""
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        converter = SMPLTo272Converter(body_model_path=body_model_path)
        result = converter.convert(sample_smpl_data)

        # Check output shape
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 272, f"Expected 272 features, got {result.shape[1]}"

        # Check data type
        assert result.dtype == np.float32

    def test_272_format_structure(self, body_model_path, sample_smpl_data):
        """Test that 272-dim format has correct structure."""
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        converter = SMPLTo272Converter(body_model_path=body_model_path)
        result = converter.convert(sample_smpl_data)

        num_frames = result.shape[0]

        # Extract components
        root_vel_xz = result[:, 0:2]
        root_ang_vel = result[:, 2:8]
        local_pos = result[:, 8:74]
        local_vel = result[:, 74:140]
        rotations_6d = result[:, 140:272]

        # Check shapes
        assert root_vel_xz.shape == (num_frames, 2), "Root velocity XZ should be 2D"
        assert root_ang_vel.shape == (num_frames, 6), "Root angular velocity should be 6D"
        assert local_pos.shape == (num_frames, 66), "Local positions should be 66D (22 joints × 3)"
        assert local_vel.shape == (num_frames, 66), "Local velocities should be 66D"
        assert rotations_6d.shape == (num_frames, 132), "Rotations should be 132D (22 joints × 6)"

        # Check first frame has identity root rotation (approximately)
        assert np.allclose(result[0, 2], 1.0, atol=1e-5), "First frame should have identity rotation (elem 2 = 1)"
        assert np.allclose(result[0, 6], 1.0, atol=1e-5), "First frame should have identity rotation (elem 6 = 1)"

    def test_feature_ranges_are_reasonable(self, body_model_path, sample_smpl_data):
        """Test that converted features have reasonable ranges."""
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        converter = SMPLTo272Converter(body_model_path=body_model_path)
        result = converter.convert(sample_smpl_data)

        # Check for NaN or Inf
        assert not np.isnan(result).any(), "Result contains NaN values"
        assert not np.isinf(result).any(), "Result contains Inf values"

        # Positions should be in reasonable range (meters, typically < 10m)
        local_pos = result[:, 8:74].reshape(-1, 22, 3)
        assert np.abs(local_pos).max() < 100, "Local positions seem unreasonably large"

        # Velocities should be reasonable (m/s, typically < 10 m/s)
        root_vel_xz = result[:, 0:2]
        assert np.abs(root_vel_xz).max() < 50, "Root velocities seem unreasonably large"


class TestFormat272ToSMPLConverter:
    """Test 272-dim to SMPL conversion."""

    def test_converter_initialization(self):
        """Test converter can be initialized."""
        from motion_toolbox.converters.motion_converters import Format272ToSMPLConverter

        converter = Format272ToSMPLConverter()
        assert converter.input_format == "format_272"
        assert converter.output_format == "smpl"

    def test_convert_272_to_smpl(self, body_model_path, sample_272_data):
        """Test basic 272-dim to SMPL conversion."""
        from motion_toolbox.converters.motion_converters import Format272ToSMPLConverter

        converter = Format272ToSMPLConverter(body_model_path=body_model_path)
        result = converter.convert(sample_272_data)

        # Check result is a dictionary with expected keys
        assert isinstance(result, dict)
        assert "body_pose" in result
        assert "global_orient" in result
        assert "transl" in result
        assert "betas" in result

        # Check shapes
        num_frames = sample_272_data.shape[0]
        assert result["body_pose"].shape == torch.Size([num_frames, 63])
        assert result["global_orient"].shape == torch.Size([num_frames, 3])
        assert result["transl"].shape == torch.Size([num_frames, 3])
        assert result["betas"].shape[1] == 10

    def test_smpl_parameters_are_tensors(self, body_model_path, sample_272_data):
        """Test that returned SMPL parameters are torch tensors."""
        from motion_toolbox.converters.motion_converters import Format272ToSMPLConverter

        converter = Format272ToSMPLConverter(body_model_path=body_model_path)
        result = converter.convert(sample_272_data)

        assert torch.is_tensor(result["body_pose"])
        assert torch.is_tensor(result["global_orient"])
        assert torch.is_tensor(result["transl"])
        assert torch.is_tensor(result["betas"])


class TestRoundTripConversion:
    """Test round-trip conversion: SMPL → 272 → SMPL."""

    def test_smpl_to_272_to_smpl_consistency(self, body_model_path, sample_smpl_data):
        """Test that round-trip conversion maintains consistency."""
        from motion_toolbox.converters.motion_converters import Format272ToSMPLConverter, SMPLTo272Converter

        # Convert SMPL → 272
        smpl_to_272 = SMPLTo272Converter(body_model_path=body_model_path)
        motion_272 = smpl_to_272.convert(sample_smpl_data)

        # Convert 272 → SMPL
        format_272_to_smpl = Format272ToSMPLConverter(body_model_path=body_model_path)
        recovered_smpl = format_272_to_smpl.convert(motion_272)

        # Compare original and recovered SMPL parameters
        # Note: Some differences are expected due to numerical precision and processing steps

        # Extract original parameters
        orig_body_pose = sample_smpl_data["pose_body"]
        orig_root_orient = sample_smpl_data["root_orient"]
        orig_trans = sample_smpl_data["trans"]

        if torch.is_tensor(orig_body_pose):
            orig_body_pose = orig_body_pose.numpy()
        if torch.is_tensor(orig_root_orient):
            orig_root_orient = orig_root_orient.numpy()
        if torch.is_tensor(orig_trans):
            orig_trans = orig_trans.numpy()

        # Extract recovered parameters
        recovered_body_pose = recovered_smpl["body_pose"].numpy()
        recovered_root_orient = recovered_smpl["global_orient"].numpy()
        recovered_trans = recovered_smpl["transl"].numpy()

        # Reshape for comparison
        orig_body_pose_flat = orig_body_pose.reshape(-1, 63)
        recovered_body_pose_flat = recovered_body_pose.reshape(-1, 63)

        # Check shapes match
        assert orig_body_pose_flat.shape == recovered_body_pose_flat.shape
        assert orig_root_orient.shape == recovered_root_orient.shape

        # Translation might have floor alignment differences, so we check relative motion
        orig_trans_diff = orig_trans[1:] - orig_trans[:-1]
        recovered_trans_diff = recovered_trans[1:] - recovered_trans[:-1]

        # Check that relative translations are similar (allowing for some error)
        trans_error = np.linalg.norm(orig_trans_diff - recovered_trans_diff, axis=-1).mean()
        assert trans_error < 0.1, f"Translation error too large: {trans_error:.4f} meters"

        print("\nRound-trip conversion errors:")
        print(f"  Translation (relative): {trans_error:.6f} m")

    def test_multiple_frames_consistency(self, body_model_path):
        """Test round-trip with synthetic multi-frame motion."""
        from motion_toolbox.converters.motion_converters import Format272ToSMPLConverter, SMPLTo272Converter

        # Create synthetic SMPL data with 10 frames
        num_frames = 10
        smpl_data = {
            "pose_body": np.random.randn(num_frames, 63) * 0.1,  # Small random poses
            "root_orient": np.random.randn(num_frames, 3) * 0.1,  # Small random root orientations
            "trans": np.linspace([0, 0, 0], [1, 0, 0], num_frames),  # Linear motion forward
            "betas": np.zeros(10),  # Neutral body shape
        }

        # Convert SMPL → 272 → SMPL
        smpl_to_272 = SMPLTo272Converter(body_model_path=body_model_path)
        motion_272 = smpl_to_272.convert(smpl_data)

        format_272_to_smpl = Format272ToSMPLConverter(body_model_path=body_model_path)
        recovered_smpl = format_272_to_smpl.convert(motion_272)

        # Verify shapes
        assert motion_272.shape == (num_frames, 272)
        assert recovered_smpl["body_pose"].shape == torch.Size([num_frames, 63])
        assert recovered_smpl["global_orient"].shape == torch.Size([num_frames, 3])


class TestConverterIntegration:
    """Test integration with Motion class and visualization pipeline."""

    def test_272_to_motion_integration(self, body_model_path, sample_272_data):
        """Test that 272-dim can be converted to Motion class."""
        from core.motion import Motion

        from motion_toolbox.converters.motion_converters import Format272ToMotionConverter

        converter = Format272ToMotionConverter(body_model_path=body_model_path)
        motion = converter.convert(sample_272_data)

        assert isinstance(motion, Motion)
        assert motion.num_frames == sample_272_data.shape[0]
        assert motion.num_joints >= 22  # At least 22 joints

    def test_smpl_to_272_to_motion_pipeline(self, body_model_path, sample_smpl_data):
        """Test full pipeline: SMPL → 272 → Motion."""
        from core.motion import Motion

        from motion_toolbox.converters.motion_converters import Format272ToMotionConverter, SMPLTo272Converter

        # SMPL → 272
        smpl_to_272 = SMPLTo272Converter(body_model_path=body_model_path)
        motion_272 = smpl_to_272.convert(sample_smpl_data)

        # 272 → Motion
        format_272_to_motion = Format272ToMotionConverter(body_model_path=body_model_path)
        motion = format_272_to_motion.convert(motion_272)

        assert isinstance(motion, Motion)
        assert motion.num_frames == motion_272.shape[0]

        # Motion should have valid joint positions
        positions = motion.positions
        assert positions.shape == (motion.num_frames, motion.num_joints, 3)
        assert not np.isnan(positions).any()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_frame_motion(self, body_model_path):
        """Test conversion with single frame."""
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        smpl_data = {
            "pose_body": np.zeros((1, 63)),
            "root_orient": np.zeros((1, 3)),
            "trans": np.zeros((1, 3)),
            "betas": np.zeros(10),
        }

        converter = SMPLTo272Converter(body_model_path=body_model_path)
        result = converter.convert(smpl_data)

        assert result.shape == (1, 272)

    def test_invalid_input_format(self, body_model_path):
        """Test that invalid input raises appropriate error."""
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        converter = SMPLTo272Converter(body_model_path=body_model_path)

        with pytest.raises(ValueError):
            converter.convert("nonexistent_file.npz")

    def test_missing_smpl_keys(self, body_model_path):
        """Test that missing SMPL keys raises error."""
        from motion_toolbox.converters.motion_converters import SMPLTo272Converter

        incomplete_data = {
            "pose_body": np.zeros((5, 63)),
            # Missing root_orient and trans
        }

        converter = SMPLTo272Converter(body_model_path=body_model_path)

        with pytest.raises(ValueError, match="Missing required SMPL data keys"):
            converter.convert(incomplete_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
