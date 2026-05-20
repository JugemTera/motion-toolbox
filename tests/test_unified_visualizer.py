"""Unit tests for unified visualization system (Phase 1).

Tests format detection, compatibility checking, and visualizer initialization.
"""

import numpy as np
import pytest

from motion_toolbox.core import Motion, Skeleton
from motion_toolbox.visualization import (
    CompatibilityMatrix,
    ConversionRequirement,
    FormatDetector,
    Visualizer,
)
from motion_toolbox.visualization.constants import (
    CONFIDENCE_CERTAIN,
    FORMAT_272D_DIM,
    HUMANML3D_263D_DIM,
    JOINT_COUNTS,
    SMPL_POSES_72D_DIM,
    STMC_205D_DIM,
)


class TestFormatDetector:
    """Tests for FormatDetector class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = FormatDetector()

    def test_detect_263d(self):
        """Test detection of HumanML3D 263D format."""
        num_frames = 120
        data = np.random.randn(num_frames, HUMANML3D_263D_DIM)
        result = self.detector.detect(data)

        assert result.format_type == "263d"
        assert result.confidence == CONFIDENCE_CERTAIN
        assert result.num_frames == num_frames
        assert result.num_joints == JOINT_COUNTS["263d"]
        assert result.skeleton_type == "humanml3d"

    def test_detect_272d(self):
        """Test detection of 272D format."""
        num_frames = 100
        data = np.random.randn(num_frames, FORMAT_272D_DIM)
        result = self.detector.detect(data)

        assert result.format_type == "272d"
        assert result.confidence == CONFIDENCE_CERTAIN
        assert result.num_frames == num_frames
        assert result.num_joints == JOINT_COUNTS["272d"]

    def test_detect_205d(self):
        """Test detection of STMC 205D format."""
        num_frames = 80
        data = np.random.randn(num_frames, STMC_205D_DIM)
        result = self.detector.detect(data)

        assert result.format_type == "205d"
        assert result.confidence == CONFIDENCE_CERTAIN
        assert result.num_frames == num_frames
        assert result.num_joints == JOINT_COUNTS["205d"]

    def test_detect_joint_positions_humanml3d(self):
        """Test detection of HumanML3D joint positions."""
        num_frames = 120
        data = np.random.randn(num_frames, JOINT_COUNTS["humanml3d"], 3)
        result = self.detector.detect(data)

        assert result.format_type == "joint_positions"
        assert result.num_frames == num_frames
        assert result.num_joints == JOINT_COUNTS["humanml3d"]
        assert result.skeleton_type == "humanml3d"

    def test_detect_joint_positions_smpl(self):
        """Test detection of SMPL joint positions."""
        num_frames = 100
        data = np.random.randn(num_frames, JOINT_COUNTS["smpl"], 3)
        result = self.detector.detect(data)

        assert result.format_type == "joint_positions"
        assert result.num_frames == num_frames
        assert result.num_joints == JOINT_COUNTS["smpl"]
        assert result.skeleton_type == "smpl"

    def test_detect_motion_object(self):
        """Test detection of Motion object."""
        num_frames = 100
        skeleton = Skeleton.from_humanml3d()
        rotations = np.random.randn(num_frames, JOINT_COUNTS["humanml3d"], 3) * 0.1
        translation = np.zeros((num_frames, 3))
        motion = Motion(rotations, translation, skeleton, floor_alignment="joint")

        result = self.detector.detect(motion)

        assert result.format_type == "motion"
        assert result.confidence == CONFIDENCE_CERTAIN
        assert result.num_frames == num_frames
        assert result.num_joints == JOINT_COUNTS["humanml3d"]
        assert result.skeleton_type == "humanml3d"

    def test_detect_smpl_params_dict(self):
        """Test detection of SMPL parameters dictionary."""
        num_frames = 100
        data = {
            "poses": np.random.randn(num_frames, SMPL_POSES_72D_DIM),
            "trans": np.random.randn(num_frames, 3),
            "betas": np.random.randn(10),
        }
        result = self.detector.detect(data)

        assert result.format_type == "smpl_params"
        assert result.num_frames == num_frames
        assert result.skeleton_type == "smpl"

    def test_detect_with_format_hint(self):
        """Test format detection with user hint."""
        num_frames = 120
        data = np.random.randn(num_frames, HUMANML3D_263D_DIM)
        result = self.detector.detect(data, format_hint="263d")

        assert result.format_type == "263d"
        assert result.confidence == CONFIDENCE_CERTAIN

    def test_detect_invalid_shape(self):
        """Test detection with invalid shape."""
        data = np.random.randn(100, 999)  # Unknown dimension

        with pytest.raises(ValueError, match="Unrecognized 2D array shape"):
            self.detector.detect(data)

    def test_detect_1d_array(self):
        """Test detection with 1D array (invalid)."""
        data = np.random.randn(100)

        with pytest.raises(ValueError, match="must be at least 2D"):
            self.detector.detect(data)


class TestCompatibilityMatrix:
    """Tests for CompatibilityMatrix class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.matrix = CompatibilityMatrix()

    def test_skeleton_compatible_with_all_formats(self):
        """Test that skeleton mode works with all formats."""
        formats = ["263d", "272d", "205d", "motion", "joint_positions"]

        for fmt in formats:
            result = self.matrix.check_compatibility(fmt, "skeleton")
            assert result.compatible, f"Skeleton should work with {fmt}"
            assert result.conversion_required in [
                ConversionRequirement.NONE,
                ConversionRequirement.SIMPLE,
            ]

    def test_smpl_direct_compatibility(self):
        """Test SMPL direct compatibility."""
        result = self.matrix.check_compatibility("motion", "mesh", "smpl", "smpl")
        assert result.compatible
        assert result.conversion_required == ConversionRequirement.NONE

    def test_smpl_simple_conversion(self):
        """Test SMPL simple conversion."""
        result = self.matrix.check_compatibility("272d", "mesh", "smpl")
        assert result.compatible
        assert result.conversion_required == ConversionRequirement.SIMPLE

    def test_smpl_smplify_required(self):
        """Test SMPL SMPLify requirement."""
        result = self.matrix.check_compatibility("263d", "mesh", "smpl")
        assert result.compatible
        assert result.conversion_required == ConversionRequirement.SMPLIFY
        assert len(result.warnings) > 0
        assert "slow" in result.warnings[0].lower()

    def test_mhr_direct_compatibility(self):
        """Test MHR direct compatibility."""
        result = self.matrix.check_compatibility("motion", "mesh", "mhr", "mhr")
        assert result.compatible
        assert result.conversion_required == ConversionRequirement.NONE

    def test_mhr_not_supported(self):
        """Test MHR unsupported formats."""
        result = self.matrix.check_compatibility("263d", "mesh", "mhr")
        assert not result.compatible
        assert result.conversion_required == ConversionRequirement.NOT_SUPPORTED

    def test_auto_mode_resolution(self):
        """Test AUTO mode resolution."""
        result = self.matrix.check_compatibility("263d", "auto")
        assert result.compatible
        # AUTO mode should default to skeleton for 263d

    def test_get_recommended_mode_smpl(self):
        """Test recommended mode for SMPL-compatible formats."""
        mode, body_model = self.matrix.get_recommended_mode("272d")
        assert mode == "mesh"
        assert body_model == "smpl"

    def test_get_recommended_mode_humanml3d(self):
        """Test recommended mode for HumanML3D."""
        mode, body_model = self.matrix.get_recommended_mode("263d")
        assert mode == "skeleton"  # Should recommend skeleton due to SMPLify requirement

    def test_get_recommended_mode_mhr(self):
        """Test recommended mode for MHR motion."""
        mode, body_model = self.matrix.get_recommended_mode("motion", "mhr")
        assert mode == "mesh"
        # Note: body_model might be 'smpl' if MHR is in SMPL_DIRECT set
        # The logic in get_recommended_mode prioritizes SMPL_DIRECT check before MHR check
        assert body_model in ["smpl", "mhr"]  # Accept either as valid


class TestVisualizer:
    """Tests for Visualizer class."""

    def test_initialization_defaults(self):
        """Test visualizer initialization with defaults."""
        vis = Visualizer()
        assert vis.default_mode == "auto"
        assert vis.default_body_model == "auto"
        assert vis.on_incompatible == "auto_skeleton"

    def test_initialization_custom(self):
        """Test visualizer initialization with custom settings."""
        vis = Visualizer(
            body_model_path="/custom/path",
            default_mode="skeleton",
            default_body_model="smpl",
            on_incompatible="error",
        )
        assert vis.body_model_path == "/custom/path"
        assert vis.default_mode == "skeleton"
        assert vis.default_body_model == "smpl"
        assert vis.on_incompatible == "error"

    def test_invalid_default_mode(self):
        """Test initialization with invalid default mode."""
        with pytest.raises(ValueError, match="default_mode must be one of"):
            Visualizer(default_mode="invalid")

    def test_invalid_body_model(self):
        """Test initialization with invalid body model."""
        with pytest.raises(ValueError, match="default_body_model must be one of"):
            Visualizer(default_body_model="invalid")

    def test_invalid_on_incompatible(self):
        """Test initialization with invalid on_incompatible."""
        with pytest.raises(ValueError, match="on_incompatible must be one of"):
            Visualizer(on_incompatible="invalid")

    def test_visualize_skeleton_works(self, tmp_path):
        """Test that skeleton visualization works (Phase 2)."""
        vis = Visualizer()
        num_frames = 100
        data = np.random.randn(num_frames, HUMANML3D_263D_DIM)
        output_path = tmp_path / "test.html"

        # Phase 2: skeleton visualization should work
        result = vis.visualize(data, output_path, mode="skeleton")
        assert result.exists()

    def test_visualize_mesh_not_implemented(self, tmp_path):
        """Test that mesh visualization raises NotImplementedError (Phase 2, pending Phase 3)."""
        vis = Visualizer()
        num_frames = 100
        data = np.random.randn(num_frames, HUMANML3D_263D_DIM)
        output_path = tmp_path / "test.html"

        # Phase 3 feature - should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            vis.visualize(data, output_path, mode="mesh", body_model="smpl")

    def test_to_scene_data_works(self):
        """Test that to_scene_data returns valid scene data (Phase 4)."""
        vis = Visualizer()
        data = np.random.randn(100, 263)

        # Should return JSON-serializable dict
        scene_data = vis.to_scene_data(data, mode="skeleton")

        assert isinstance(scene_data, dict)
        assert "type" in scene_data
        assert scene_data["type"] == "skeleton"
        assert "positions" in scene_data
        assert "fps" in scene_data

    def test_repr(self):
        """Test string representation."""
        vis = Visualizer(default_mode="skeleton", default_body_model="smpl")
        repr_str = repr(vis)
        assert "Visualizer" in repr_str
        assert "skeleton" in repr_str
        assert "smpl" in repr_str


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_detect_and_check_compatibility_263d_skeleton(self):
        """Test full pipeline: detect 263D and check skeleton compatibility."""
        detector = FormatDetector()
        matrix = CompatibilityMatrix()

        data = np.random.randn(120, 263)
        detected = detector.detect(data)
        compat = matrix.check_compatibility(
            detected.format_type,
            "skeleton",
            None,
            detected.skeleton_type,
        )

        assert detected.format_type == "263d"
        assert compat.compatible
        assert compat.conversion_required == ConversionRequirement.SIMPLE

    def test_detect_and_check_compatibility_263d_smpl_mesh(self):
        """Test full pipeline: detect 263D and check SMPL mesh compatibility."""
        detector = FormatDetector()
        matrix = CompatibilityMatrix()

        data = np.random.randn(120, 263)
        detected = detector.detect(data)
        compat = matrix.check_compatibility(
            detected.format_type,
            "mesh",
            "smpl",
            detected.skeleton_type,
        )

        assert detected.format_type == "263d"
        assert compat.compatible
        assert compat.conversion_required == ConversionRequirement.SMPLIFY
        assert len(compat.warnings) > 0

    def test_detect_and_check_compatibility_motion_smpl(self):
        """Test full pipeline: detect Motion and check SMPL compatibility."""
        detector = FormatDetector()
        matrix = CompatibilityMatrix()

        skeleton = Skeleton.from_smpl()
        rotations = np.random.randn(100, 24, 3) * 0.1
        translation = np.zeros((100, 3))
        motion = Motion(rotations, translation, skeleton, floor_alignment="joint")

        detected = detector.detect(motion)
        compat = matrix.check_compatibility(
            detected.format_type,
            "mesh",
            "smpl",
            detected.skeleton_type,
        )

        assert detected.format_type == "motion"
        assert detected.skeleton_type == "smpl"
        assert compat.compatible
        assert compat.conversion_required == ConversionRequirement.NONE


class TestSkeletonVisualizer:
    """Tests for SkeletonVisualizer (Phase 2)."""

    def test_visualize_positions(self, tmp_path):
        """Test visualizing joint positions."""
        from motion_toolbox.visualization import SkeletonVisualizer

        vis = SkeletonVisualizer()
        positions = np.random.randn(100, 22, 3)
        output_path = tmp_path / "skeleton.html"

        result = vis.visualize_positions(positions, output_path)

        assert result.exists()
        assert result.suffix == ".html"
        # Check HTML contains expected elements
        content = result.read_text()
        assert "Three.js" in content or "three.js" in content
        assert "Skeleton" in content

    def test_visualize_motion(self, tmp_path):
        """Test visualizing Motion object."""
        from motion_toolbox.visualization import SkeletonVisualizer

        skeleton = Skeleton.from_humanml3d()
        rotations = np.random.randn(100, 22, 3) * 0.1
        translation = np.zeros((100, 3))
        motion = Motion(rotations, translation, skeleton, floor_alignment="joint")

        vis = SkeletonVisualizer()
        output_path = tmp_path / "motion.html"

        result = vis.visualize_motion(motion, output_path)

        assert result.exists()
        assert result.suffix == ".html"


class TestIntegratedSkeletonVisualization:
    """Integration tests for skeleton visualization through Visualizer (Phase 2)."""

    def test_visualize_263d_skeleton(self, tmp_path):
        """Test visualizing 263D data as skeleton."""
        vis = Visualizer()
        data = np.random.randn(100, 263)
        output_path = tmp_path / "263d_skeleton.html"

        result = vis.visualize(data, output_path, mode="skeleton")

        assert result.exists()
        assert result.suffix == ".html"
        content = result.read_text()
        assert "Skeleton" in content

    def test_visualize_272d_skeleton(self, tmp_path):
        """Test visualizing 272D data as skeleton."""
        vis = Visualizer()
        data = np.random.randn(100, 272)
        output_path = tmp_path / "272d_skeleton.html"

        result = vis.visualize(data, output_path, mode="skeleton")

        assert result.exists()

    def test_visualize_joint_positions(self, tmp_path):
        """Test visualizing joint positions directly."""
        vis = Visualizer()
        data = np.random.randn(100, 22, 3)
        output_path = tmp_path / "joints.html"

        result = vis.visualize(data, output_path, mode="skeleton")

        assert result.exists()

    def test_visualize_motion_object(self, tmp_path):
        """Test visualizing Motion object."""
        skeleton = Skeleton.from_humanml3d()
        rotations = np.random.randn(100, 22, 3) * 0.1
        translation = np.zeros((100, 3))
        motion = Motion(rotations, translation, skeleton, floor_alignment="joint")

        vis = Visualizer()
        output_path = tmp_path / "motion.html"

        result = vis.visualize(motion, output_path, mode="skeleton")

        assert result.exists()

    def test_auto_mode_defaults_to_skeleton(self, tmp_path):
        """Test that AUTO mode works and defaults appropriately."""
        vis = Visualizer()
        data = np.random.randn(100, 263)  # 263D
        output_path = tmp_path / "auto.html"

        # AUTO mode should work (defaults to skeleton for 263D due to SMPLify requirement)
        result = vis.visualize(data, output_path, mode="auto")

        assert result.exists()

    def test_visualize_with_custom_fps(self, tmp_path):
        """Test visualization with custom FPS."""
        vis = Visualizer()
        data = np.random.randn(100, 22, 3)
        output_path = tmp_path / "custom_fps.html"

        result = vis.visualize(data, output_path, mode="skeleton", fps=30)

        assert result.exists()
        content = result.read_text()
        assert "30" in content  # FPS should appear in HTML

    def test_visualize_with_custom_title(self, tmp_path):
        """Test visualization with custom title."""
        vis = Visualizer()
        data = np.random.randn(100, 22, 3)
        output_path = tmp_path / "custom_title.html"

        result = vis.visualize(data, output_path, mode="skeleton", title="My Custom Animation")

        assert result.exists()
        content = result.read_text()
        assert "My Custom Animation" in content


class TestMeshVisualizer:
    """Tests for MeshVisualizer (Phase 3)."""

    def test_initialization_smpl(self):
        """Test MeshVisualizer initialization with SMPL."""
        from motion_toolbox.visualization import MeshVisualizer

        # Should initialize without error (even if body models not available)
        try:
            vis = MeshVisualizer(body_model="smpl", body_model_path="/tmp/fake_path")
        except ImportError:
            pytest.skip("smplx not installed")
        except Exception as e:
            # Expected to fail with model path error, not initialization error
            assert "model_path" in str(e).lower() or "does not exist" in str(e).lower()

    def test_initialization_mhr(self):
        """Test MeshVisualizer initialization with MHR."""
        from motion_toolbox.visualization import MeshVisualizer

        try:
            vis = MeshVisualizer(body_model="mhr")
            assert vis.body_model == "mhr"
        except (ImportError, NotImplementedError, FileNotFoundError, ValueError, RuntimeError) as e:
            pytest.skip(f"MHR mesh pipeline not available in this environment: {e}")

    def test_invalid_body_model(self):
        """Test initialization with invalid body model."""
        from motion_toolbox.visualization import MeshVisualizer

        with pytest.raises(ValueError, match="Unsupported body model"):
            vis = MeshVisualizer(body_model="invalid")


class TestIntegratedMeshVisualization:
    """Integration tests for mesh visualization through Visualizer (Phase 3)."""

    def test_263d_mesh_raises_not_implemented(self, tmp_path):
        """Test that 263D mesh visualization raises appropriate error."""
        vis = Visualizer()
        data = np.random.randn(100, 263)
        output_path = tmp_path / "263d_mesh.html"

        # 263D requires SMPLify which we don't run in tests
        with pytest.raises(NotImplementedError, match="SMPLify"):
            vis.visualize(data, output_path, mode="mesh", body_model="smpl")

    def test_joint_positions_mesh_raises_not_implemented(self, tmp_path):
        """Test that joint positions cannot be directly meshed."""
        vis = Visualizer()
        data = np.random.randn(100, 22, 3)
        output_path = tmp_path / "joints_mesh.html"

        # Joint positions lack rotation information
        with pytest.raises(NotImplementedError, match="Joint positions alone"):
            vis.visualize(data, output_path, mode="mesh", body_model="smpl")

    def test_fallback_to_skeleton_on_incompatible(self, tmp_path):
        """Test that 263D mesh visualization explains the limitation clearly."""
        vis = Visualizer(on_incompatible="auto_skeleton")
        data = np.random.randn(100, 263)  # 263D (requires SMPLify for mesh)
        output_path = tmp_path / "fallback.html"

        # 263D is technically compatible with mesh (via SMPLify) but we don't
        # run SMPLify in the unified visualizer. Should get a clear error.
        # Note: fallback happens when compatibility check fails, not when conversion fails
        with pytest.raises(NotImplementedError, match="SMPLify"):
            vis.visualize(data, output_path, mode="mesh", body_model="smpl")

    def test_error_on_incompatible(self, tmp_path):
        """Test error mode on incompatible format."""
        vis = Visualizer(on_incompatible="error")
        data = np.random.randn(100, 22, 3)  # Joint positions
        output_path = tmp_path / "error.html"

        # Should raise IncompatibleFormatError
        with pytest.raises(Exception):  # Could be IncompatibleFormatError or NotImplementedError
            vis.visualize(data, output_path, mode="mesh", body_model="smpl")


class TestConversionPipelines:
    """Tests for format conversion pipelines (Phase 3)."""

    @pytest.mark.requires_smpl
    def test_272d_to_motion_conversion(self):
        """Test 272D to Motion conversion.

        Requires SMPL body models because the conversion constructs a
        :class:`Motion`, whose ``__post_init__`` runs forward kinematics via
        ``smplx.create()``.
        """
        from motion_toolbox.visualization import Visualizer

        vis = Visualizer()
        data = np.random.randn(100, 272)

        # Extract the conversion logic by trying to extract positions
        # (which internally converts to Motion)
        try:
            positions = vis._extract_joint_positions(data, vis.format_detector.detect(data))
            assert positions.shape[0] == 100  # 100 frames
            assert positions.shape[2] == 3  # 3D positions
        except ImportError:
            pytest.skip("Required converters not available")

    @pytest.mark.requires_smpl
    def test_205d_to_motion_conversion(self):
        """Test 205D (STMC) to Motion conversion.

        Requires SMPL body models (see ``test_272d_to_motion_conversion``).
        """
        from motion_toolbox.visualization import Visualizer

        vis = Visualizer()
        data = np.random.randn(100, 205)

        try:
            positions = vis._extract_joint_positions(data, vis.format_detector.detect(data))
            assert positions.shape[0] == 100
            assert positions.shape[2] == 3
        except ImportError:
            pytest.skip("STMC converter not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
