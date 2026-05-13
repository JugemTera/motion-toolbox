"""Tests for the Skeleton class."""

import numpy as np
import pytest

from motion_toolbox.core.skeleton import Skeleton


class TestSkeletonBasics:
    """Test basic Skeleton functionality."""

    def test_simple_skeleton_creation(self):
        """Test creating a simple 2-joint skeleton."""
        joint_names = ("root", "child")
        parent_indices = (None, 0)
        offsets = np.array([[0, 0, 0], [0, 1, 0]])

        skeleton = Skeleton(joint_names, parent_indices, offsets)

        assert skeleton.num_joints == 2
        assert skeleton.joint_names == ("root", "child")
        assert skeleton.parent_indices == (None, 0)
        np.testing.assert_array_equal(skeleton.offsets, offsets)
        assert skeleton.root_index == 0

    def test_skeleton_immutability(self):
        """Test that skeleton is immutable."""
        joint_names = ("root", "child")
        parent_indices = (None, 0)
        offsets = np.array([[0, 0, 0], [0, 1, 0]])

        skeleton = Skeleton(joint_names, parent_indices, offsets)

        # Offsets should be read-only
        with pytest.raises((ValueError, RuntimeError)):
            skeleton.offsets[0] = [1, 2, 3]

        # Dataclass should be frozen
        with pytest.raises(AttributeError):
            skeleton.joint_names = ("new", "names")

    def test_get_joint_index(self):
        """Test getting joint index by name."""
        joint_names = ("pelvis", "spine", "head")
        parent_indices = (None, 0, 1)
        offsets = np.array([[0, 0, 0], [0, 0.1, 0], [0, 0.2, 0]])

        skeleton = Skeleton(joint_names, parent_indices, offsets)

        assert skeleton.get_joint_index("pelvis") == 0
        assert skeleton.get_joint_index("spine") == 1
        assert skeleton.get_joint_index("head") == 2

        with pytest.raises(ValueError):
            skeleton.get_joint_index("nonexistent")

    def test_get_children(self):
        """Test getting children of a joint."""
        # Create a branching skeleton: root -> (child1, child2)
        joint_names = ("root", "child1", "child2", "grandchild")
        parent_indices = (None, 0, 0, 1)
        offsets = np.array(
            [
                [0, 0, 0],  # root
                [-1, 1, 0],  # child1
                [1, 1, 0],  # child2
                [0, 1, 0],  # grandchild
            ]
        )

        skeleton = Skeleton(joint_names, parent_indices, offsets)

        # Root should have two children
        assert sorted(skeleton.get_children(0)) == [1, 2]
        # child1 should have one child
        assert skeleton.get_children(1) == [3]
        # child2 should have no children
        assert skeleton.get_children(2) == []
        # grandchild should have no children
        assert skeleton.get_children(3) == []


class TestKinematicChains:
    """Test kinematic chain functionality."""

    def test_linear_chain(self):
        """Test kinematic chains for a linear skeleton."""
        joint_names = ("root", "spine", "neck", "head")
        parent_indices = (None, 0, 1, 2)
        offsets = np.array([[0, 0, 0], [0, 0.1, 0], [0, 0.1, 0], [0, 0.1, 0]])

        skeleton = Skeleton(joint_names, parent_indices, offsets)
        chains = skeleton.get_kinematic_chains()

        # Should have one chain from root to head
        assert len(chains) == 1
        assert chains[0] == (0, 1, 2, 3)

    def test_branching_chains(self):
        """Test kinematic chains for a branching skeleton."""
        # Create Y-shaped skeleton: root -> spine -> (left_arm, right_arm)
        joint_names = ("root", "spine", "left_arm", "right_arm")
        parent_indices = (None, 0, 1, 1)
        offsets = np.array(
            [
                [0, 0, 0],  # root
                [0, 1, 0],  # spine
                [-1, 0, 0],  # left_arm
                [1, 0, 0],  # right_arm
            ]
        )

        skeleton = Skeleton(joint_names, parent_indices, offsets)
        chains = skeleton.get_kinematic_chains()

        # Should have two chains
        assert len(chains) == 2
        expected_chains = [(0, 1, 2), (0, 1, 3)]
        for chain in chains:
            assert chain in expected_chains

    def test_get_joint_path(self):
        """Test getting path from root to specific joint."""
        joint_names = ("root", "spine", "left_shoulder", "left_elbow", "left_wrist")
        parent_indices = (None, 0, 1, 2, 3)
        offsets = np.zeros((5, 3))

        skeleton = Skeleton(joint_names, parent_indices, offsets)

        # Path to root should be just root
        assert skeleton.get_joint_path(0) == (0,)

        # Path to left_elbow should be root -> spine -> left_shoulder -> left_elbow
        assert skeleton.get_joint_path(3) == (0, 1, 2, 3)

        # Path to left_wrist
        assert skeleton.get_joint_path(4) == (0, 1, 2, 3, 4)


class TestValidation:
    """Test skeleton validation."""

    def test_mismatched_lengths(self):
        """Test validation of mismatched attribute lengths."""
        with pytest.raises(ValueError, match="same length"):
            Skeleton(
                ("root", "child"),
                (None,),  # Wrong length
                np.array([[0, 0, 0], [0, 1, 0]]),
            )

    def test_wrong_offsets_shape(self):
        """Test validation of offsets shape."""
        with pytest.raises(ValueError, match="offsets must have shape"):
            Skeleton(
                ("root", "child"),
                (None, 0),
                np.array([[0, 0], [0, 1]]),  # Wrong shape - missing Z
            )

    def test_invalid_parent_index(self):
        """Test validation of parent indices."""
        # Parent index out of range
        with pytest.raises(ValueError, match="Invalid parent index"):
            Skeleton(
                ("root", "child"),
                (None, 5),  # Invalid parent index
                np.array([[0, 0, 0], [0, 1, 0]]),
            )

        # Parent index >= joint index (forward reference)
        with pytest.raises(ValueError, match="Parent index.*must be less than"):
            Skeleton(
                ("joint0", "joint1"),
                (1, None),  # joint0's parent is joint1 (forward ref)
                np.array([[0, 0, 0], [0, 1, 0]]),
            )

    def test_multiple_roots(self):
        """Test validation of single root requirement."""
        with pytest.raises(ValueError, match="exactly one root"):
            Skeleton(
                ("root1", "root2"),
                (None, None),  # Two roots
                np.array([[0, 0, 0], [1, 0, 0]]),
            )

    def test_no_root(self):
        """Test validation of root requirement (invalid parent chain)."""
        with pytest.raises(ValueError, match="Parent index|exactly one root"):
            Skeleton(
                ("child1", "child2"),
                (1, 0),  # Circular reference, no root
                np.array([[0, 0, 0], [1, 0, 0]]),
            )


class TestSerialization:
    """Test skeleton serialization and deserialization."""

    def test_to_dict(self):
        """Test converting skeleton to dictionary."""
        joint_names = ("root", "child")
        parent_indices = (None, 0)
        offsets = np.array([[0, 0, 0], [0, 1, 0]])

        skeleton = Skeleton(joint_names, parent_indices, offsets)
        data = skeleton.to_dict()

        expected = {
            "joint_names": ["root", "child"],
            "parent_indices": [-1, 0],  # None becomes -1
            "offsets": [[0, 0, 0], [0, 1, 0]],
        }

        assert data == expected

    def test_from_dict(self):
        """Test creating skeleton from dictionary."""
        data = {
            "joint_names": ["root", "child"],
            "parent_indices": [-1, 0],  # -1 becomes None
            "offsets": [[0, 0, 0], [0, 1, 0]],
        }

        skeleton = Skeleton.from_dict(data)

        assert skeleton.joint_names == ("root", "child")
        assert skeleton.parent_indices == (None, 0)
        np.testing.assert_array_equal(skeleton.offsets, np.array([[0, 0, 0], [0, 1, 0]]))

    def test_roundtrip_serialization(self):
        """Test that serialization preserves skeleton data."""
        original = Skeleton.from_h36m()
        data = original.to_dict()
        reconstructed = Skeleton.from_dict(data)

        assert original.joint_names == reconstructed.joint_names
        assert original.parent_indices == reconstructed.parent_indices
        np.testing.assert_array_equal(original.offsets, reconstructed.offsets)


class TestPrebuiltSkeletons:
    """Test prebuilt skeleton creation."""

    def test_smpl_skeleton(self):
        """Test SMPL skeleton creation."""
        skeleton = Skeleton.from_smpl()

        assert skeleton.num_joints == 24
        assert skeleton.joint_names[0] == "pelvis"
        assert skeleton.root_index == 0

        # Check that skeleton is valid
        chains = skeleton.get_kinematic_chains()
        assert len(chains) > 0

        # All joints should be reachable from root
        all_joints_in_chains = set()
        for chain in chains:
            all_joints_in_chains.update(chain)
        assert len(all_joints_in_chains) == skeleton.num_joints

    def test_h36m_skeleton(self):
        """Test Human3.6M skeleton creation."""
        skeleton = Skeleton.from_h36m()

        assert skeleton.num_joints == 19
        assert skeleton.joint_names[0] == "hip"
        assert skeleton.root_index == 0

        # Check that skeleton is valid
        chains = skeleton.get_kinematic_chains()
        assert len(chains) > 0

        # All joints should be reachable from root
        all_joints_in_chains = set()
        for chain in chains:
            all_joints_in_chains.update(chain)
        assert len(all_joints_in_chains) == skeleton.num_joints


class TestStringRepresentation:
    """Test string representation of skeleton."""

    def test_str_representation(self):
        """Test string representation includes key info."""
        skeleton = Skeleton.from_smpl()
        str_repr = str(skeleton)

        assert "Skeleton(24 joints)" in str_repr
        assert "pelvis" in str_repr
        assert "->" in str_repr  # Should show kinematic chains


if __name__ == "__main__":
    pytest.main([__file__])
