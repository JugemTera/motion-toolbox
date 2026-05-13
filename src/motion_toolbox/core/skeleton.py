"""Skeleton data structure for kinematic chains.

This module provides an immutable Skeleton class that represents a kinematic chain
with joint names, parent-child relationships, and T-pose offsets.
"""

from dataclasses import dataclass

import numpy as np

from .mano_constants import (
    BIMANUAL_WRIST_SEPARATION_X,
    MANO_JOINT_NAMES_LEFT,
    MANO_JOINT_NAMES_RIGHT,
    MANO_PARENT_INDICES_SINGLE,
    MANO_T_POSE_OFFSETS_LEFT,
    MANO_T_POSE_OFFSETS_RIGHT,
)


@dataclass(frozen=True)
class Skeleton:
    """Immutable skeleton representation for kinematic chains.

    This class defines the structure of a skeleton with joint names, parent relationships,
    and T-pose offsets. It's designed to be immutable for thread safety and predictable behavior.

    Args:
        joint_names: Tuple of joint names in order
        parent_indices: Tuple of parent indices for each joint (None for root)
        offsets: 3D offset of each joint from its parent in T-pose, shape (num_joints, 3)

    Examples:
        >>> import numpy as np
        >>> # Simple 2-joint skeleton
        >>> joint_names = ("root", "child")
        >>> parent_indices = (None, 0)
        >>> offsets = np.array([[0, 0, 0], [0, 1, 0]])  # child is 1 unit up from root
        >>> skeleton = Skeleton(joint_names, parent_indices, offsets)
        >>> skeleton.num_joints
        2
        >>> skeleton.get_kinematic_chains()
        [(0, 1)]
    """

    joint_names: tuple[str, ...]
    parent_indices: tuple[int | None, ...]
    offsets: np.ndarray

    def __post_init__(self):
        """Validate skeleton data consistency."""
        # Check that all attributes have the same length
        if not (len(self.joint_names) == len(self.parent_indices) == len(self.offsets)):
            raise ValueError("joint_names, parent_indices, and offsets must have the same length")

        # Check offsets shape
        if self.offsets.shape != (len(self.joint_names), 3):
            raise ValueError(f"offsets must have shape ({len(self.joint_names)}, 3), got {self.offsets.shape}")

        # Validate parent indices
        for i, parent_idx in enumerate(self.parent_indices):
            if parent_idx is not None:
                if not (0 <= parent_idx < len(self.joint_names)):
                    raise ValueError(f"Invalid parent index {parent_idx} for joint {i}")
                if parent_idx >= i:
                    raise ValueError(f"Parent index {parent_idx} must be less than joint index {i}")

        # Ensure there's exactly one root (parent_index is None)
        root_count = sum(1 for p in self.parent_indices if p is None)
        if root_count != 1:
            raise ValueError(f"Skeleton must have exactly one root joint, found {root_count}")

        # Make offsets read-only
        self.offsets.flags.writeable = False

    @property
    def num_joints(self) -> int:
        """Number of joints in the skeleton."""
        return len(self.joint_names)

    @property
    def root_index(self) -> int:
        """Index of the root joint."""
        for i, parent in enumerate(self.parent_indices):
            if parent is None:
                return i
        raise RuntimeError("No root joint found")  # Should never happen due to validation

    def get_joint_index(self, joint_name: str) -> int:
        """Get the index of a joint by name.

        Args:
            joint_name: Name of the joint

        Returns:
            Index of the joint

        Raises:
            ValueError: If joint name is not found
        """
        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            raise ValueError(f"Joint '{joint_name}' not found in skeleton")

    def get_children(self, joint_index: int) -> list[int]:
        """Get the children of a joint.

        Args:
            joint_index: Index of the parent joint

        Returns:
            List of child joint indices
        """
        return [i for i, parent in enumerate(self.parent_indices) if parent == joint_index]

    def get_kinematic_chains(self) -> list[tuple[int, ...]]:
        """Get all kinematic chains from root to leaf joints.

        A kinematic chain is a sequence of connected joints from root to a leaf.
        This is useful for forward kinematics computation.

        Returns:
            List of kinematic chains, each as a tuple of joint indices
        """
        chains = []

        def build_chain(joint_idx: int, current_chain: list[int]):
            current_chain = current_chain + [joint_idx]
            children = self.get_children(joint_idx)

            if not children:  # Leaf joint
                chains.append(tuple(current_chain))
            else:
                for child in children:
                    build_chain(child, current_chain)

        build_chain(self.root_index, [])
        return chains

    def get_joint_path(self, joint_index: int) -> tuple[int, ...]:
        """Get the path from root to a specific joint.

        Args:
            joint_index: Index of the target joint

        Returns:
            Tuple of joint indices from root to target joint
        """
        path = []
        current = joint_index

        while current is not None:
            path.append(current)
            current = self.parent_indices[current]

        return tuple(reversed(path))

    def get_bone_connections(self) -> list[tuple[int, int]]:
        """Get bone connections as (parent, child) pairs for visualization.

        Returns:
            List of (parent_index, child_index) tuples representing bones
        """
        connections = []
        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx is not None:  # Skip root joint
                connections.append((parent_idx, child_idx))
        return connections

    def to_dict(self) -> dict:
        """Convert skeleton to dictionary representation.

        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "joint_names": list(self.joint_names),
            "parent_indices": [p if p is not None else -1 for p in self.parent_indices],
            "offsets": self.offsets.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skeleton":
        """Create skeleton from dictionary representation.

        Args:
            data: Dictionary with skeleton data

        Returns:
            Skeleton instance
        """
        joint_names = tuple(data["joint_names"])
        parent_indices = tuple(None if p == -1 else p for p in data["parent_indices"])
        offsets = np.array(data["offsets"])

        return cls(joint_names, parent_indices, offsets)

    @classmethod
    def from_smpl(cls) -> "Skeleton":
        """Create SMPL skeleton with standard joint names and hierarchy (24 joints).

        Returns:
            SMPL skeleton instance
        """
        # SMPL joint names (24 joints)
        joint_names = (
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hand",
            "right_hand",
        )

        # Parent relationships (pelvis is root)
        parent_indices = (None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)

        # SMPL T-pose offsets for Z-up coordinate system (AMASS format)
        # Transform from Y-up to Z-up: [X, Y, Z] -> [X, Z, Y]
        offsets = np.array(
            [
                [0.0, 0.0, 0.0],  # pelvis (root)
                [-0.1, 0.0, -0.05],  # left_hip (X, Z, Y) = (-0.1, 0.0, -0.05)
                [0.1, 0.0, -0.05],  # right_hip
                [0.0, 0.0, 0.1],  # spine1 (up in Z now)
                [0.0, 0.0, -0.4],  # left_knee (down in Z)
                [0.0, 0.0, -0.4],  # right_knee
                [0.0, 0.0, 0.1],  # spine2 (up in Z)
                [0.0, 0.0, -0.4],  # left_ankle
                [0.0, 0.0, -0.4],  # right_ankle
                [0.0, 0.0, 0.1],  # spine3 (up in Z)
                [0.0, 0.1, -0.1],  # left_foot (forward in Y, down in Z)
                [0.0, 0.1, -0.1],  # right_foot
                [0.0, 0.0, 0.2],  # neck (up in Z)
                [-0.15, 0.0, 0.0],  # left_collar
                [0.15, 0.0, 0.0],  # right_collar
                [0.0, 0.0, 0.15],  # head (up in Z)
                [-0.2, 0.0, 0.0],  # left_shoulder
                [0.2, 0.0, 0.0],  # right_shoulder
                [0.0, 0.0, -0.25],  # left_elbow (down in Z)
                [0.0, 0.0, -0.25],  # right_elbow
                [0.0, 0.0, -0.25],  # left_wrist
                [0.0, 0.0, -0.25],  # right_wrist (down in Z)
                [0.0, -0.05, 0.0],  # left_hand (forward in Y)
                [0.0, -0.05, 0.0],  # right_hand
            ]
        )

        return cls(joint_names, parent_indices, offsets)

    @classmethod
    def from_smplh(cls) -> "Skeleton":
        """Create SMPL+H skeleton with hand joints (52 joints total).

        This extends SMPL with detailed hand articulation compatible with HumanML3D.

        Returns:
            SMPL+H skeleton instance
        """
        # Start with base SMPL joints (24)
        base_names = [
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hand",
            "right_hand",
        ]

        # Add hand joints (28 additional: 14 per hand)
        left_hand_joints = [
            "left_index1",
            "left_index2",
            "left_index3",
            "left_middle1",
            "left_middle2",
            "left_middle3",
            "left_pinky1",
            "left_pinky2",
            "left_pinky3",
            "left_ring1",
            "left_ring2",
            "left_ring3",
            "left_thumb1",
            "left_thumb2",
        ]

        right_hand_joints = [
            "right_index1",
            "right_index2",
            "right_index3",
            "right_middle1",
            "right_middle2",
            "right_middle3",
            "right_pinky1",
            "right_pinky2",
            "right_pinky3",
            "right_ring1",
            "right_ring2",
            "right_ring3",
            "right_thumb1",
            "right_thumb2",
        ]

        joint_names = tuple(base_names + left_hand_joints + right_hand_joints)

        # Base parent indices (24)
        base_parents = [None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        # Hand parent indices - left hand joints (24-37)
        left_hand_parents = [
            22,
            24,
            25,  # index chain (parent: left_hand)
            22,
            27,
            28,  # middle chain
            22,
            30,
            31,  # pinky chain
            22,
            33,
            34,  # ring chain
            22,
            36,  # thumb chain
        ]

        # Right hand joints (38-51)
        right_hand_parents = [
            23,
            38,
            39,  # index chain (parent: right_hand)
            23,
            41,
            42,  # middle chain
            23,
            44,
            45,  # pinky chain
            23,
            47,
            48,  # ring chain
            23,
            50,  # thumb chain
        ]

        parent_indices = tuple(base_parents + left_hand_parents + right_hand_parents)

        # Base offsets (24)
        base_offsets = np.array(
            [
                [0.0, 0.0, 0.0],  # pelvis
                [-0.1, -0.05, 0.0],  # left_hip
                [0.1, -0.05, 0.0],  # right_hip
                [0.0, 0.1, 0.0],  # spine1
                [0.0, -0.4, 0.0],  # left_knee
                [0.0, -0.4, 0.0],  # right_knee
                [0.0, 0.1, 0.0],  # spine2
                [0.0, -0.4, 0.0],  # left_ankle
                [0.0, -0.4, 0.0],  # right_ankle
                [0.0, 0.1, 0.0],  # spine3
                [0.0, -0.1, 0.1],  # left_foot
                [0.0, -0.1, 0.1],  # right_foot
                [0.0, 0.2, 0.0],  # neck
                [-0.15, 0.0, 0.0],  # left_collar
                [0.15, 0.0, 0.0],  # right_collar
                [0.0, 0.15, 0.0],  # head
                [-0.2, 0.0, 0.0],  # left_shoulder
                [0.2, 0.0, 0.0],  # right_shoulder
                [0.0, -0.25, 0.0],  # left_elbow
                [0.0, -0.25, 0.0],  # right_elbow
                [0.0, -0.25, 0.0],  # left_wrist
                [0.0, -0.25, 0.0],  # right_wrist
                [0.0, -0.05, 0.0],  # left_hand
                [0.0, -0.05, 0.0],  # right_hand
            ]
        )

        # Hand offsets (28 additional joints)
        # Left hand finger offsets
        left_hand_offsets = np.array(
            [
                # Index finger
                [0.0, -0.03, 0.02],
                [0.0, -0.025, 0.0],
                [0.0, -0.02, 0.0],
                # Middle finger
                [0.0, -0.035, 0.005],
                [0.0, -0.03, 0.0],
                [0.0, -0.025, 0.0],
                # Pinky finger
                [0.0, -0.025, -0.015],
                [0.0, -0.02, 0.0],
                [0.0, -0.015, 0.0],
                # Ring finger
                [0.0, -0.03, -0.005],
                [0.0, -0.025, 0.0],
                [0.0, -0.02, 0.0],
                # Thumb
                [0.02, -0.02, 0.01],
                [0.02, -0.015, 0.0],
            ]
        )

        # Right hand finger offsets (mirror left)
        right_hand_offsets = left_hand_offsets.copy()
        right_hand_offsets[:, 0] *= -1  # Mirror X coordinate

        # Combine all offsets
        offsets = np.vstack([base_offsets, left_hand_offsets, right_hand_offsets])

        return cls(joint_names, parent_indices, offsets)

    @classmethod
    def from_smplx(cls) -> "Skeleton":
        """Create SMPL-X skeleton with hand and face joints (55 joints total).

        This extends SMPL+H with basic facial landmarks for full body capture.

        Returns:
            SMPL-X skeleton instance
        """
        # Start with SMPL+H skeleton
        smplh_skel = cls.from_smplh()

        # Add 3 facial landmark joints
        face_joints = ("jaw", "left_eye", "right_eye")

        joint_names = smplh_skel.joint_names + face_joints
        parent_indices = smplh_skel.parent_indices + (15, 15, 15)  # All parent to head

        # Face joint offsets relative to head
        face_offsets = np.array(
            [
                [0.0, -0.05, 0.08],  # jaw
                [-0.03, 0.02, 0.06],  # left_eye
                [0.03, 0.02, 0.06],  # right_eye
            ]
        )

        offsets = np.vstack([smplh_skel.offsets, face_offsets])

        return cls(joint_names, parent_indices, offsets)

    @classmethod
    def from_mano(cls, is_right: bool = False) -> "Skeleton":
        """Create MANO hand skeleton (16 joints).

        Args:
            is_right: If True, create right hand skeleton. Otherwise left.

        Returns:
            MANO hand skeleton instance
        """
        joint_names = MANO_JOINT_NAMES_RIGHT if is_right else MANO_JOINT_NAMES_LEFT
        offsets = MANO_T_POSE_OFFSETS_RIGHT if is_right else MANO_T_POSE_OFFSETS_LEFT

        return cls(joint_names, MANO_PARENT_INDICES_SINGLE, offsets)

    @classmethod
    def from_bimanual_mano(cls) -> "Skeleton":
        """Create bimanual MANO skeleton (32 joints: 16 left + 16 right).

        The left wrist is the root. The right wrist is parented to a virtual
        node at the midpoint, but for simplicity we make the right wrist a
        child of the left wrist with an offset representing typical hand spacing.

        Returns:
            Bimanual MANO skeleton instance
        """
        left = cls.from_mano(is_right=False)
        right = cls.from_mano(is_right=True)

        joint_names = left.joint_names + right.joint_names

        right_parents = []
        for i, p in enumerate(right.parent_indices):
            if p is None:
                right_parents.append(0)
            else:
                right_parents.append(p + left.num_joints)
        parent_indices = left.parent_indices + tuple(right_parents)

        right_offsets = right.offsets.copy()
        right_offsets[0] = np.array([BIMANUAL_WRIST_SEPARATION_X, 0.0, 0.0])

        offsets = np.vstack([left.offsets, right_offsets])

        return cls(joint_names, parent_indices, offsets)

    @classmethod
    def from_humanml3d(cls) -> "Skeleton":
        """Create HumanML3D compatible skeleton (22 joints).

        This is the standard skeleton used in the HumanML3D dataset,
        which is based on a subset of Human3.6M joints.

        Returns:
            HumanML3D skeleton instance
        """
        # HumanML3D joint order and names
        joint_names = (
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        )

        parent_indices = (None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

        # HumanML3D standard offsets (compatible with their preprocessing)
        offsets = np.array(
            [
                [0.0, 0.0, 0.0],  # pelvis
                [-0.09, -0.01, 0.0],  # left_hip
                [0.09, -0.01, 0.0],  # right_hip
                [0.0, 0.16, 0.0],  # spine1
                [0.0, -0.45, 0.0],  # left_knee
                [0.0, -0.45, 0.0],  # right_knee
                [0.0, 0.18, 0.0],  # spine2
                [0.0, -0.42, 0.0],  # left_ankle
                [0.0, -0.42, 0.0],  # right_ankle
                [0.0, 0.16, 0.0],  # spine3
                [0.0, -0.04, 0.15],  # left_foot
                [0.0, -0.04, 0.15],  # right_foot
                [0.0, 0.14, 0.0],  # neck
                [-0.12, 0.02, 0.0],  # left_collar
                [0.12, 0.02, 0.0],  # right_collar
                [0.0, 0.12, 0.0],  # head
                [-0.17, 0.0, 0.0],  # left_shoulder
                [0.17, 0.0, 0.0],  # right_shoulder
                [0.0, -0.28, 0.0],  # left_elbow
                [0.0, -0.28, 0.0],  # right_elbow
                [0.0, -0.24, 0.0],  # left_wrist
                [0.0, -0.24, 0.0],  # right_wrist
            ]
        )

        return cls(joint_names, parent_indices, offsets)

    @classmethod
    def from_h36m(cls) -> "Skeleton":
        """Create Human3.6M skeleton with standard joint names and hierarchy.

        Returns:
            Human3.6M skeleton instance
        """
        # Human3.6M joint names (32 joints, but commonly use 17)
        joint_names = (
            "hip",
            "right_hip",
            "right_knee",
            "right_ankle",
            "right_toe",
            "left_hip",
            "left_knee",
            "left_ankle",
            "left_toe",
            "spine",
            "thorax",
            "neck",
            "head",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        )

        parent_indices = (None, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 10, 16, 17)

        # Standard H36M T-pose offsets
        offsets = np.array(
            [
                [0.0, 0.0, 0.0],  # hip
                [0.12, -0.05, 0.0],  # right_hip
                [0.0, -0.45, 0.0],  # right_knee
                [0.0, -0.45, 0.0],  # right_ankle
                [0.0, 0.0, 0.15],  # right_toe
                [-0.12, -0.05, 0.0],  # left_hip
                [0.0, -0.45, 0.0],  # left_knee
                [0.0, -0.45, 0.0],  # left_ankle
                [0.0, 0.0, 0.15],  # left_toe
                [0.0, 0.15, 0.0],  # spine
                [0.0, 0.25, 0.0],  # thorax
                [0.0, 0.15, 0.0],  # neck
                [0.0, 0.15, 0.0],  # head
                [-0.18, 0.0, 0.0],  # left_shoulder
                [0.0, -0.3, 0.0],  # left_elbow
                [0.0, -0.25, 0.0],  # left_wrist
                [0.18, 0.0, 0.0],  # right_shoulder
                [0.0, -0.3, 0.0],  # right_elbow
                [0.0, -0.25, 0.0],  # right_wrist
            ]
        )

        return cls(joint_names, parent_indices, offsets)

    @classmethod
    def from_r15(cls) -> "Skeleton":
        """Create Roblox R15 skeleton with standard joint names and hierarchy (16 joints).

        The R15 skeleton is used by Roblox avatars and has a simplified hierarchy
        compared to SMPL. It consists of:
        - Root node
        - Lower/Upper torso and head
        - Left/Right arms (upper, lower, hand)
        - Left/Right legs (upper, lower, foot)

        Note: The scale factor between SMPL and R15 is approximately 0.0357.
        Offsets are in meters (Y-up coordinate system).

        Returns:
            R15 skeleton instance
        """
        # R15 joint names (16 joints)
        # Order matches the r15_bone_name_from_index from smpl_to_fbx.py
        joint_names = (
            "HumanoidRootNode",
            "LowerTorso",
            "UpperTorso",
            "Head",
            "LeftUpperArm",
            "LeftLowerArm",
            "LeftHand",
            "RightUpperArm",
            "RightLowerArm",
            "RightHand",
            "LeftUpperLeg",
            "LeftLowerLeg",
            "LeftFoot",
            "RightUpperLeg",
            "RightLowerLeg",
            "RightFoot",
        )

        # Parent relationships
        # HumanoidRootNode is root
        # LowerTorso -> HumanoidRootNode
        # UpperTorso -> LowerTorso
        # Head -> UpperTorso
        # Arms -> UpperTorso
        # Legs -> LowerTorso
        parent_indices = (
            None,  # HumanoidRootNode (root)
            0,  # LowerTorso -> HumanoidRootNode
            1,  # UpperTorso -> LowerTorso
            2,  # Head -> UpperTorso
            2,  # LeftUpperArm -> UpperTorso
            4,  # LeftLowerArm -> LeftUpperArm
            5,  # LeftHand -> LeftLowerArm
            2,  # RightUpperArm -> UpperTorso
            7,  # RightLowerArm -> RightUpperArm
            8,  # RightHand -> RightLowerArm
            1,  # LeftUpperLeg -> LowerTorso
            10,  # LeftLowerLeg -> LeftUpperLeg
            11,  # LeftFoot -> LeftLowerLeg
            1,  # RightUpperLeg -> LowerTorso
            13,  # RightLowerLeg -> RightUpperLeg
            14,  # RightFoot -> RightLowerLeg
        )

        # R15 T-pose offsets in meters (Y-up coordinate system)
        # Based on typical R15 proportions scaled to meters
        # Scale factor SMPL->R15 is 0.0357, so R15 is smaller
        # These offsets represent approximate bone lengths in a T-pose
        offsets = np.array(
            [
                [0.0, 0.0, 0.0],  # HumanoidRootNode (root)
                [0.0, 0.0, 0.0],  # LowerTorso (at root position)
                [0.0, 0.35, 0.0],  # UpperTorso (above LowerTorso)
                [0.0, 0.25, 0.0],  # Head (above UpperTorso)
                [-0.18, 0.15, 0.0],  # LeftUpperArm (shoulder offset from UpperTorso)
                [0.0, -0.22, 0.0],  # LeftLowerArm (below LeftUpperArm)
                [0.0, -0.22, 0.0],  # LeftHand (below LeftLowerArm)
                [0.18, 0.15, 0.0],  # RightUpperArm (shoulder offset from UpperTorso)
                [0.0, -0.22, 0.0],  # RightLowerArm (below RightUpperArm)
                [0.0, -0.22, 0.0],  # RightHand (below RightLowerArm)
                [-0.08, -0.05, 0.0],  # LeftUpperLeg (hip offset from LowerTorso)
                [0.0, -0.35, 0.0],  # LeftLowerLeg (below LeftUpperLeg)
                [0.0, -0.35, 0.0],  # LeftFoot (below LeftLowerLeg)
                [0.08, -0.05, 0.0],  # RightUpperLeg (hip offset from LowerTorso)
                [0.0, -0.35, 0.0],  # RightLowerLeg (below RightUpperLeg)
                [0.0, -0.35, 0.0],  # RightFoot (below RightLowerLeg)
            ]
        )

        return cls(joint_names, parent_indices, offsets)

    def __str__(self) -> str:
        """String representation of the skeleton."""
        chains = self.get_kinematic_chains()
        chain_strs = []
        for chain in chains:
            chain_names = [self.joint_names[i] for i in chain]
            chain_strs.append(" -> ".join(chain_names))

        return f"Skeleton({self.num_joints} joints):\n" + "\n".join(f"  {chain}" for chain in chain_strs)
