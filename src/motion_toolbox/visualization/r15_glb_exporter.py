"""R15 motion export utilities for Roblox avatar animation.

This module provides classes and functions for exporting motion sequences to GLB format
using the Roblox R15 skeleton. It uses the original R15 FBX (exported to GLB) as a base
and adds SMPL-converted animation data.

- R15GLBExporter: Class for exporting Motion instances to GLB files with R15 mesh
"""

import copy
from pathlib import Path

import numpy as np

try:
    from pygltflib import (
        ARRAY_BUFFER,
        ELEMENT_ARRAY_BUFFER,
        FLOAT,
        GLTF2,
        UNSIGNED_INT,
        Accessor,
        Animation,
        AnimationChannel,
        AnimationChannelTarget,
        AnimationSampler,
        BufferView,
        Material,
        Mesh,
        Node,
        PbrMetallicRoughness,
        Primitive,
    )
except ImportError:
    raise ImportError("Please install pygltflib: pip install pygltflib")

from ..converters.motion_converters import SMPLToR15Converter
from ..core.motion import Motion
from .gltf_utils import checkerboard_geometry

# Default paths: assets live at repository root (not under src/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_R15_GLB_PATH = _REPO_ROOT / "assets" / "r15" / "r15_itw_original.glb"
DEFAULT_R15_MESH_PATH = DEFAULT_R15_GLB_PATH  # Alias for backward compatibility


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to quaternion (x, y, z, w) for GLTF.

    Args:
        axis_angle: Axis-angle rotations with shape (..., 3)

    Returns:
        Quaternions with shape (..., 4) in (x, y, z, w) format for GLTF.
    """
    from scipy.spatial.transform import Rotation as R

    aa = np.asarray(axis_angle).copy()
    rot = R.from_rotvec(aa.reshape(-1, 3))
    q = rot.as_quat()  # (x, y, z, w) format - what GLTF uses
    return q.reshape(axis_angle.shape[:-1] + (4,)).astype(np.float32)


class R15GLBExporter:
    """Exports Motion instances to .glb using Roblox R15 mesh.

    This exporter converts SMPL motion data to R15 format and exports
    animated GLB files. It uses the original R15 FBX (exported to GLB by Blender)
    as a base and adds animation data.

    The approach:
    1. Load a pre-exported GLB of the R15 rig (with proper Y-up conversion)
    2. Convert SMPL motion to R15 joint rotations using SMPLToR15Converter
    3. Add animation data to the GLB

    Args:
        fps (int): Frames per second for animation (default 30).
        glb_path (str, optional): Path to R15 base GLB file.
    """

    # R15 joint names in order (matching SMPLToR15Converter output)
    R15_JOINT_NAMES = [
        "HumanoidRootNode",  # 0 - NOT animated (static root)
        "LowerTorso",  # 1 - gets pelvis rotation + translation
        "UpperTorso",  # 2 - spine1 @ spine2 @ spine3
        "Head",  # 3 - neck @ head
        "LeftUpperArm",  # 4 - l_collar @ l_shoulder + 90° Z
        "LeftLowerArm",  # 5 - l_elbow with alignment
        "LeftHand",  # 6 - l_wrist with alignment
        "RightUpperArm",  # 7 - r_collar @ r_shoulder - 90° Z
        "RightLowerArm",  # 8 - r_elbow with alignment
        "RightHand",  # 9 - r_wrist with alignment
        "LeftUpperLeg",  # 10 - l_hip
        "LeftLowerLeg",  # 11 - l_knee
        "LeftFoot",  # 12 - l_ankle
        "RightUpperLeg",  # 13 - r_hip
        "RightLowerLeg",  # 14 - r_knee
        "RightFoot",  # 15 - r_ankle
    ]

    def __init__(
        self,
        fps: int = 30,
        glb_path: str | None = None,
        include_floor: bool = True,
    ):
        self.fps = fps
        self.glb_path = glb_path or DEFAULT_R15_GLB_PATH
        self.include_floor = include_floor

        if not Path(self.glb_path).exists():
            raise FileNotFoundError(
                f"R15 GLB file not found: {self.glb_path}\nPlease ensure the R15 base GLB is available in assets/r15/"
            )

        # Load original GLB
        self.base_glb: GLTF2 = GLTF2.load(str(self.glb_path))  # type: ignore[assignment]

        # Build node name to index mapping
        self.node_name_to_idx = {}
        for i, node in enumerate(self.base_glb.nodes):
            self.node_name_to_idx[node.name] = i

    def export(self, motion: Motion, filename: str):
        """Export a Motion instance as animated .glb file.

        Args:
            motion: Motion object with SMPL skeleton (24+ joints)
            filename: Output path for the .glb file
        """
        # Convert SMPL motion to R15 format
        # apply_scale=False because our GLB was exported at ~27x scale (human-sized)
        # so translation should remain in meters (matching SMPL)
        converter = SMPLToR15Converter(apply_scale=False)
        r15_data = converter.convert(motion)

        joint_rotations = r15_data["joint_rotations"]  # (frames, 16, 3)
        root_translation = r15_data["root_translation"].copy()  # (frames, 3)

        # Floor alignment:
        # 1. ARMATURE_OFFSET: R15's Armature has a base Y of 0.918m
        # 2. LEG_LENGTH_DIFF: R15's legs are shorter than SMPL's
        # The bone hierarchy suggests R15 feet are lower, but visually they're higher
        # This is likely due to how the mesh is bound to bones
        # Empirical offset to align feet with floor
        ARMATURE_OFFSET = 0.918  # Armature rest Y
        LEG_LENGTH_DIFF = 0.25  # Increased from 0.20 - try different values
        FLOOR_OFFSET = ARMATURE_OFFSET + LEG_LENGTH_DIFF
        root_translation[:, 1] -= FLOOR_OFFSET

        gltf = self._add_animation(joint_rotations, root_translation)
        gltf.save(filename)
        print(f"Exported R15 animated GLB: {filename}")

    def _add_animation(
        self,
        joint_rotations: np.ndarray,
        root_translation: np.ndarray,
    ) -> GLTF2:
        """Add animation data to the base GLB.

        Args:
            joint_rotations: R15 joint rotations as axis-angles, shape (frames, 16, 3)
            root_translation: Root translation, shape (frames, 3)

        Returns:
            GLTF2 object with animation added
        """
        gltf: GLTF2 = copy.deepcopy(self.base_glb)  # type: ignore[assignment]

        num_frames = joint_rotations.shape[0]
        times = np.linspace(0, (num_frames - 1) / self.fps, num_frames).astype(np.float32)

        # Get existing binary blob
        existing_blob = gltf.binary_blob() or b""
        buffer_data = bytearray(existing_blob)
        buffer_offset = len(buffer_data)

        # Ensure buffer_offset is 4-byte aligned
        padding = (4 - buffer_offset % 4) % 4
        buffer_data += b"\x00" * padding
        buffer_offset += padding

        # === Time accessor ===
        times_data = times.tobytes()
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(times_data)))
        gltf.accessors.append(
            Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=FLOAT,
                count=num_frames,
                type="SCALAR",
                min=[float(times.min())],
                max=[float(times.max())],
            )
        )
        times_acc = len(gltf.accessors) - 1
        buffer_data += times_data
        buffer_offset += len(times_data)

        # === Translation accessor for LowerTorso ===
        trans_data = root_translation.astype(np.float32).tobytes()
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(trans_data)))
        gltf.accessors.append(
            Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=FLOAT,
                count=num_frames,
                type="VEC3",
            )
        )
        trans_acc = len(gltf.accessors) - 1
        buffer_data += trans_data
        buffer_offset += len(trans_data)

        # === Rotation accessors ===
        rotation_accessors = {}
        for r15_idx, joint_name in enumerate(self.R15_JOINT_NAMES):
            # Skip HumanoidRootNode (index 0) - it stays static, not animated
            if r15_idx == 0:
                continue

            if joint_name not in self.node_name_to_idx:
                print(f"Warning: Joint {joint_name} not found in GLB")
                continue

            # Get rotations for this joint
            joint_rot_aa = joint_rotations[:, r15_idx, :]  # (frames, 3)
            joint_quats = axis_angle_to_quat(joint_rot_aa)

            rot_data = joint_quats.astype(np.float32).tobytes()
            gltf.bufferViews.append(BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(rot_data)))
            gltf.accessors.append(
                Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    componentType=FLOAT,
                    count=num_frames,
                    type="VEC4",
                )
            )
            rotation_accessors[joint_name] = len(gltf.accessors) - 1
            buffer_data += rot_data
            buffer_offset += len(rot_data)

        # === Create animation ===
        samplers = []
        channels = []

        # Translation for LowerTorso (the functional root in R15)
        lower_torso_idx = self.node_name_to_idx.get("LowerTorso")
        if lower_torso_idx is not None:
            samplers.append(AnimationSampler(input=times_acc, output=trans_acc, interpolation="LINEAR"))
            channels.append(
                AnimationChannel(
                    sampler=len(samplers) - 1,
                    target=AnimationChannelTarget(node=lower_torso_idx, path="translation"),
                )
            )

        # Rotations for all animated joints (starting from LowerTorso)
        for joint_name, rot_acc in rotation_accessors.items():
            node_idx = self.node_name_to_idx.get(joint_name)
            if node_idx is not None:
                samplers.append(AnimationSampler(input=times_acc, output=rot_acc, interpolation="LINEAR"))
                channels.append(
                    AnimationChannel(
                        sampler=len(samplers) - 1,
                        target=AnimationChannelTarget(node=node_idx, path="rotation"),
                    )
                )

        gltf.animations = [Animation(name="R15_Animation", samplers=samplers, channels=channels)]

        # === Add checkerboard floor (optional) ===
        if self.include_floor:
            floor_verts, floor_faces, floor_colors, _ = checkerboard_geometry(length=4.0, tile_width=0.25)

            # Floor vertices
            floor_vert_data = floor_verts.astype(np.float32).tobytes()
            gltf.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=buffer_offset,
                    byteLength=len(floor_vert_data),
                    target=ARRAY_BUFFER,
                )
            )
            gltf.accessors.append(
                Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    componentType=FLOAT,
                    count=floor_verts.shape[0],
                    type="VEC3",
                    max=[float(x) for x in floor_verts.max(axis=0)],
                    min=[float(x) for x in floor_verts.min(axis=0)],
                )
            )
            floor_vert_acc = len(gltf.accessors) - 1
            buffer_data += floor_vert_data
            buffer_offset += len(floor_vert_data)

            # Floor indices
            floor_idx_data = floor_faces.flatten().astype(np.uint32).tobytes()
            gltf.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=buffer_offset,
                    byteLength=len(floor_idx_data),
                    target=ELEMENT_ARRAY_BUFFER,
                )
            )
            gltf.accessors.append(
                Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    componentType=UNSIGNED_INT,
                    count=floor_faces.size,
                    type="SCALAR",
                    max=[int(floor_faces.max())],
                    min=[int(floor_faces.min())],
                )
            )
            floor_idx_acc = len(gltf.accessors) - 1
            buffer_data += floor_idx_data
            buffer_offset += len(floor_idx_data)

            # Floor colors
            floor_color_data = floor_colors.astype(np.float32).tobytes()
            gltf.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=buffer_offset,
                    byteLength=len(floor_color_data),
                    target=ARRAY_BUFFER,
                )
            )
            gltf.accessors.append(
                Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    componentType=FLOAT,
                    count=floor_verts.shape[0],
                    type="VEC4",
                    max=[float(x) for x in floor_colors.max(axis=0)],
                    min=[float(x) for x in floor_colors.min(axis=0)],
                )
            )
            floor_color_acc = len(gltf.accessors) - 1
            buffer_data += floor_color_data
            buffer_offset += len(floor_color_data)

            # Floor material
            floor_material = Material(
                name="FloorMaterial",
                pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[1, 1, 1, 1]),
                doubleSided=True,
                alphaCutoff=None,
            )
            if gltf.materials is None:
                gltf.materials = []
            floor_mat_idx = len(gltf.materials)
            gltf.materials.append(floor_material)

            # Floor mesh
            floor_mesh = Mesh(
                name="checkerboard_floor",
                primitives=[
                    Primitive(
                        attributes={
                            "POSITION": floor_vert_acc,
                            "COLOR_0": floor_color_acc,
                        },
                        indices=floor_idx_acc,
                        material=floor_mat_idx,
                    )
                ],
            )
            if gltf.meshes is None:
                gltf.meshes = []
            floor_mesh_idx = len(gltf.meshes)
            gltf.meshes.append(floor_mesh)

            # Floor node
            floor_node = Node(name="checkerboard_floor", mesh=floor_mesh_idx)
            floor_node_idx = len(gltf.nodes)
            gltf.nodes.append(floor_node)

            # Add floor to scene
            if gltf.scenes and len(gltf.scenes) > 0:
                if gltf.scenes[0].nodes is None:
                    gltf.scenes[0].nodes = []
                gltf.scenes[0].nodes.append(floor_node_idx)

        # Update buffer size
        gltf.buffers[0].byteLength = len(buffer_data)
        gltf.set_binary_blob(bytes(buffer_data))

        return gltf
