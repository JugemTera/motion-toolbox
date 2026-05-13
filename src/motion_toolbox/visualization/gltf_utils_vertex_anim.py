"""GLB builder for vertex animation (morph targets).

This module creates GLB files with vertex animation using morph targets,
suitable for models like MHR that don't use SMPL-style skeletal animation.
"""

import numpy as np
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
    Attributes,
    Buffer,
    BufferView,
    Mesh,
    Node,
    Primitive,
    Scene,
)

from .gltf_utils import checkerboard_geometry


def build_gltf_vertex_animation(
    all_vertices: np.ndarray,  # (num_frames, num_verts, 3)
    init_vertices: np.ndarray,  # (1, num_verts, 3) - base mesh
    faces: np.ndarray,
    fps: int = 30,
    include_floor: bool = True,
) -> GLTF2:
    """Build a GLTF with vertex animation using morph targets.

    Args:
        all_vertices: Vertex positions for all frames (num_frames, num_verts, 3)
        init_vertices: Initial vertex positions for the base mesh (1, num_verts, 3)
        faces: Face indices
        fps: Frames per second
        include_floor: Whether to include floor geometry

    Returns:
        GLTF2 object
    """
    gltf = GLTF2()

    num_frames = all_vertices.shape[0]
    num_vertices = all_vertices.shape[1]

    buffer_data = b""
    buffer_views = []
    buffer_offset = 0
    accessors = []

    # ================================
    # BUFFER 0: Base Vertex Positions
    # ================================
    base_verts = init_vertices[0].astype(np.float32)
    vert_data = base_verts.tobytes()
    buffer_data += vert_data

    vert_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(vert_data), target=ARRAY_BUFFER)
    buffer_offset += len(vert_data)
    buffer_views.append(vert_bv)

    vert_acc = Accessor(
        bufferView=len(buffer_views) - 1,
        componentType=FLOAT,
        count=num_vertices,
        type="VEC3",
        max=[float(x) for x in base_verts.max(axis=0)],
        min=[float(x) for x in base_verts.min(axis=0)],
    )
    accessors.append(vert_acc)
    base_vertices_accessor_idx = len(accessors) - 1

    # ================================
    # BUFFER 1: Face Indices
    # ================================
    idx_data = faces.flatten().astype(np.uint32).tobytes()
    buffer_data += idx_data

    idx_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(idx_data), target=ELEMENT_ARRAY_BUFFER)
    buffer_offset += len(idx_data)
    buffer_views.append(idx_bv)

    idx_acc = Accessor(
        bufferView=len(buffer_views) - 1,
        componentType=UNSIGNED_INT,
        count=faces.size,
        type="SCALAR",
        max=[int(faces.max())],
        min=[int(faces.min())],
    )
    accessors.append(idx_acc)
    face_indices_accessor_idx = len(accessors) - 1

    # ================================
    # BUFFERS 2+: Morph Target Displacements
    # ================================
    # Each morph target stores the displacement from base mesh
    morph_target_accessors = []

    for frame_idx in range(num_frames):
        # Compute displacement from base mesh
        displacement = (all_vertices[frame_idx] - base_verts).astype(np.float32)
        morph_data = displacement.tobytes()
        buffer_data += morph_data

        morph_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(morph_data), target=ARRAY_BUFFER)
        buffer_offset += len(morph_data)
        buffer_views.append(morph_bv)

        morph_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=num_vertices,
            type="VEC3",
            max=[float(x) for x in displacement.max(axis=0)],
            min=[float(x) for x in displacement.min(axis=0)],
        )
        accessors.append(morph_acc)
        morph_target_accessors.append(len(accessors) - 1)

    # ================================
    # Animation Time Accessor
    # ================================
    time_data = np.linspace(0, (num_frames - 1) / fps, num_frames).astype(np.float32).tobytes()
    buffer_data += time_data

    time_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(time_data))
    buffer_offset += len(time_data)
    buffer_views.append(time_bv)

    time_acc = Accessor(
        bufferView=len(buffer_views) - 1,
        componentType=FLOAT,
        count=num_frames,
        type="SCALAR",
        max=[float((num_frames - 1) / fps)],
        min=[0.0],
    )
    accessors.append(time_acc)
    time_accessor_idx = len(accessors) - 1

    # ================================
    # Animation Weights Accessor
    # ================================
    # Create weights for each frame: only one morph target active at a time
    weights_data_list = []
    for frame_idx in range(num_frames):
        frame_weights = np.zeros(num_frames, dtype=np.float32)
        frame_weights[frame_idx] = 1.0
        weights_data_list.append(frame_weights.tobytes())

    weights_data = b"".join(weights_data_list)
    buffer_data += weights_data

    weights_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(weights_data))
    buffer_offset += len(weights_data)
    buffer_views.append(weights_bv)

    weights_acc = Accessor(
        bufferView=len(buffer_views) - 1,
        componentType=FLOAT,
        count=num_frames * num_frames,  # total number of weight values (keyframes * morph_targets)
        type="SCALAR",
        max=[1.0],
        min=[0.0],
    )
    accessors.append(weights_acc)
    weights_accessor_idx = len(accessors) - 1

    # ================================
    # Floor Geometry (Optional)
    # ================================
    floor_node_idx = None
    if include_floor:
        floor_vertices, floor_faces, floor_vert_colors, _ = checkerboard_geometry()

        # Floor vertices
        floor_vert_data = floor_vertices.astype(np.float32).tobytes()
        buffer_data += floor_vert_data

        floor_vert_bv = BufferView(
            buffer=0, byteOffset=buffer_offset, byteLength=len(floor_vert_data), target=ARRAY_BUFFER
        )
        buffer_offset += len(floor_vert_data)
        buffer_views.append(floor_vert_bv)

        floor_vert_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=floor_vertices.shape[0],
            type="VEC3",
            max=[float(x) for x in floor_vertices.max(axis=0)],
            min=[float(x) for x in floor_vertices.min(axis=0)],
        )
        accessors.append(floor_vert_acc)
        floor_vert_acc_idx = len(accessors) - 1

        # Floor vertex colors
        floor_color_data = floor_vert_colors.astype(np.float32).tobytes()
        buffer_data += floor_color_data

        floor_color_bv = BufferView(
            buffer=0, byteOffset=buffer_offset, byteLength=len(floor_color_data), target=ARRAY_BUFFER
        )
        buffer_offset += len(floor_color_data)
        buffer_views.append(floor_color_bv)

        floor_color_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=floor_vertices.shape[0],
            type="VEC4",
            max=[float(x) for x in floor_vert_colors.max(axis=0)],
            min=[float(x) for x in floor_vert_colors.min(axis=0)],
        )
        accessors.append(floor_color_acc)
        floor_color_acc_idx = len(accessors) - 1

        # Floor indices
        floor_idx_data = floor_faces.flatten().astype(np.uint32).tobytes()
        buffer_data += floor_idx_data

        floor_idx_bv = BufferView(
            buffer=0, byteOffset=buffer_offset, byteLength=len(floor_idx_data), target=ELEMENT_ARRAY_BUFFER
        )
        buffer_offset += len(floor_idx_data)
        buffer_views.append(floor_idx_bv)

        floor_idx_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=UNSIGNED_INT,
            count=floor_faces.size,
            type="SCALAR",
            max=[int(floor_faces.max())],
            min=[int(floor_faces.min())],
        )
        accessors.append(floor_idx_acc)
        floor_idx_acc_idx = len(accessors) - 1

        # Floor mesh with vertex colors
        floor_prim = Primitive(
            attributes=Attributes(POSITION=floor_vert_acc_idx, COLOR_0=floor_color_acc_idx),
            indices=floor_idx_acc_idx,
        )
        floor_mesh = Mesh(primitives=[floor_prim], name="Floor")
        gltf.meshes = [floor_mesh]

        floor_node = Node(mesh=len(gltf.meshes) - 1, name="FloorNode")
        gltf.nodes = [floor_node]
        floor_node_idx = len(gltf.nodes) - 1

    # ================================
    # Character Mesh with Morph Targets
    # ================================
    # Create morph targets list
    morph_targets = []
    for morph_acc_idx in morph_target_accessors:
        morph_targets.append({"POSITION": morph_acc_idx})

    char_prim = Primitive(
        attributes=Attributes(POSITION=base_vertices_accessor_idx),
        indices=face_indices_accessor_idx,
        targets=morph_targets,
    )
    char_mesh = Mesh(primitives=[char_prim], name="CharacterMesh")
    if not gltf.meshes:
        gltf.meshes = []
    gltf.meshes.append(char_mesh)
    char_mesh_idx = len(gltf.meshes) - 1

    char_node = Node(
        mesh=char_mesh_idx,
        name="Character",
    )
    # Set weights as an attribute after creation
    char_node.weights = [0.0] * num_frames  # Initial weights (all morph targets off)

    if not gltf.nodes:
        gltf.nodes = []
    gltf.nodes.append(char_node)
    char_node_idx = len(gltf.nodes) - 1

    # ================================
    # Animation
    # ================================
    anim_sampler = AnimationSampler(
        input=time_accessor_idx,
        output=weights_accessor_idx,
        interpolation="LINEAR",
    )

    anim_channel = AnimationChannel(
        sampler=0,
        target=AnimationChannelTarget(node=char_node_idx, path="weights"),
    )

    animation = Animation(
        samplers=[anim_sampler],
        channels=[anim_channel],
    )
    gltf.animations = [animation]

    # ================================
    # Scene
    # ================================
    if floor_node_idx is not None:
        scene_nodes = [floor_node_idx, char_node_idx]
    else:
        scene_nodes = [char_node_idx]

    scene = Scene(nodes=scene_nodes)
    gltf.scenes = [scene]
    gltf.scene = 0

    # ================================
    # Finalize
    # ================================
    buffer = Buffer(byteLength=len(buffer_data))
    gltf.buffers = [buffer]
    gltf.bufferViews = buffer_views
    gltf.accessors = accessors
    gltf.set_binary_blob(buffer_data)

    return gltf
