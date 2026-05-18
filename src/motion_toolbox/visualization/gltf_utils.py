"""Utilities for building GLTF/GLB files for motion visualization."""

import numpy as np
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    GLTF2,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationSampler,
    Buffer,
    BufferView,
    Material,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Scene,
    Skin,
)

# SMPL standard names (first 24)
SMPL_JOINT_NAMES = [
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


def get_joint_name(index: int, total_joints: int) -> str:
    """Get a safe joint name, using SMPL names if applicable."""
    if total_joints == 24 and index < len(SMPL_JOINT_NAMES):
        return SMPL_JOINT_NAMES[index]
    # Use generic names for other models (MHR, etc) or indices > 24
    return f"joint_{index}"


def axis_angle_to_quat(axis_angle):
    """Convert axis-angle representation to quaternion."""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    small_angle = angle < 1e-8

    axis = np.zeros_like(axis_angle)
    axis[~small_angle[..., 0]] = axis_angle[~small_angle[..., 0]] / angle[~small_angle[..., 0]]
    axis[small_angle[..., 0]] = np.array([1.0, 0.0, 0.0])

    half_angle = angle * 0.5
    sin_half = np.sin(half_angle)
    quat = np.concatenate([axis * sin_half, np.cos(half_angle)], axis=-1)  # (x, y, z, w)
    return quat.astype(np.float32)


def checkerboard_geometry(
    length=12.0,
    color0=[0.8, 0.9, 0.9],
    color1=[0.6, 0.7, 0.7],
    tile_width=0.5,
    alpha=1.0,
    up="y",
    c1=0.0,
    c2=0.0,
):
    """Generate checkerboard floor geometry."""
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    num_rows = num_cols = max(2, int(length / tile_width))
    radius = float(num_rows * tile_width) / 2.0
    vertices = []
    vert_colors = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)
                cur_verts[:, 0] += c1
                cur_verts[:, 2] += c2
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)
                cur_verts[:, 0] += c1
                cur_verts[:, 1] += c2

            cur_faces = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
            cur_faces += 4 * (i * num_cols + j)
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.float32)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors


def build_gltf(
    vertices,
    init_joints,
    faces,
    joints,
    global_orient,
    transl,
    body_pose,
    parents,
    fps=30,
    include_floor=True,
    lbs_weights=None,
):
    """Build GLTF2 structure with animation data."""
    gltf = GLTF2()
    num_joints = joints.shape[1]
    num_frames = joints.shape[0]

    # Use the first frame's joints as the rest-pose template for skinning.
    J_template = init_joints[0]
    num_vertices = vertices.shape[1]

    # Prepare skinning data
    if lbs_weights is None:
        raise ValueError("lbs_weights must be provided")

    # Convert to 4 joints per vertex format
    joints_per_vertex = []
    weights_per_vertex = []

    for v_idx in range(num_vertices):
        vertex_weights = lbs_weights[v_idx]
        # Find the 4 most influential joints
        joint_indices = np.argsort(vertex_weights)[-4:][::-1]
        joint_weights = vertex_weights[joint_indices]
        # Normalize weights
        weight_sum = joint_weights.sum()
        if weight_sum > 0:
            joint_weights = joint_weights / weight_sum
        else:
            joint_indices = np.array([0, 0, 0, 0], dtype=np.uint16)
            joint_weights = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        joints_per_vertex.extend(joint_indices.astype(np.uint16))
        weights_per_vertex.extend(joint_weights.astype(np.float32))

    buffer_data = b""
    buffer_views = []
    buffer_offset = 0
    accessors = []

    # BUFFER 0: Vertex Position Data
    vert_data = vertices[0].astype(np.float32).tobytes()
    buffer_data += vert_data
    vert_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(vert_data), target=ARRAY_BUFFER)
    buffer_offset += len(vert_data)
    buffer_views.append(vert_bv)
    vert_acc = Accessor(
        bufferView=len(buffer_views) - 1,
        componentType=FLOAT,
        count=vertices.shape[1],
        type="VEC3",
        max=[float(x) for x in vertices[0].max(axis=0)],
        min=[float(x) for x in vertices[0].min(axis=0)],
    )
    accessors.append(vert_acc)
    vertices_accessor_idx = len(accessors) - 1

    # BUFFER 1: Face Indices Data
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

    # BUFFER 2: Inverse Bind Matrices
    ibm = np.zeros((num_joints, 4, 4), dtype=np.float32)
    world_transforms = np.zeros((num_joints, 4, 4), dtype=np.float32)

    # Build hierarchy transforms
    for joint_idx in range(num_joints):
        transform = np.eye(4, dtype=np.float32)
        parent_idx = int(parents[joint_idx])  # ensure int

        if parent_idx == -1:
            transform[:3, 3] = J_template[joint_idx]
        else:
            local_pos = J_template[joint_idx] - J_template[parent_idx]
            transform[:3, 3] = local_pos
            transform = np.dot(world_transforms[parent_idx], transform)
        world_transforms[joint_idx] = transform
        ibm[joint_idx] = np.linalg.inv(transform)

    ibm_column_major = ibm.transpose(0, 2, 1).astype(np.float32)
    ibm_data = ibm_column_major.tobytes()
    buffer_data += ibm_data
    ibm_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(ibm_data))
    buffer_offset += len(ibm_data)
    buffer_views.append(ibm_bv)
    ibm_acc = Accessor(bufferView=len(buffer_views) - 1, componentType=FLOAT, count=num_joints, type="MAT4")
    accessors.append(ibm_acc)
    ibm_accessor_idx = len(accessors) - 1

    # BUFFER 3: Vertex Joint Indices
    joints_data = np.array(joints_per_vertex, dtype=np.uint16).tobytes()
    buffer_data += joints_data
    joints_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(joints_data), target=ARRAY_BUFFER)
    buffer_offset += len(joints_data)
    buffer_views.append(joints_bv)
    joints_acc = Accessor(
        bufferView=len(buffer_views) - 1, componentType=UNSIGNED_SHORT, count=num_vertices, type="VEC4"
    )
    accessors.append(joints_acc)
    Joint_accessor_idx = len(accessors) - 1

    # BUFFER 4: Vertex Joint Weights
    weights_data = np.array(weights_per_vertex, dtype=np.float32).tobytes()
    buffer_data += weights_data
    weights_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(weights_data), target=ARRAY_BUFFER)
    buffer_offset += len(weights_data)
    buffer_views.append(weights_bv)
    weights_acc = Accessor(bufferView=len(buffer_views) - 1, componentType=FLOAT, count=num_vertices, type="VEC4")
    accessors.append(weights_acc)
    weights_accessor_idx = len(accessors) - 1

    # BUFFER 5: Animation Time Data
    times = np.arange(num_frames, dtype=np.float32) / fps
    times_data = times.tobytes()
    buffer_data += times_data
    times_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(times_data))
    buffer_offset += len(times_data)
    buffer_views.append(times_bv)
    times_acc = Accessor(
        bufferView=len(buffer_views) - 1,
        componentType=FLOAT,
        count=num_frames,
        type="SCALAR",
        max=[float(times.max())],
        min=[float(times.min())],
    )
    accessors.append(times_acc)
    times_accessor_idx = len(accessors) - 1

    # BUFFER 6: Joint Translation Animation Data
    trans_accessor_idx = len(accessors)
    trans_data = joints[:, 0, :].tobytes()
    buffer_data += trans_data
    trans_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(trans_data))
    buffer_offset += len(trans_data)
    buffer_views.append(trans_bv)
    trans_acc = Accessor(bufferView=len(buffer_views) - 1, componentType=FLOAT, count=num_frames, type="VEC3")
    accessors.append(trans_acc)

    # BUFFER 7+: Joint Rotation Animation Data

    # Handle potentially flat body_pose (e.g. from MHR where it's just a long vector)
    # or structured (frames, joints, 3)
    if body_pose.ndim == 2:
        # Assume it's flattened rotations
        # We need to know how many joints are covered.
        # global_orient is 1 joint (root)
        # So body_pose covers num_joints - 1

        expected_channels = (num_joints - 1) * 3
        if body_pose.shape[1] >= expected_channels:
            # Truncate extra params (like scale) if any
            body_pose_reshaped = body_pose[:, :expected_channels].reshape(num_frames, num_joints - 1, 3)
        else:
            # Pad if missing (shouldn't happen if logic is correct)
            pad_size = expected_channels - body_pose.shape[1]
            padding = np.zeros((num_frames, pad_size), dtype=body_pose.dtype)
            body_pose_reshaped = np.concatenate([body_pose, padding], axis=1).reshape(num_frames, num_joints - 1, 3)
    else:
        body_pose_reshaped = body_pose

    # Combine with global orient
    full_pose = np.concatenate([global_orient[:, np.newaxis, :], body_pose_reshaped], axis=1)

    rotation_accessor_start = len(accessors)

    # Verify joint count match
    if full_pose.shape[1] != num_joints:
        # If we have mismatch, we only animate what we have
        # This can happen if model_parameters includes non-rotational data we failed to strip
        pass

    for joint_idx in range(min(num_joints, full_pose.shape[1])):
        joint_rotations = []
        for frame_idx in range(num_frames):
            aa = full_pose[frame_idx, joint_idx]
            quat = axis_angle_to_quat(aa)
            joint_rotations.extend(quat.tolist())

        rot_data = np.array(joint_rotations, dtype=np.float32).tobytes()
        buffer_data += rot_data
        rot_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(rot_data))
        buffer_offset += len(rot_data)
        buffer_views.append(rot_bv)
        rot_acc = Accessor(bufferView=len(buffer_views) - 1, componentType=FLOAT, count=num_frames, type="VEC4")
        accessors.append(rot_acc)

    # Floor Geometry
    if include_floor:
        floor_vertices, floor_faces, floor_vert_colors, floor_face_colors = checkerboard_geometry()
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

    # Construct glTF nodes
    node_list = []
    for jid in range(num_joints):
        parent = int(parents[jid])
        children = [c for c, p in enumerate(parents) if int(p) == jid and c != jid]

        if parent == -1:
            local_translation = J_template[jid]
        else:
            local_translation = J_template[jid] - J_template[parent]

        name = get_joint_name(jid, num_joints)

        node = Node(
            name=name,
            children=children if children else [],
            translation=[float(x) for x in local_translation],
            rotation=[0.0, 0.0, 0.0, 1.0],
            scale=[1.0, 1.0, 1.0],
        )
        node_list.append(node)

    mesh = Mesh(
        name="BodyMesh",
        primitives=[
            Primitive(
                attributes={
                    "POSITION": vertices_accessor_idx,
                    "JOINTS_0": Joint_accessor_idx,
                    "WEIGHTS_0": weights_accessor_idx,
                },
                indices=1,
            )
        ],
    )

    meshes = [mesh]
    if include_floor:
        floor_material = Material(
            name="FloorMaterial",
            pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[1, 1, 1, 1]),
            doubleSided=True,
        )
        gltf.materials = [floor_material]
        floor_mesh = Mesh(
            name="checkerboard_floor",
            primitives=[
                Primitive(
                    attributes={"POSITION": floor_vert_acc_idx, "COLOR_0": floor_color_acc_idx},
                    indices=floor_idx_acc_idx,
                    material=0,
                )
            ],
        )
        meshes.append(floor_mesh)

    gltf.meshes = meshes

    skin = Skin(name="BodySkin", joints=list(range(num_joints)), inverseBindMatrices=ibm_accessor_idx)
    gltf.skins = [skin]

    mesh_node = Node(name="Body", mesh=0, skin=0)
    node_list.append(mesh_node)
    # Attach mesh to root (usually node 0)
    node_list[0].children.append(len(node_list) - 1)

    if include_floor:
        floor_node = Node(name="checkerboard_floor", mesh=1)
        node_list.append(floor_node)

    gltf.nodes = node_list

    if include_floor:
        gltf.scenes = [Scene(nodes=[0, len(node_list) - 1])]
    else:
        gltf.scenes = [Scene(nodes=[0])]

    gltf.scene = 0
    gltf.asset = {"version": "2.0", "generator": "MotionExport"}
    gltf.buffers = [Buffer(byteLength=len(buffer_data))]
    gltf.bufferViews = buffer_views
    gltf.accessors = accessors
    gltf.set_binary_blob(buffer_data)

    # Animation
    samplers = []
    channels = []

    # Translation (Root)
    sampler_trans = AnimationSampler(input=times_accessor_idx, output=trans_accessor_idx, interpolation="LINEAR")
    samplers.append(sampler_trans)
    chan_trans = AnimationChannel(sampler=len(samplers) - 1, target={"node": 0, "path": "translation"})
    channels.append(chan_trans)

    # Rotations
    # Only add channels for joints we actually have rotation data for
    max_anim_joints = min(num_joints, full_pose.shape[1])

    for joint_idx in range(max_anim_joints):
        rot_accessor_idx = rotation_accessor_start + joint_idx
        sampler_rot = AnimationSampler(input=times_accessor_idx, output=rot_accessor_idx, interpolation="LINEAR")
        samplers.append(sampler_rot)
        chan_rot = AnimationChannel(sampler=len(samplers) - 1, target={"node": joint_idx, "path": "rotation"})
        channels.append(chan_rot)

    animation = Animation(samplers=samplers, channels=channels)
    gltf.animations = [animation]

    return gltf
