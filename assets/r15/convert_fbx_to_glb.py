"""Blender Python script to convert R15 FBX to clean GLB.

This script:
1. Imports the R15 FBX with proper scaling (28x)
2. Keeps only the _Geo body meshes (removes _Att attachment meshes)
3. Exports a clean GLB with armature and skinning

Usage:
    blender --factory-startup --background --python convert_fbx_to_glb.py -- --input r15_rig.fbx --output r15_rig.glb
"""

import argparse
import sys
import os

try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    sys.exit(1)


# Scale factor to convert FBX units to meters (character should be ~1.7m tall)
FBX_SCALE = 28.0


def get_args():
    """Parse command line arguments."""
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]
    except ValueError:
        args = []
    
    parser = argparse.ArgumentParser(description="Convert R15 FBX to clean GLB")
    parser.add_argument("--input", type=str, default="r15_rig.fbx", help="Input FBX file path")
    parser.add_argument("--output", type=str, default="r15_rig.glb", help="Output GLB file path")
    parser.add_argument("--scale", type=float, default=FBX_SCALE, help=f"Scale factor (default: {FBX_SCALE})")
    
    return parser.parse_args(args)


def clear_scene():
    """Remove all objects and data from the scene."""
    # Remove all objects
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    
    # Remove all mesh data
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    
    # Remove all armature data
    for arm in list(bpy.data.armatures):
        bpy.data.armatures.remove(arm)
    
    # Remove cameras and lights (default Blender objects)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)


def import_fbx(filepath, scale):
    """Import FBX file with specified scale."""
    print(f"Importing FBX with scale {scale}...")
    bpy.ops.import_scene.fbx(filepath=filepath, global_scale=scale)
    
    # Count imported objects
    mesh_count = sum(1 for obj in bpy.data.objects if obj.type == 'MESH')
    armature_count = sum(1 for obj in bpy.data.objects if obj.type == 'ARMATURE')
    print(f"Imported: {mesh_count} meshes, {armature_count} armature(s)")


def filter_meshes():
    """Keep only _Geo body meshes, remove _Att attachments and other objects."""
    armature = None
    geo_meshes = []
    to_delete = []
    
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
        elif obj.type == 'MESH':
            if '_Geo' in obj.name:
                geo_meshes.append(obj)
                print(f"  Keeping: {obj.name}")
            else:
                to_delete.append(obj)
                print(f"  Removing: {obj.name}")
        else:
            # Remove any other object types
            to_delete.append(obj)
    
    # Delete unwanted objects
    for obj in to_delete:
        bpy.data.objects.remove(obj)
    
    # Clean up orphan mesh data
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    
    print(f"\nKept {len(geo_meshes)} body meshes + armature")
    return geo_meshes, armature


def export_glb(filepath, meshes, armature):
    """Export selected objects to GLB."""
    print(f"\nExporting to GLB: {filepath}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Select all objects for export
    for obj in bpy.data.objects:
        obj.select_set(True)
    
    if armature:
        bpy.context.view_layer.objects.active = armature
    
    # Export
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        use_selection=True,
        export_animations=False,
        export_skins=True,
        export_yup=True,
    )
    
    print(f"Export complete: {filepath}")


def main():
    args = get_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(script_dir, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)
    
    print("=" * 60)
    print("R15 FBX to GLB Converter")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Scale:  {args.scale}")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    clear_scene()
    import_fbx(input_path, args.scale)
    meshes, armature = filter_meshes()
    
    if not meshes:
        print("ERROR: No body meshes found!")
        sys.exit(1)
    
    export_glb(output_path, meshes, armature)
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
