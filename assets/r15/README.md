# R15 Avatar Assets

This directory contains the Roblox R15 avatar assets used for motion visualization.

## Files

- `r15_itw_original.glb` - The main R15 GLB file used by `R15GLBExporter`. This is the original R15 FBX from the autoanim-itw-motion-dataset repo, exported to GLB by Blender.

- `r15_rig.fbx` - A copy of the original R15 FBX file from the itw repo.

- `convert_fbx_to_glb.py` - Blender script to regenerate the GLB from FBX if needed.

## Important: Scale Factor

The R15 FBX uses centimeter units. Blender's FBX importer applies a default scale of 0.01, resulting in a **6.4cm tall** character.

To match SMPL's height (~1.7m), use `global_scale=26.6`:
- Original imported height: 0.064m (6.4cm)
- Target height: 1.7m (to match SMPL)
- Scale factor: 1.7 / 0.064 = **26.6**

## Regenerating the GLB

If you need to regenerate `r15_itw_original.glb`:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --factory-startup --background --python-expr "
import bpy

# Clear scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)

# Import FBX with scale=26.6 to match SMPL height (1.7m)
# The original FBX is in cm, Blender imports at 0.01 scale = 6.4cm
# Scale factor: 1.7m / 0.064m = 26.6
bpy.ops.import_scene.fbx(filepath='r15_rig.fbx', global_scale=26.6)

# Select all
for obj in bpy.data.objects:
    obj.select_set(True)

# Export to GLB with Y-up
bpy.ops.export_scene.gltf(
    filepath='r15_itw_original.glb',
    export_format='GLB',
    use_selection=True,
    export_animations=False,
    export_skins=True,
    export_yup=True,
)
"
```

## Usage

The R15 avatar is used by `R15GLBExporter` to create animated GLB files from SMPL motion data:

```python
from motion_toolbox.visualization import R15GLBExporter

exporter = R15GLBExporter(fps=30)
exporter.export(motion, "output.glb")
```

## R15 Skeleton

The R15 skeleton has 16 animated joints:

| Index | Joint Name | SMPL Mapping |
|-------|-----------|--------------|
| 0 | HumanoidRootNode | Static (not animated) |
| 1 | LowerTorso | Pelvis + translation |
| 2 | UpperTorso | Spine1 @ Spine2 @ Spine3 |
| 3 | Head | Neck @ Head |
| 4 | LeftUpperArm | L_Collar @ L_Shoulder + 90° Z |
| 5 | LeftLowerArm | L_Elbow with alignment |
| 6 | LeftHand | L_Wrist with alignment |
| 7 | RightUpperArm | R_Collar @ R_Shoulder - 90° Z |
| 8 | RightLowerArm | R_Elbow with alignment |
| 9 | RightHand | R_Wrist with alignment |
| 10 | LeftUpperLeg | L_Hip |
| 11 | LeftLowerLeg | L_Knee |
| 12 | LeftFoot | L_Ankle |
| 13 | RightUpperLeg | R_Hip |
| 14 | RightLowerLeg | R_Knee |
| 15 | RightFoot | R_Ankle |

Key differences from SMPL:
- HumanoidRootNode is static; LowerTorso is the functional root
- Translation is applied to LowerTorso, not the root
- Arms have 90° Z corrections for T-pose alignment
- Lower arms have 80° Z alignment corrections
