# Hand Motion Format Reference (format_hand, 393D)

## Overview

The 393-dimensional bimanual hand motion representation is the hand equivalent of the
272-dim body format. It captures two-hand motion using MANO-compatible joint structure
with canonicalized positions, velocities, and 6D rotation representations.

## Layout

| Slice       | Dims | Description                                         |
|-------------|------|-----------------------------------------------------|
| `[0:3]`     | 3    | Root velocity 3D (midpoint of wrists, canonical frame) |
| `[3:9]`     | 6    | Root angular velocity 6D (orientation delta)        |
| `[9:105]`   | 96   | Joint positions: 32 joints x 3                      |
| `[105:201]` | 96   | Joint velocities: 32 joints x 3                     |
| `[201:393]` | 192  | Joint rotations: 32 joints x 6D                     |
| **Total**   | **393** |                                                  |

## Joint Order (32 joints)

Joints 0-15: **Left hand** (MANO order)
Joints 16-31: **Right hand** (MANO order)

Per-hand joint order (16 joints):

| Index | Name     | Parent |
|-------|----------|--------|
| 0     | wrist    | (root) |
| 1     | index1   | 0      |
| 2     | index2   | 1      |
| 3     | index3   | 2      |
| 4     | middle1  | 0      |
| 5     | middle2  | 4      |
| 6     | middle3  | 5      |
| 7     | ring1    | 0      |
| 8     | ring2    | 7      |
| 9     | ring3    | 8      |
| 10    | pinky1   | 0      |
| 11    | pinky2   | 10     |
| 12    | pinky3   | 11     |
| 13    | thumb1   | 0      |
| 14    | thumb2   | 13     |
| 15    | thumb3   | 14     |

## Key Differences from Body 272D

1. **3D root velocity** (not 2D XZ) -- hands move freely in space.
2. **Reference frame** -- defined by the midpoint of both wrists (origin) and the
   wrist-to-wrist direction (X axis), with Y up and Z forward. For body 272, the
   reference is the pelvis with yaw heading on the ground plane.
3. **32 joints** (2x 16 MANO) instead of 22 SMPL joints.
4. **No foot contacts** -- not applicable for hands.

## Canonical Frame Construction

1. Origin = midpoint(left_wrist, right_wrist)
2. X axis = direction from left wrist to right wrist (normalized)
3. Y axis = world up (0, 1, 0)
4. Z axis = cross(X, Y) -- forward direction
5. Y axis is recomputed as cross(Z, X) for orthogonality.

All positions are expressed relative to this frame. The root velocity captures
how the midpoint moves in this canonical frame. Angular velocity captures how
the canonical frame itself rotates between frames.

## Converters

- `MANOToHandRepConverter`: MANO bimanual params -> 393D
- `HandRepToMANOConverter`: 393D -> MANO bimanual params
- `FormatHandToMotionConverter`: 393D -> world joint positions (for visualization)

## Normalization

Like body 272D, the format_hand representation should be normalized using per-channel
Mean.npy and Std.npy computed from the training dataset. A feat_bias of 1.0 is added
to Std to prevent division by near-zero values on low-variance channels.
