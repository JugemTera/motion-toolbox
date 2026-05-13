# SMPL Sample Data

This directory contains sample SMPL motion data for testing the motion toolbox.

## Files

- `gvhmr_pred_4d004_z.npz` - GVHMR prediction (upright, Y-up coordinate system) ✅
- `metrabs_1.npz` - MetRABS prediction
- `gvhmr_gt_4d004.pt` - GVHMR ground truth

## Notes on SMPL Data Orientation

The 272-dimensional motion format is **canonicalized** and represents motion in a standard upright orientation (Y-up, character standing on ground plane at Y≈0).

When converting SMPL → 272D → Motion:
- **Upright SMPL data** (positive Y translations, head above pelvis) → stays upright
- **Upside-down SMPL data** (negative Y translations, head below pelvis) → gets flipped to upright (by design)

This is correct behavior - the 272D format does not preserve absolute vertical position or orientation, only relative motion patterns.

## Testing

For round-trip testing (SMPL → 272D → Motion → visualization), use `gvhmr_pred_4d004_z.npz` which is known to be in correct upright orientation.
