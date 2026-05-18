# SMPLify / joints2smpl prior files (not shipped in git)

The HumanML3D → SMPL pipeline expects these files in this directory:

- `gmm_08.pkl` — Gaussian mixture pose prior (from SMPLify / joints2smpl).
- `neutral_smpl_mean_params.h5` — neutral SMPL mean shape parameters.

Download them from the [SMPLify](https://github.com/classner/SMPLify) or [joints2smpl](https://github.com/wangsen1313/joints2smpl) distribution you are licensed to use, and place them here. See [THIRD_PARTY_LICENSES.md](../../../../../THIRD_PARTY_LICENSES.md) for attribution.

Optional: `smplx_parts_segm.pkl` if required by your SMPL-X setup (see `src/config.py` in joints2smpl).
