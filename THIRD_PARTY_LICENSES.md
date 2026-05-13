Third-party components and licenses
=====================================

This software bundles or integrates the following third-party materials. **Verify each license before redistribution** (especially SMPL, MANO, HumanML3D, and any game-engine assets).

| Component | Location / use | Upstream / license notes |
|-----------|----------------|-------------------------|
| **SMPL / SMPL-X** | `smplx` dependency; body model files user-supplied | [SMPL](https://smpl.is.tue.mpg.de/) — academic / non-commercial terms from MPI |
| **SMPLify / joints2smpl** | `src/motion_toolbox/converters/joints2smpl/` | Derived from SMPLify / joints2smpl lineages; see upstream licenses |
| **Pose prior files** | `joints2smpl/smpl_models/` (user must add `gmm_08.pkl`, `neutral_smpl_mean_params.h5`) | Not redistributed here; obtain per upstream license |
| **HumanML3D** | Sample `.npy` under `sample_data/humanml3d/` | [HumanML3D](https://github.com/EricGuo5513/HumanML3D) — follow dataset license |
| **272-dim representation** | Sample / docs reference | [272-dim-Motion-Representation](https://github.com/Li-xingXiao/272-dim-Motion-Representation) |
| **MHR (Momentum Human Rig)** | Optional extra `mhr`; `scripts/setup_mhr.sh` | [facebookresearch/MHR](https://github.com/facebookresearch/MHR) |
| **STMC** | ALIGN / STMC converters | See StableMotion / STMC upstream licenses referenced in docs |
| **Plotly / Three.js** | HTML viewers | Plotly and Three.js per their respective licenses |
| **Text-to-motion evaluators** | `scripts/download_t2m_evaluators.sh` | Follow original T2M / MDM / HumanML3D evaluator licenses |
| **R15 mesh (optional)** | `assets/r15/` | **Redistribution may require permission** from the asset owner; confirm before open release |

If you add new vendored code or data, extend this table.
