# SMPLify and HumanML3D → SMPL mesh

Converting HumanML3D **263D** features directly to an SMPL **mesh** requires SMPLify-style optimization. The unified visualizer does **not** run SMPLify automatically (it is slow and needs GPU + SMPL model files).

**Practical options**

1. **Skeleton mode** — fast, no SMPLify: `Visualizer(...).visualize(data, "out.html", mode="skeleton")`.
2. **Convert to 272D** (or use joint positions) then mesh — see [UNIFIED_VISUALIZER_GUIDE.md](UNIFIED_VISUALIZER_GUIDE.md).
3. **Run the standalone scripts** in `scripts/` (e.g. `humanml3d_to_smpl_visualization.py`) with SMPL models installed under `~/.motion_toolbox/datasets/`.

For joints2smpl prior files (`gmm_08.pkl`, `neutral_smpl_mean_params.h5`), see `src/motion_toolbox/converters/joints2smpl/smpl_models/README.md`.
