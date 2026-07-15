#!/usr/bin/env bash
# =====================================================================
# Reproduce the brain MRE results (Sec. IV.E): Table 1, Table 2, the
# reconstruction figures, and the supporting loss-tangent number.
#
# The committed sweep_log.txt was produced inside the project's Docker
# container (note the /app paths and KeOps messages). Reproduce in the
# same environment:
#
#     ./docker-run.sh build      # once
#     ./docker-run.sh run        # start container (detached)
#     ./docker-run.sh shell      # drop into /app, then run this script:
#     bash experiments/reproduce_brain.sh
#
# The script cd's to its own repo root, so it runs from anywhere. Solver defaults match
# the paper: n_shell = n_interior = 2000, Matern nu = 2.5, pde_noise_rel
# 1e-4, cov_jitter_rel 1e-7, seed 0, real2 (full complex) solve path.
# Runtime: Table 1 sweep ~25-30 min (27 cases); each Table 2 run ~1 min.
# =====================================================================
set -euo pipefail

# Run from the repo root regardless of where this script lives or is invoked from
# (every path below is repo-root-relative). Works with the script at the repo root
# OR in experiments/.
_self="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$_self/experiments" ]; then cd "$_self"; else cd "$_self/.."; fi

DRIVER=experiments/helmholtz_brain_forward_bvp.py
OUT=experiments/helmholtz_brain_outputs

# ---------------------------------------------------------------------
# Table 1 -- generalization sweep (3 subjects x 3 freqs x 3 components),
# multiscale LMC, base length scales 4/16 mm, rho = 0, curl observable.
# Writes the per-case *_curl_main.* (incl. the anchor = Fig. brain_recon:
# helmholtz_brain_U01_UDEL_0001_01_70Hz_compx_curl_main.pdf) and
# *_curl_mismatch.* figures, and the log the 7 OFAT rows are read from.
# ---------------------------------------------------------------------
# One PROCESS PER CASE (not a single in-process --sweep): the 27-case sweep
# accumulates RAM (JAX/KeOps caches + the ~8.3k-voxel dense std evals) and the
# OOM killer stops it ("Killed", SIGKILL, no traceback) around the 4th case on a
# typical machine. Separate processes release memory between cases, keeping peak
# RAM at a single case's footprint. Results are identical (the driver seeds per
# process), and sweep_log.txt is the same concatenation the 7 OFAT rows read from.
: > "$OUT/sweep_log.txt"
for SUBJ in U01_UDEL_0001_01 U01_UDEL_0002_01 U01_UDEL_0003_01; do
  for FREQ in 30 50 70; do
    for COMP in 0 1 2; do
      python "$DRIVER" --subject "$SUBJ" --freq "$FREQ" --component "$COMP" \
        --observable curl --prior lmc --lmc-lengthscales 4,16 \
        --output-scale 2e-4 --bc-noise-rel 1e-6 --sparse-figures --no-suptitle \
        2>&1 | tee -a "$OUT/sweep_log.txt"
    done
  done
done

# ---------------------------------------------------------------------
# Table 2 -- prior ablation at the anchor (subject 0001, 70 Hz, comp x=0).
# Each run prints:
#   Pearson |q_pred| vs |q_meas|            -> "Pearson" column
#   median |err|/|q| (all)                  -> "rel. err" column
#   Pearson(std, |r|/|k^2 q|)               -> "corr(sigma,r)" column
# ---------------------------------------------------------------------
ANCHOR="--subject U01_UDEL_0001_01 --freq 70 --component 0 \
  --observable curl --output-scale 2e-4 --bc-noise-rel 1e-6 --sparse-figures"

# IID, single scale 15 mm  -> 0.686 / 0.448 / 0.474
python "$DRIVER" $ANCHOR --prior iid --lengthscale-mm 15 \
  2>&1 | tee "$OUT/ablation_iid_15.log"

# LMC 3/16 mm, rho = 0      -> 0.747 / 0.389 / 0.364
python "$DRIVER" $ANCHOR --prior lmc --lmc-lengthscales 3,16 \
  2>&1 | tee "$OUT/ablation_lmc_3_16_rho0.log"

# LMC 4/16 mm, rho = 0      -> 0.767 / 0.375 / 0.347  (champion / anchor)
python "$DRIVER" $ANCHOR --prior lmc --lmc-lengthscales 4,16 \
  2>&1 | tee "$OUT/ablation_lmc_4_16_rho0.log"

# LMC 4/16 mm, rho = 0.6    -> 0.761 / 0.379 / 0.352
python "$DRIVER" $ANCHOR --prior lmc --lmc-lengthscales 4,16 --coreg-corr 0.6 \
  2>&1 | tee "$OUT/ablation_lmc_4_16_rho0.6.log"

# ---------------------------------------------------------------------
# Supporting -- loss tangent G''/G' over the brain mask (for the setup
# text / Methods; resolves "~0.2-0.4" vs the data's ~0.5). Not a value in
# the IV.E results text. Uses nibabel (available inside the container).
# Expected medians: 30Hz ~0.36, 50Hz ~0.50, 70Hz ~0.55-0.62; anchor 0.555.
# ---------------------------------------------------------------------
python - <<'PY'
import numpy as np, nibabel as nib
from pathlib import Path
root=Path("data/brain_experiment_data/mre_udel")
for s in ["U01_UDEL_0001_01","U01_UDEL_0002_01","U01_UDEL_0003_01"]:
    mask=np.asanyarray(nib.load(str(root/f"{s}_v4"/f"{s}_register_to_MRE"
        /f"{s}_MREreg_brainmask.nii")).dataobj).astype(bool)
    for f in (30,50,70):
        fd=root/f"{s}_v4"/f"{s}_MRE_AP_{f}Hz"; st=f"{s}_MRE_AP_{f}Hz"
        Gr=np.asanyarray(nib.load(str(fd/f"{st}_props_shear_real.nii")).dataobj).astype(float)
        Gi=np.asanyarray(nib.load(str(fd/f"{st}_props_shear_imag.nii")).dataobj).astype(float)
        m=mask&(Gr>0)
        print(f"{s} {f:>2}Hz  median loss tangent = {np.median(Gi[m]/Gr[m]):.3f}")
PY

# ---------------------------------------------------------------------
# Figures. The --sweep above already wrote, for ALL 27 cases (3 subjects
# x 3 freqs x 3 components), both figure types in PNG + SVG + PDF:
#   helmholtz_brain_<subj>_<f>Hz_comp<c>_curl_main.{png,svg,pdf}      (3x3 grid)
#   helmholtz_brain_<subj>_<f>Hz_comp<c>_curl_mismatch.png            (diagnostic)
# To paper-style ALL of them (no on-figure title), append --no-suptitle to
# the sweep command. To recompute only the figures, just rerun the sweep
# (there is no figure-only mode; figures come from the same solve).
#
# Headline / anchor figure (Fig. brain_recon) at EXACT posterior std (drop
# --sparse-figures, ~70 s) and paper-styled (--no-suptitle):
python "$DRIVER" \
  --subject U01_UDEL_0001_01 --freq 70 --component 0 \
  --observable curl --prior lmc --lmc-lengthscales 4,16 \
  --output-scale 2e-4 --bc-noise-rel 1e-6 --no-suptitle

echo
echo "Done. Table 1 rows: read from $OUT/sweep_log.txt."
echo "Table 2 rows: read from $OUT/ablation_*.log."
echo "Figures (all subjects/freqs/components): $OUT/helmholtz_brain_*_curl_{main,mismatch}.*"
# ---------------------------------------------------------------------
# Other paper sections (not part of IV.E) are notebooks under experiments/:
#   1D real:    inHomo_realHelmholtz_dirichlet_1d_{3,5,7}pts.ipynb
#   1D complex: inHomo_complexHelmholtz_dirichlet_1d_{3,5,7}pts.ipynb
#   2D:         inHomo_realHelmholtz_dirichlet_2d_{64,100,144}pts.ipynb
#   3D cube:    inHomo_realHelmholtz_dirichlet_3d_{729,1000,1331}pts.ipynb
#   baselines:  fdm_*.ipynb, pinn_*.ipynb
# Run them with:  ./docker-run.sh jupyter
# NOTE: the SNR scatter (make_snr_pearson_figure.py) reads HARD-CODED SNR
# values; there is no upstream SNR generator, which is why SNR was dropped
# from the IV.E text. Do not treat that figure as reproduced.
# ---------------------------------------------------------------------
