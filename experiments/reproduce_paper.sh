#!/usr/bin/env bash
# =====================================================================
# Reproduce ALL experiments / tables / figures in the paper.
#
# Run inside the project's Docker container (the environment the results
# were generated in):
#     ./docker-run.sh build      # once
#     ./docker-run.sh run
#     ./docker-run.sh shell      # drop into /app, then:
#     bash experiments/reproduce_paper.sh
#
# Sections IV.A-IV.D are Jupyter notebooks; they are executed in place
# (outputs + the .pdf figures they save under experiments/ are refreshed).
# Section IV.E (brain) is delegated to experiments/reproduce_brain.sh, which
# runs one process per case (the in-process sweep is OOM-prone). The brain
# part needs the external MRE data in place -- see docs/HELMHOLTZ.md.
# =====================================================================
set -euo pipefail

# Run from the repo root regardless of where this script lives or is invoked
# from (every path below is repo-root-relative). Works whether the script sits
# at the repo root OR in experiments/.
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_self/experiments" ]; then cd "$_self"; else cd "$_self/.."; fi

EXP=experiments
BASE=$EXP/helmholtz_equation_baselines          # FDM + PINN baseline notebooks
NB="jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1"

# ---------------------------------------------------------------------
# Sec. IV.A -- 1D real-valued Helmholtz (Table tab:tab1d_real + figures
# inHomo_realHelmholtz_dirichlet_1d_3pts.pdf, solution/rhs_belief_*pts.pdf)
# ---------------------------------------------------------------------
$NB $EXP/inHomo_realHelmholtz_dirichlet_1d_3pts.ipynb
$NB $EXP/inHomo_realHelmholtz_dirichlet_1d_5pts.ipynb
$NB $EXP/inHomo_realHelmholtz_dirichlet_1d_7pts.ipynb
$NB $BASE/fdm_inHomo_realHelmholtz_dirichlet_1d.ipynb      # FDM baseline
$NB $BASE/pinn_inHomo_realHelmholtz_dirichlet_1d.ipynb     # PINN baseline

# ---------------------------------------------------------------------
# Sec. IV.B -- 1D complex-valued Helmholtz (Table tab:tab1d_complex +
# figure inHomo_complexHelmholtz_dirichlet_1d_3pts.pdf)
# ---------------------------------------------------------------------
$NB $EXP/inHomo_complexHelmholtz_dirichlet_1d_3pts.ipynb
$NB $EXP/inHomo_complexHelmholtz_dirichlet_1d_5pts.ipynb
$NB $EXP/inHomo_complexHelmholtz_dirichlet_1d_7pts.ipynb
$NB $BASE/fdm_inHomo_complexHelmholtz_dirichlet_1d.ipynb
$NB $BASE/pinn_inHomo_complexHelmholtz_dirichlet_1d.ipynb

# ---------------------------------------------------------------------
# Sec. IV.C -- 2D Helmholtz (Table tab:tab2d + helmholtz2d_posterior_*.pdf,
# helmholtz2d_pde_residual_*.pdf)
# ---------------------------------------------------------------------
$NB $EXP/inHomo_realHelmholtz_dirichlet_2d_64pts.ipynb
$NB $EXP/inHomo_realHelmholtz_dirichlet_2d_100pts.ipynb
$NB $EXP/inHomo_realHelmholtz_dirichlet_2d_144pts.ipynb
$NB $BASE/fdm_inHomo_realHelmholtz_dirichlet_2d.ipynb
$NB $BASE/pinn_inHomo_realHelmholtz_dirichlet_2d.ipynb

# ---------------------------------------------------------------------
# Sec. IV.D -- 3D cube Helmholtz (Table tab:tab3d + helmholtz3d_posterior_*.pdf)
# ---------------------------------------------------------------------
$NB $EXP/inHomo_realHelmholtz_dirichlet_3d_729pts.ipynb
$NB $EXP/inHomo_realHelmholtz_dirichlet_3d_1000pts.ipynb
$NB $EXP/inHomo_realHelmholtz_dirichlet_3d_1331pts.ipynb
$NB $BASE/fdm_inHomo_realHelmholtz_dirichlet_3d.ipynb
$NB $BASE/pinn_inHomo_realHelmholtz_dirichlet_3d.ipynb

# ---------------------------------------------------------------------
# Sec. IV.E -- brain MRE (Table tab:brain_generalization,
# tab:brain_prior_ablation + all reconstruction/mismatch figures).
# Delegated to reproduce_brain.sh: one process per case (robust to OOM),
# plus the Table 2 ablation, the loss-tangent number, and the headline
# figure. Requires the external MRE data (see docs/HELMHOLTZ.md).
# ---------------------------------------------------------------------
bash $EXP/reproduce_brain.sh

echo "Done. Notebook figures: $EXP/*.pdf ; brain outputs: $EXP/helmholtz_brain_outputs/"
