#!/usr/bin/env bash
#
# reproduce.sh — one-shot driver for the noisy/sparse data-assimilation
# experiment (PDE collocation + noisy scattered u-observations + monotonicity).
#
# Runs, in order (all in the STARVED-corner headline regime, the default —
# observations scattered with a data-free (1,1) corner, the partial-BC showcase
# under noise; pass --uniform to sweep.py for the full-domain contrast):
#   1. R-free quick sweep (regime characterisation: rmse_unc, MC P(C|E))
#   2. single-snapshot demo  run.py        (run.png + comparison.txt)
#   3. full R-backed sweep   sweep.py       (gain CSV, the headline numbers)
#   4. publication figure    figure.py      (svg/pdf/tiff gain curves)
#
# Run from the repo root inside the container (the repo is mounted at /app):
#
#     cd /app && bash experiments/linpde_gp_constrained/poisson_2d_data_assimilation/reproduce.sh
#
# Requires: linpde-gp + probnum + jax + scipy for steps 1-2 (equality side and
# the R-free MC P(C|E)); additionally rpy2 + R packages
# mvtnorm/tmvtnorm/TruncatedNormal/truncnorm for the constrained mean in
# steps 2-3 and the gain curves in step 4.

set -euo pipefail

# Resolve repo root: this script lives in experiments/.../poisson_2d_data_assimilation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

export PYTHONPATH="src:gp_constr:src/linpde_gp_constrained:${PYTHONPATH:-}"

EXP="experiments/linpde_gp_constrained/poisson_2d_data_assimilation"

echo "============================================================"
echo "[1/4] R-free quick sweep — regime characterisation"
echo "============================================================"
python "$EXP/sweep.py" --quick

echo
echo "============================================================"
echo "[2/4] Single-snapshot demo — run.py (run.png + comparison.txt)"
echo "============================================================"
python "$EXP/run.py"

echo
echo "============================================================"
echo "[3/4] Full R-backed sweep — sweep.py (gain CSV; the slow step)"
echo "============================================================"
python "$EXP/sweep.py"

echo
echo "============================================================"
echo "[4/4] Headline figure — figure.py (svg/pdf/tiff)"
echo "============================================================"
python "$EXP/figure.py"

echo
echo "============================================================"
echo "Done. Snapshot diagnostics:"
echo "============================================================"
cat "$EXP/comparison.txt" 2>/dev/null || echo "(comparison.txt not found)"
echo
echo "Sweep CSV: $EXP/poisson_2d_data_assimilation_sweep.csv"
