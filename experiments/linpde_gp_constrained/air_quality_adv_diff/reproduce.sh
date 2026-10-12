#!/usr/bin/env bash
#
# reproduce.sh -- one-shot driver for the positivity-constrained steady
# advection-diffusion air-quality experiment (PDE collocation + sparse monitors
# + nonnegativity of concentration).
#
# Runs, in order:
#   1. Operator-dispatch check           check_dispatch.py
#   2. Truth + FD convergence check      plume.py
#   3. Offline London sample             fetch_data.py --sample
#   4. Dense-recovery check              run.py --recovery-check
#   5. Synthetic headline snapshot       run.py        (fields.npz, diagnostics.json)
#   6. --data run on the sample          run.py --data (suffixed *_openaq outputs)
#   7. Sweep + LaTeX macros              sweep.py      (sweep CSV, aq_numbers.tex)
#   8. Publication figure                figure.py     (svg/pdf/tiff, wide + _column)
#
# Run from the repo root inside the container (the repo is mounted at /app):
#
#     cd /app && bash experiments/linpde_gp_constrained/air_quality_adv_diff/reproduce.sh
#
# Requires: linpde-gp + probnum + jax + scipy throughout; additionally
# rpy2 + R packages mvtnorm/tmvtnorm/TruncatedNormal/truncnorm for the EXACT
# constrained arm and exact P(C|E). Without R the constrained arm falls back to
# the experiment-local truncated-normal sampler (run.py) and every check except
# the exact P(C|E) still passes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

export PYTHONPATH="src:gp_constr:src/linpde_gp_constrained:${PYTHONPATH:-}"

EXP="experiments/linpde_gp_constrained/air_quality_adv_diff"
DATA="$EXP/data/openaq_london_no2_2024-07-15.csv"

echo "[1/8] Operator dispatch check"
python "$EXP/check_dispatch.py"

echo; echo "[2/8] Truth check -- FD truth + manufactured-solution convergence"
python "$EXP/plume.py"

echo; echo "[3/8] Offline London sample (synthetic values) -- fetch_data.py --sample"
python "$EXP/fetch_data.py" --sample

echo; echo "[4/8] Recovery check -- dense low-noise recovery (RMSE < 1e-3)"
python "$EXP/run.py" --recovery-check

echo; echo "[5/8] Synthetic headline snapshot -- run.py"
python "$EXP/run.py"

echo; echo "[6/8] --data run on the offline sample -- run.py --data"
python "$EXP/run.py" --data "$DATA"

echo; echo "[7/8] Sweep + LaTeX macros -- sweep.py (slow; constrained arm)"
python "$EXP/sweep.py"

echo; echo "[8/8] Headline figure -- figure.py"
python "$EXP/figure.py"

echo; echo "============================================================"
echo "Done. Snapshot diagnostics:"
echo "============================================================"
cat "$EXP/comparison.txt" 2>/dev/null || echo "(comparison.txt not found)"
echo
echo "Sweep CSV:   $EXP/air_quality_adv_diff_sweep.csv"
echo "LaTeX macros: $EXP/aq_numbers.tex"
