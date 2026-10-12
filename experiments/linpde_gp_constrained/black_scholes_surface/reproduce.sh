#!/usr/bin/env bash
#
# reproduce.sh — one-shot driver for the arbitrage-constrained call-surface
# experiment (quotes + Dupire PDE collocation + no-arbitrage inequalities).
#
# Runs, in order:
#   1. truth       check_truth.py           (BS truth: parity, quadrature, Dupire residual)
#   2. dispatch    check_dispatch.py        (closed-form ExpQuad dispatch for the operator)
#   3. recovery    run.py --recovery-check  (dense noiseless quotes -> RMSE < 1e-3)
#   4. regime      sweep.py --quick         (R-free regime characterisation)
#   5. snapshot    run.py                   (headline snapshot; asserts the checks)
#   6. full sweep  sweep.py                 (gain CSV + bs_numbers.tex; the slow R step)
#   7. figure      figure.py                (wide + _column svg/pdf/tiff)
#
# Run from the repo root inside the container (the repo is mounted at /app):
#
#     cd /app && bash experiments/linpde_gp_constrained/black_scholes_surface/reproduce.sh
#
# Steps 1-4 are R-free (linpde-gp + probnum + jax + scipy); steps 5-7
# additionally need rpy2 + R packages mvtnorm/tmvtnorm/TruncatedNormal/
# truncnorm for the constrained arms (without them step 5 degrades to the
# equality-side checks and steps 6-7 abort).
#
# Real data (optional, outside the container first):
#     python experiments/.../black_scholes_surface/fetch_data.py
#     python experiments/.../black_scholes_surface/run.py --data data/spx_chain_<date>.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

export PYTHONPATH="src:gp_constr:src/linpde_gp_constrained:${PYTHONPATH:-}"

EXP="experiments/linpde_gp_constrained/black_scholes_surface"

echo "============================================================"
echo "[1/7] Truth check — Black-Scholes truth + Dupire residual"
echo "============================================================"
python "$EXP/check_truth.py"

echo
echo "============================================================"
echo "[2/7] Operator dispatch check"
echo "============================================================"
python "$EXP/check_dispatch.py"

echo
echo "============================================================"
echo "[3/7] Recovery check — equality sanity (dense noiseless quotes)"
echo "============================================================"
python "$EXP/run.py" --recovery-check

echo
echo "============================================================"
echo "[4/7] Regime scout — R-free quick sweep"
echo "============================================================"
python "$EXP/sweep.py" --quick

echo
echo "============================================================"
echo "[5/7] Headline snapshot — run.py (regime and constrained-arm checks)"
echo "============================================================"
python "$EXP/run.py"

echo
echo "============================================================"
echo "[6/7] Full R-backed sweep — sweep.py (gain CSV + bs_numbers.tex)"
echo "============================================================"
python "$EXP/sweep.py"

echo
echo "============================================================"
echo "[7/7] Publication figure — figure.py (wide + column)"
echo "============================================================"
python "$EXP/figure.py"

echo
echo "============================================================"
echo "Done. Snapshot diagnostics:"
echo "============================================================"
cat "$EXP/comparison.txt" 2>/dev/null || echo "(comparison.txt not found)"
echo
echo "Sweep CSV:   $EXP/black_scholes_sweep.csv"
echo "TeX macros:  $EXP/bs_numbers.tex"
