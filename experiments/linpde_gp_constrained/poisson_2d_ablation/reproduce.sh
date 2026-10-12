#!/usr/bin/env bash
#
# reproduce.sh — one-shot driver for the 3-way ablation (equality-only vs
# inequality-only vs both) on the free-boundary 2D Poisson problem.
#
# Runs:
#   1. run.py    — computes all three arms, writes comparison.txt,
#                  diagnostics.json, fields.npz, and ablation.png
#   2. figure.py — publication figure (svg/pdf/tiff) from fields.npz
#
# Run from the repo root inside the container (the repo is mounted at /app):
#
#     cd /app && bash experiments/linpde_gp_constrained/poisson_2d_ablation/reproduce.sh
#
# Requires: linpde-gp + probnum + jax + scipy for the equality-only arm and the
# R-free prior P(C); additionally rpy2 + R packages
# mvtnorm/tmvtnorm/TruncatedNormal/truncnorm for the inequality-only and both
# arms (the truncated-MVN moments). Without R the script degrades gracefully:
# the equality-only diagnostics and the R-free prior P(C) are still produced.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

export PYTHONPATH="src:gp_constr:src/linpde_gp_constrained:${PYTHONPATH:-}"

EXP="experiments/linpde_gp_constrained/poisson_2d_ablation"

echo "============================================================"
echo "[1/2] 3-way ablation — run.py"
echo "============================================================"
python "$EXP/run.py"

echo
echo "============================================================"
echo "[2/2] Publication figure — figure.py"
echo "============================================================"
python "$EXP/figure.py"

echo
echo "============================================================"
echo "Done. 3-way comparison:"
echo "============================================================"
cat "$EXP/comparison.txt" 2>/dev/null || echo "(comparison.txt not found)"
