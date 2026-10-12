#!/usr/bin/env bash
#
# reproduce_spx.sh — real-data driver (the publishable headline).
#
# The dense SPX chain is the NEGATIVE CONTROL (data pin the surface, the
# inequalities are inert); the THINNED chain is the headline (model-free
# no-arbitrage information recovers its value as the market goes illiquid).
#
# Prerequisite (run OUTSIDE the container, needs internet + yfinance):
#     pip install yfinance
#     python experiments/.../black_scholes_surface/fetch_data.py
#   -> writes data/spx_chain_<date>.csv  (pass that filename as $1 below)
#
# Run from the repo root INSIDE the container (R backend required):
#     cd /app && bash experiments/linpde_gp_constrained/black_scholes_surface/reproduce_spx.sh data/spx_chain_<date>.csv [NSEEDS]
#
#   $2 NSEEDS = thinned subsets per n_obs (default 20; 15-25 recommended). The
#   full sweep is ~len(NOBS_VALUES)*NSEEDS*535 s of wall-clock: ~13 h at 15,
#   ~18 h at 20, ~22 h at 25.
#
# Steps:
#   1. dense control snapshot   run.py --data <csv> --dense
#   2. thinned headline snapshot run.py --data <csv>          (diagnostics only)
#   3. R-free thinned regime scout  sweep_spx.py --data <csv> --quick --seeds N
#   4. full R-backed thinned sweep  sweep_spx.py --data <csv> --seeds N (slow)
#   5. real-data figure         figure_spx.py

set -euo pipefail

CSV="${1:?usage: reproduce_spx.sh data/spx_chain_<date>.csv [NSEEDS] (see data/README.md)}"
NSEEDS="${2:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"
export PYTHONPATH="src:gp_constr:src/linpde_gp_constrained:${PYTHONPATH:-}"
EXP="experiments/linpde_gp_constrained/black_scholes_surface"

echo "============================================================"
echo "[1/5] Dense negative control — run.py --dense"
echo "============================================================"
python "$EXP/run.py" --data "$CSV" --dense

echo
echo "============================================================"
echo "[2/5] Thinned headline snapshot — run.py (illiquid regime)"
echo "============================================================"
python "$EXP/run.py" --data "$CSV"

echo
echo "============================================================"
echo "[3/5] R-free thinned regime scout — sweep_spx.py --quick"
echo "============================================================"
python "$EXP/sweep_spx.py" --data "$CSV" --quick --seeds "$NSEEDS"

echo
echo "============================================================"
echo "[4/5] Full R-backed thinned sweep — sweep_spx.py (slow)"
echo "============================================================"
python "$EXP/sweep_spx.py" --data "$CSV" --seeds "$NSEEDS"

echo
echo "============================================================"
echo "[5/5] Real-data figure — figure_spx.py"
echo "============================================================"
python "$EXP/figure_spx.py"

echo
echo "Done. SPX sweep CSV: $EXP/black_scholes_spx_sweep.csv"
echo "TeX macros:          $EXP/bs_spx_numbers.tex"
