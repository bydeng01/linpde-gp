r"""Sweep the positivity constraint's value vs data quality (n_obs, sigma_obs).

Headline claim: the positivity inequality's marginal value --- the held-out RMSE
reduction and the calibration (sub-zero tail) reduction it buys on top of the
PDE + monitors --- GROWS as the data degrade (sparser / noisier), because the
equality posterior then spends more probability mass below zero for positivity
to reclaim. As the data alone determine c >= 0, the gain shrinks toward zero.

Design
------
A *cross* through the (sigma_obs, n_obs) plane (not the full Cartesian grid):

  * vs sigma_obs : n_obs fixed at FIXED_NOBS, sigma_obs over SIGMA_VALUES;
  * vs n_obs     : sigma_obs fixed at FIXED_SIGMA, n_obs over NOBS_VALUES.

Each operating point repeats over SEEDS noise realisations (fixed Halton monitor
locations, so the seed varies only the noise). Gains are reported as
RATIO-OF-MEANS, 1 - mean(rmse_con)/mean(rmse_unc), a stable aggregate estimator
(the per-seed ratio has a noisy denominator). ``--grid full`` runs the full
Cartesian product for the optional gain heatmap.

Modes
-----
  * default     : full pipeline; constrained arm via the gp_constr seam (R) when
                  available, else the R-free truncated-normal fallback (run.py).
  * ``--quick`` : equality side + MC P(C|E) only (no constrained solve); use it
                  to scout the regime before the full sweep.

Output: ``air_quality_adv_diff_sweep.csv`` + LaTeX macros ``aq_numbers.tex``.

Run from the repo root inside the container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \
      python experiments/linpde_gp_constrained/air_quality_adv_diff/sweep.py
    PYTHONPATH=... python .../sweep.py --quick     # fast, no constrained solve
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plume  # noqa: E402
from run import ARMS, solve  # noqa: E402

# --- Sweep ranges. Keep in sync with figure.py / the .tex caption. ---
SIGMA_VALUES = [0.05, 0.10, 0.15, 0.22, 0.30]  # observation-noise std
NOBS_VALUES = [8, 11, 14, 20, 28]  # number of monitors
SEEDS = [0, 1, 2, 3, 4]

FIXED_NOBS = 14  # held fixed while sweeping sigma_obs (matches run.py headline)
FIXED_SIGMA = 0.10  # held fixed while sweeping n_obs

_FIELDS = [
    "axis",
    "n_obs",
    "sigma_obs",
    "obs_seed",
    "backend",
    "p_ce_mc",
    "p_c_mon_mc",
    "rmse_held_E",
    "rmse_held_EC",
    "gain_held_rel",
    "rmse_grid_E",
    "rmse_grid_EC",
    "gain_grid_rel",
    "band_E",
    "band_EC",
    "subzero_E",
    "subzero_EC",
    "negfrac_E",
    "minval_E",
    "constrained_ok",
]


def _row(axis, diag) -> dict:
    a = diag["arms"]
    E, EC = a["monitors+E"], a["monitors+E+C"]
    ok = int(np.isfinite(EC["rmse_held"]))
    return {
        "axis": axis,
        "n_obs": diag["n_obs"],
        "sigma_obs": diag["sigma_obs"],
        "obs_seed": diag["obs_seed"],
        "backend": diag["backend"],
        "p_ce_mc": diag["p_ce_mc"],
        "p_c_mon_mc": diag["p_c_mon_mc"],
        "rmse_held_E": E["rmse_held"],
        "rmse_held_EC": EC["rmse_held"],
        "gain_held_rel": diag["gain_rmse_held_rel"],
        "rmse_grid_E": E["rmse_grid"],
        "rmse_grid_EC": EC["rmse_grid"],
        "gain_grid_rel": diag["gain_rmse_grid_rel"],
        "band_E": E["band"],
        "band_EC": EC["band"],
        "subzero_E": E["subzero_tail"],
        "subzero_EC": EC["subzero_tail"],
        "negfrac_E": E["neg_frac"],
        "minval_E": E["min_val"],
        "constrained_ok": ok,
    }


def _nm(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _ns(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.std()) if arr.size else float("nan")


def _rom(group, key_unc, key_con):
    """Ratio-of-means relative gain 1 - mean(con)/mean(unc), in percent."""
    ru = _nm([r[key_unc] for r in group])
    rc = _nm([r[key_con] for r in group])
    return 100.0 * (1.0 - rc / ru) if ru > 0 else float("nan")


def _fmt_pce(p: float) -> str:
    if p != p:
        return "\\mathrm{n/a}"
    if p <= 0.0:
        return "0"
    if p < 0.01:
        import math

        e = math.floor(math.log10(p))
        m = p / 10**e
        return f"{m:.0f}\\times10^{{{e}}}"
    return f"{p:.2f}"


def write_numbers_tex(rows, out_dir) -> None:
    """Emit aq_numbers.tex \\newcommand macros, seed-averaged from the sweep."""
    if not any(r["constrained_ok"] for r in rows):
        print("NOTE: no constrained results — aq_numbers.tex not refreshed (--quick).")
        return

    def at(n=None, s=None):
        out = rows
        if n is not None:
            out = [r for r in out if r["n_obs"] == n]
        if s is not None:
            out = [r for r in out if abs(r["sigma_obs"] - s) < 1e-12]
        return out

    sig_pts = sorted(set(r["sigma_obs"] for r in at(n=FIXED_NOBS)))
    sig_lo, sig_hi = sig_pts[0], sig_pts[-1]
    g_sig_lo = _rom(at(FIXED_NOBS, sig_lo), "rmse_held_E", "rmse_held_EC")
    g_sig_hi = _rom(at(FIXED_NOBS, sig_hi), "rmse_held_E", "rmse_held_EC")
    # Calibration (sub-zero tail) gains -- the robust headline curve.
    t_sig_lo = _rom(at(FIXED_NOBS, sig_lo), "subzero_E", "subzero_EC")
    t_sig_hi = _rom(at(FIXED_NOBS, sig_hi), "subzero_E", "subzero_EC")

    n_pts = sorted(set(r["n_obs"] for r in at(s=FIXED_SIGMA)))
    n_lo, n_hi = n_pts[0], n_pts[-1]
    g_n_lo = _rom(at(n_lo, FIXED_SIGMA), "rmse_held_E", "rmse_held_EC")
    g_n_hi = _rom(at(n_hi, FIXED_SIGMA), "rmse_held_E", "rmse_held_EC")
    t_n_lo = _rom(at(n_lo, FIXED_SIGMA), "subzero_E", "subzero_EC")
    t_n_hi = _rom(at(n_hi, FIXED_SIGMA), "subzero_E", "subzero_EC")

    tc = at(FIXED_NOBS, FIXED_SIGMA)
    rmse_E = _nm([r["rmse_held_E"] for r in tc])
    rmse_EC = _nm([r["rmse_held_EC"] for r in tc])
    gain = 100.0 * (1.0 - round(rmse_EC, 3) / round(rmse_E, 3))
    band_E = _nm([r["band_E"] for r in tc])
    band_EC = _nm([r["band_EC"] for r in tc])
    sz_E = _nm([r["subzero_E"] for r in tc])
    sz_EC = _nm([r["subzero_EC"] for r in tc])
    tail_red = 100.0 * (1.0 - sz_EC / sz_E) if sz_E > 0 else float("nan")
    negfrac = _nm([r["negfrac_E"] for r in tc])
    minval = _nm([r["minval_E"] for r in tc])
    p_ce = _nm([r["p_ce_mc"] for r in tc])

    macros = {
        "AqSeeds": f"{len(SEEDS)}",
        "AqNobsFixed": f"{FIXED_NOBS}",
        "AqSigFixed": f"{FIXED_SIGMA:g}",
        "AqSigLo": f"{sig_lo:g}",
        "AqGainSigLo": f"{g_sig_lo:.0f}",
        "AqSigHi": f"{sig_hi:g}",
        "AqGainSigHi": f"{g_sig_hi:.0f}",
        "AqTailRedSigLo": f"{t_sig_lo:.0f}",
        "AqTailRedSigHi": f"{t_sig_hi:.0f}",
        "AqNLo": f"{n_lo}",
        "AqGainNLo": f"{g_n_lo:.0f}",
        "AqNHi": f"{n_hi}",
        "AqGainNHi": f"{g_n_hi:.0f}",
        "AqTailRedNLo": f"{t_n_lo:.0f}",
        "AqTailRedNHi": f"{t_n_hi:.0f}",
        "AqRmseE": f"{rmse_E:.3f}",
        "AqRmseEC": f"{rmse_EC:.3f}",
        "AqGain": f"{gain:.0f}",
        "AqBandE": f"{band_E:.2f}",
        "AqBandEC": f"{band_EC:.2f}",
        "AqSubzeroE": f"{sz_E:.3f}",
        "AqSubzeroEC": f"{sz_EC:.3f}",
        "AqTailRed": f"{tail_red:.0f}",
        "AqNegFrac": f"{100*negfrac:.0f}",
        "AqMinVal": f"{minval:.2f}",
        "AqPCE": _fmt_pce(p_ce),
    }
    path = os.path.join(out_dir, "aq_numbers.tex")
    with open(path, "w") as fh:
        fh.write(
            "% Auto-generated by sweep.py — seed-averaged over "
            f"{len(SEEDS)} noise realisations. Do not edit by hand.\n"
        )
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"Wrote LaTeX macros -> {path}")


def _configs(grid: str):
    seen = set()
    if grid == "full":
        for n_obs in NOBS_VALUES:
            for sig in SIGMA_VALUES:
                if (n_obs, sig) not in seen:
                    seen.add((n_obs, sig))
                    yield "grid", n_obs, sig
        return
    for sig in SIGMA_VALUES:
        if (FIXED_NOBS, sig) not in seen:
            seen.add((FIXED_NOBS, sig))
            yield "sigma", FIXED_NOBS, sig
    for n_obs in NOBS_VALUES:
        if (n_obs, FIXED_SIGMA) not in seen:
            seen.add((n_obs, FIXED_SIGMA))
            yield "nobs", n_obs, FIXED_SIGMA


def main() -> None:
    quick = "--quick" in sys.argv
    grid = "full" if ("--grid" in sys.argv and "full" in sys.argv) else "cross"
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(out_dir, "air_quality_adv_diff_sweep.csv")

    params = plume.adv_diff_params()
    truth = plume.solve_truth(n_fine=161, params=params)
    ts = plume.make_truth_sampler(truth)

    configs = list(_configs(grid))
    total = len(configs) * len(SEEDS)
    print(
        f"Sweep: design={grid}, {len(configs)} operating points x "
        f"{len(SEEDS)} seeds = {total} solves. quick={quick}.",
        flush=True,
    )

    rows = []
    t0 = time.time()
    done = 0
    for axis, n_obs, sig in configs:
        for seed in SEEDS:
            data = solve(
                n_obs=n_obs,
                sigma_obs=sig,
                obs_seed=seed,
                truth_sampler=ts,
                params=params,
                compute_constrained=not quick,
                n_samples=120,
            )
            diag = data["diagnostics"]
            rows.append(_row(axis, diag))
            done += 1
            print(
                f"  [{done:>3}/{total}] axis={axis:<5} n_obs={n_obs:<3} "
                f"sig={sig:<5} seed={seed}  "
                f"gain_held={diag['gain_rmse_held_rel']*100:5.1f}%  "
                f"P(C|E)={diag['p_ce_mc']:.3f}  ({time.time()-t0:.0f}s)",
                flush=True,
            )

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    def _avg(axis, key_x):
        sel = [r for r in rows if r["axis"] == axis]
        xs = sorted(set(r[key_x] for r in sel))
        print(f"\n{axis} sweep (seed-averaged, ratio-of-means held-RMSE gain):")
        print(f"  {key_x:>10}  gain_held%  gain_grid%  tail_red%   P(C|E)")
        for x in xs:
            grp = [r for r in sel if r[key_x] == x]
            gh = _rom(grp, "rmse_held_E", "rmse_held_EC")
            gg = _rom(grp, "rmse_grid_E", "rmse_grid_EC")
            tr = _rom(grp, "subzero_E", "subzero_EC")
            pc = _nm([r["p_ce_mc"] for r in grp])
            print(f"  {x:>10}  {gh:>9.1f}  {gg:>9.1f}  {tr:>8.1f}   {pc:>6.3f}")

    if grid == "cross":
        _avg("sigma", "sigma_obs")
        _avg("nobs", "n_obs")
    else:
        _avg("grid", "sigma_obs")
        _avg("grid", "n_obs")

    print(
        "\nHypothesis check: held-RMSE and tail-reduction gains should INCREASE "
        "with sigma_obs and DECREASE with n_obs."
    )
    write_numbers_tex(rows, out_dir)


if __name__ == "__main__":
    main()
