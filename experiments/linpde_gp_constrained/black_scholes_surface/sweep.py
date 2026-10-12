"""
Sweep the inequalities' held-out RMSE gain against quote count (n_obs) and
quote noise (sigma_obs) for the arbitrage-constrained call-surface experiment.

This produces the headline numbers for ``figure.py`` and
``bs_numbers.tex``. Hypothesis (as in poisson_2d_data_assimilation):
the no-arbitrage inequalities' marginal value over the Dupire-PDE equality
posterior --- the reduction in held-out RMSE on the extrapolation regions ---
grows as the quotes degrade (fewer / noisier) and shrinks as the data alone
pin the surface.

Design
------
Cross through the (n_obs, sigma_obs) plane (full Cartesian product costs
~5x the R truncated-MVN solves at no extra insight for the two curves):

  * vs n_obs     : sigma_obs fixed at FIXED_SIGMA, n_obs over NOBS_VALUES;
  * vs sigma_obs : n_obs fixed at FIXED_NOBS, sigma_obs over SIGMA_VALUES.

Each operating point is repeated over SEEDS noise realisations (quote sites
are fixed nested Halton, so seeds vary the noise only) and gains are reported
as RATIO-OF-MEANS, 1 - mean(rmse_con)/mean(rmse_unc) --- the per-seed ratio
divides by a noisy denominator and is too high-variance to plot (measured for
the Poisson assimilation sweep; same estimator here).

Modes
-----
  * default        : full pipeline (needs gp_constr's R backend). Writes
                     ``black_scholes_sweep.csv`` + ``bs_numbers.tex``.
  * ``--quick``    : R-free. Equality arms + MC P(C|E) only; regime scouting
                     before committing to the R sweep. CSV suffix ``_quick``.

Run from the repo root inside the container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/black_scholes_surface/sweep.py
    # fast, no R:
    PYTHONPATH=... python .../sweep.py --quick
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from run import solve_bs

# --- Sweep ranges. Keep these in sync with figure.py / the .tex caption. ---
NOBS_VALUES = [10, 16, 25, 40, 60]  # quotes at FIXED_SIGMA
SIGMA_VALUES = [0.002, 0.005, 0.01, 0.02, 0.04]  # noise at FIXED_NOBS
SEEDS = [0, 1, 2, 3, 4]

FIXED_NOBS = 16
FIXED_SIGMA = 0.01

_FIELDS = [
    "axis",
    "n_obs",
    "sigma_obs",
    "obs_seed",
    "rmse_unc",
    "rmse_con",
    "gain_abs",
    "gain_rel",
    "rmse_grid_unc",
    "rmse_grid_con",
    "band_unc",
    "band_con",
    "viol_spread_unc",
    "viol_cal_unc",
    "viol_bfly_unc",
    "viol_spread_con",
    "viol_cal_con",
    "viol_bfly_con",
    "p_ce_mc",
    "p_ce",
    "constrained_ok",
]


def _row(axis: str, out: dict) -> dict:
    dE = out["results"]["quotes+E"]["diagnostics"]
    dEC = out["results"]["quotes+E+C"]["diagnostics"]
    ok = bool(dEC["ok"])
    ru, rc = dE["rmse_test"], dEC.get("rmse_test", float("nan"))
    return {
        "axis": axis,
        "n_obs": out["params"]["n_obs"],
        "sigma_obs": out["params"]["sigma_obs"],
        "obs_seed": out["params"]["obs_seed"],
        "rmse_unc": ru,
        "rmse_con": rc,
        "gain_abs": ru - rc,
        "gain_rel": (1.0 - rc / ru) if ru > 0 else float("nan"),
        "rmse_grid_unc": dE["rmse_grid"],
        "rmse_grid_con": dEC.get("rmse_grid", float("nan")),
        "band_unc": dE["band"],
        "band_con": dEC.get("band", float("nan")),
        "viol_spread_unc": dE["viol_spread_rate"],
        "viol_cal_unc": dE["viol_cal_rate"],
        "viol_bfly_unc": dE["viol_bfly_rate"],
        "viol_spread_con": dEC.get("viol_spread_rate", float("nan")),
        "viol_cal_con": dEC.get("viol_cal_rate", float("nan")),
        "viol_bfly_con": dEC.get("viol_bfly_rate", float("nan")),
        "p_ce_mc": dE["p_ce_mc"],
        "p_ce": dEC.get("p_ce", float("nan")),
        "constrained_ok": int(ok),
    }


def _nm(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _fmt_pce(p: float) -> str:
    if p != p:
        return "\\mathrm{n/a}"
    if p <= 0.0:
        return "0"
    if p < 0.01:
        import math

        e = math.floor(math.log10(p))
        return f"{p / 10 ** e:.0f}\\times10^{{{e}}}"
    return f"{p:.2f}"


def write_numbers_tex(rows: list[dict], out_dir: str) -> None:
    """Emit ``bs_numbers.tex`` (\\newcommand macros, seed-averaged from the
    sweep rows; pattern: poisson_2d_data_assimilation/assim_numbers.tex). The
    section \\input s this file, so one R-backed sweep refreshes every number;
    a --quick run (no constrained results) leaves it untouched.
    """
    if not any(r["gain_rel"] == r["gain_rel"] for r in rows):
        print(
            "NOTE: no constrained results — bs_numbers.tex not refreshed "
            "(this was a --quick run)."
        )
        return

    def at(n=None, s=None):
        out = rows
        if n is not None:
            out = [r for r in out if r["n_obs"] == n]
        if s is not None:
            out = [r for r in out if abs(r["sigma_obs"] - s) < 1e-12]
        return out

    def rom(group):
        ru = _nm([r["rmse_unc"] for r in group])
        rc = _nm([r["rmse_con"] for r in group])
        return 100.0 * (1.0 - rc / ru) if ru > 0 else float("nan")

    sig_pts = sorted(set(r["sigma_obs"] for r in at(n=FIXED_NOBS)))
    n_pts = sorted(set(r["n_obs"] for r in at(s=FIXED_SIGMA)))
    sig_lo, sig_hi = sig_pts[0], sig_pts[-1]
    n_lo, n_hi = n_pts[0], n_pts[-1]

    tc = at(FIXED_NOBS, FIXED_SIGMA)
    rmse_unc = _nm([r["rmse_unc"] for r in tc])
    rmse_con = _nm([r["rmse_con"] for r in tc])
    gain = 100.0 * (1.0 - rmse_con / rmse_unc)
    p_ce = _nm([r["p_ce"] for r in tc])
    if p_ce != p_ce:
        p_ce = _nm([r["p_ce_mc"] for r in tc])

    macros = {
        "BsSeeds": f"{len(SEEDS)}",
        "BsNobsFixed": f"{FIXED_NOBS}",
        "BsSigFixed": f"{FIXED_SIGMA:g}",
        "BsSigLo": f"{sig_lo:g}",
        "BsGainSigLo": f"{rom(at(FIXED_NOBS, sig_lo)):.0f}",
        "BsSigHi": f"{sig_hi:g}",
        "BsGainSigHi": f"{rom(at(FIXED_NOBS, sig_hi)):.0f}",
        "BsNLo": f"{n_lo}",
        "BsGainNLo": f"{rom(at(n_lo, FIXED_SIGMA)):.0f}",
        "BsNHi": f"{n_hi}",
        "BsGainNHi": f"{rom(at(n_hi, FIXED_SIGMA)):.0f}",
        "BsRmseUnc": f"{rmse_unc:.3f}",
        "BsRmseCon": f"{rmse_con:.3f}",
        "BsGain": f"{gain:.0f}",
        "BsBandUnc": f"{_nm([r['band_unc'] for r in tc]):.3f}",
        "BsBandCon": f"{_nm([r['band_con'] for r in tc]):.3f}",
        "BsViolSpreadUnc": f"{100 * _nm([r['viol_spread_unc'] for r in tc]):.0f}",
        "BsViolCalUnc": f"{100 * _nm([r['viol_cal_unc'] for r in tc]):.0f}",
        "BsViolBflyUnc": f"{100 * _nm([r['viol_bfly_unc'] for r in tc]):.0f}",
        "BsViolSpreadCon": f"{100 * _nm([r['viol_spread_con'] for r in tc]):.0f}",
        "BsViolCalCon": f"{100 * _nm([r['viol_cal_con'] for r in tc]):.0f}",
        "BsViolBflyCon": f"{100 * _nm([r['viol_bfly_con'] for r in tc]):.0f}",
        "BsPCE": _fmt_pce(p_ce),
    }
    path = os.path.join(out_dir, "bs_numbers.tex")
    with open(path, "w") as fh:
        fh.write(
            "% Auto-generated by sweep.py — seed-averaged over "
            f"{len(SEEDS)} noise realisations. Do not edit by hand.\n"
        )
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"Wrote LaTeX macros -> {path}")


def _configs():
    """Yield (axis_label, n_obs, sigma_obs) cross points (deduplicated)."""
    seen = set()
    for n_obs in NOBS_VALUES:
        key = (n_obs, FIXED_SIGMA)
        if key not in seen:
            seen.add(key)
            yield "nobs", n_obs, FIXED_SIGMA
    for sig in SIGMA_VALUES:
        key = (FIXED_NOBS, sig)
        if key not in seen:
            seen.add(key)
            yield "sigma", FIXED_NOBS, sig


def main() -> None:
    quick = "--quick" in sys.argv
    out_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = "_quick" if quick else ""
    out_csv = os.path.join(out_dir, f"black_scholes_sweep{suffix}.csv")

    configs = list(_configs())
    total = len(configs) * len(SEEDS)
    print(
        f"Sweep: cross design, {len(configs)} operating points x "
        f"{len(SEEDS)} seeds = {total} solves. quick={quick} "
        f"(constrained/R {'OFF' if quick else 'ON'}).",
        flush=True,
    )

    rows = []
    t0 = time.time()
    done = 0
    for axis, n_obs, sig in configs:
        # The (expensive, rtmvnorm-backed) constrained band is computed ONLY
        # at the table cell (FIXED_NOBS, FIXED_SIGMA); the gain curves need
        # the constrained mean alone.
        is_table_cell = n_obs == FIXED_NOBS and abs(sig - FIXED_SIGMA) < 1e-12
        for seed in SEEDS:
            out = solve_bs(
                n_obs=n_obs,
                sigma_obs=sig,
                obs_seed=seed,
                arms=("quotes+E", "quotes+E+C"),
                compute_constrained=not quick,
                compute_band=(is_table_cell and not quick),
            )
            rows.append(_row(axis, out))
            done += 1
            r = rows[-1]
            print(
                f"  [{done:>3}/{total}] axis={axis:<5} n_obs={n_obs:<3} "
                f"sigma={sig:<6} seed={seed}  "
                f"rmse_unc={r['rmse_unc']:.4f} rmse_con={r['rmse_con']:.4f} "
                f"gain_rel={r['gain_rel']:.3f}  p_ce_mc={r['p_ce_mc']:.3f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    # --- Compact seed-averaged summary (ratio-of-means gain per cell). ---
    def _avg(axis, key_x):
        sel = [r for r in rows if r["axis"] == axis]
        xs = sorted(set(r[key_x] for r in sel))
        print(f"\n{axis} sweep (seed-averaged, ratio-of-means):")
        print(f"  {key_x:>10}  rmse_unc   rmse_con   gain_rom   p_ce_mc")
        for x in xs:
            grp = [r for r in sel if r[key_x] == x]
            ru = _nm([r["rmse_unc"] for r in grp])
            rc = _nm([r["rmse_con"] for r in grp])
            g = 1.0 - rc / ru if ru > 0 else float("nan")
            pc = _nm([r["p_ce_mc"] for r in grp])
            print(f"  {x:>10}  {ru:>8.4f}   {rc:>8.4f}   {g:>8.3f}   {pc:>7.3f}")

    _avg("nobs", "n_obs")
    _avg("sigma", "sigma_obs")
    print(
        "\nHypothesis check: gain_rom should DECREASE with n_obs and "
        "INCREASE with sigma_obs (the inequalities matter most where the "
        "quotes pin the surface least)."
    )

    if not quick:
        write_numbers_tex(rows, out_dir)


if __name__ == "__main__":
    main()
