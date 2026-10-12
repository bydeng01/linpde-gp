"""
Sweep the inequality's RMSE gain against observation noise (sigma_obs) and
observation count (n_obs) for the 2D Poisson data-assimilation experiment.

This produces the headline numbers for ``figure.py`` and
``assim_numbers.tex``. The central hypothesis is that the
monotonicity inequality's *marginal value* --- the reduction in posterior-mean
RMSE it buys on top of the PDE + noisy data --- grows as the data degrade
(noisier or sparser), and shrinks toward zero as the data alone determine u.

Design
------
By default we sweep a *cross* through the (sigma_obs, n_obs) plane rather than
the full Cartesian grid, because each constrained solve calls gp_constr's R
truncated-MVN moment computation (cost ~ O(d^2) in the constraint dimension
d = (#axes) * n_v^2 = 32). The cross gives both headline curves at a fraction of
the cost:

  * vs sigma_obs : n_obs fixed at FIXED_NOBS, sigma_obs over SIGMA_VALUES;
  * vs n_obs     : sigma_obs fixed at FIXED_SIGMA, n_obs over NOBS_VALUES.

Each operating point is repeated over SEEDS observation realisations so the
gain can be reported with a mean +/- std band. Pass ``--grid full`` for the
full Cartesian product (feeds the optional gain heatmap in figure.py) at
correspondingly higher R cost.

Modes
-----
  * default        : full pipeline (needs gp_constr's R backend), STARVED-corner
                     headline (obs_hi = OBS_HI_STARVED). Writes the unconstrained
                     and constrained RMSE, hence the mean gain. CSV:
                     ``poisson_2d_data_assimilation_sweep.csv``.
  * ``--uniform``  : scatter observations over the FULL domain (contrast). The
                     inequality then mostly trims the band (weak/negative mean
                     gain). CSV: ``..._sweep_uniform.csv``.
  * ``--quick``    : R-free. Computes only the unconstrained RMSE and the MC
                     P(C|E). Use it to characterise the regime / pick FIXED_* and
                     the value ranges before committing to the (slow) R sweep.

Output
------
A CSV ``poisson_2d_data_assimilation_sweep.csv`` in this directory.

Run from the repo root inside the container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/poisson_2d_data_assimilation/sweep.py
    # fast, no R:
    PYTHONPATH=... python .../sweep.py --quick
    # full grid for the heatmap:
    PYTHONPATH=... python .../sweep.py --grid full
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

if "--from-csv" in sys.argv:
    # Macro-only refresh reads the committed CSV; it needs neither the solver
    # stack nor R. Keep the heavy import out of that path.
    solve_assimilation = None
else:
    from run import solve_assimilation


# --- Sweep ranges. Keep these in sync with figure.py / the .tex caption. ---
SIGMA_VALUES = [0.0, 0.02, 0.05, 0.10, 0.20]  # observation-noise std
NOBS_VALUES = [6, 10, 16, 25, 40]  # number of scattered obs
SEEDS = [0, 1, 2, 3, 4]  # noise realisations (5 -> smoother
# error bars; locations are fixed
# Halton, so seeds vary noise only)

FIXED_NOBS = 16  # n_obs held fixed while sweeping sigma_obs
FIXED_SIGMA = 0.10  # sigma_obs held fixed while sweeping n_obs

# Observation sampling box. The HEADLINE (default) leaves the (1,1) corner
# DATA-FREE (obs_hi = OBS_HI_STARVED), reproducing the partial-BC showcase
# geometry under noise: this is the regime in which the *scale-free*
# monotonicity inequality corrects the MEAN, so the "RMSE gain grows as data
# degrade" story holds on the mean. Pass ``--uniform`` to scatter observations
# over the FULL domain instead; there the smooth PDE + data already determine u
# and the inequality only trims the predictive band (a contrast, not the
# headline).
OBS_HI_FULL = 1.0 - 1e-3
OBS_HI_STARVED = 0.85

_FIELDS = [
    "axis",
    "n_obs",
    "sigma_obs",
    "obs_seed",
    "rmse_unc",
    "rmse_con",
    "gain_abs",
    "gain_rel",
    "band_unc",
    "band_con",
    "max_abs_unc",
    "max_abs_con",
    "min_d0_unc",
    "min_d1_unc",
    "p_ce_mc",
    "p_ce",
    "constrained_ok",
]


def _row(axis: str, diag: dict, constrained_ok: bool) -> dict:
    return {
        "axis": axis,
        "n_obs": diag["n_obs"],
        "sigma_obs": diag["sigma_obs"],
        "obs_seed": diag["obs_seed"],
        "rmse_unc": diag["rmse_unc"],
        "rmse_con": diag.get("rmse_con", float("nan")),
        "gain_abs": diag.get("gain_abs", float("nan")),
        "gain_rel": diag.get("gain_rel", float("nan")),
        "band_unc": diag["band_unc"],
        "band_con": diag.get("band_con", float("nan")),
        "max_abs_unc": diag["max_abs_unc"],
        "max_abs_con": diag.get("max_abs_con", float("nan")),
        "min_d0_unc": diag.get("min_d0_unc", float("nan")),
        "min_d1_unc": diag.get("min_d1_unc", float("nan")),
        "p_ce_mc": diag["p_ce_mc"],
        "p_ce": diag.get("p_ce", float("nan")),
        "constrained_ok": int(constrained_ok),
    }


def _nm(vals):
    """nanmean returning nan (no warning) when every entry is nan."""
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _ns(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.std()) if arr.size else float("nan")


def _fmt_pce(p: float) -> str:
    """Format P(C|E) for LaTeX math mode."""
    if p != p:  # nan
        return "\\mathrm{n/a}"
    if p <= 0.0:
        return "0"
    if p < 0.01:
        import math

        e = math.floor(math.log10(p))
        m = p / 10**e
        return f"{m:.0f}\\times10^{{{e}}}"
    return f"{p:.2f}"


def write_numbers_tex(rows: list[dict], out_dir: str) -> None:
    """Emit ``assim_numbers.tex`` with \\newcommand macros for the section's
    prose and table, computed seed-averaged from the sweep rows. The section
    ``\\input``s this file, so one R-backed run refreshes every number; the file
    is only written for the headline (starved) sweep that has constrained
    results, so the ``--uniform`` / ``--quick`` runs don't clobber it.
    """
    if not any(r["gain_rel"] == r["gain_rel"] for r in rows):
        print(
            "NOTE: no constrained results — assim_numbers.tex not refreshed "
            "(this was a --quick run)."
        )
        return

    def at(n_filter=None, s_filter=None):
        out = rows
        if n_filter is not None:
            out = [r for r in out if r["n_obs"] == n_filter]
        if s_filter is not None:
            out = [r for r in out if abs(r["sigma_obs"] - s_filter) < 1e-12]
        return out

    # sigma curve at FIXED_NOBS (exclude the noiseless sigma=0 anchor).
    sig_pts = sorted(
        s for s in set(r["sigma_obs"] for r in at(n_filter=FIXED_NOBS)) if s > 0
    )
    sig_lo, sig_hi = sig_pts[0], sig_pts[-1]

    # Gains are reported as RATIO-OF-MEANS, 1 - mean(rmse_con)/mean(rmse_unc),
    # NOT the mean of the per-seed ratios. The per-seed ratio divides by a noisy
    # denominator and is extremely high-variance (its curve is non-monotone and
    # its std spans zero); the ratio-of-means is a stable estimator of the
    # aggregate skill and is what the paired-RMSE figure shows. See figure.py.
    def rom(group):
        ru = _nm([r["rmse_unc"] for r in group])
        rc = _nm([r["rmse_con"] for r in group])
        return 100.0 * (1.0 - rc / ru) if ru > 0 else float("nan")

    g_sig_lo = rom(at(FIXED_NOBS, sig_lo))
    g_sig_hi = rom(at(FIXED_NOBS, sig_hi))

    # n curve at FIXED_SIGMA. NLo = the n with the largest seed-averaged gain
    # (the sparse end, robust to the very-sparse magnitude-underpinning dip);
    # NHi = the densest.
    n_pts = sorted(set(r["n_obs"] for r in at(s_filter=FIXED_SIGMA)))
    gains_n = {n: rom(at(n, FIXED_SIGMA)) for n in n_pts}
    n_lo = max(gains_n, key=gains_n.get)
    n_hi = n_pts[-1]

    # Table cell (FIXED_NOBS, FIXED_SIGMA), seed-averaged.
    tc = at(FIXED_NOBS, FIXED_SIGMA)
    rmse_unc = _nm([r["rmse_unc"] for r in tc])
    rmse_con = _nm([r["rmse_con"] for r in tc])
    # Compute the table gain from the displayed-precision RMSEs so the table is
    # internally consistent (caption % matches the two RMSE cells).
    gain = 100.0 * (1.0 - round(rmse_con, 3) / round(rmse_unc, 3))
    band_unc = _nm([r["band_unc"] for r in tc])
    band_con = _nm([r["band_con"] for r in tc])
    max_unc = _nm([r["max_abs_unc"] for r in tc])
    max_con = _nm([r["max_abs_con"] for r in tc])
    min_slope = min(
        _nm([r["min_d0_unc"] for r in tc]), _nm([r["min_d1_unc"] for r in tc])
    )
    p_ce = _nm([r["p_ce"] for r in tc])
    if p_ce != p_ce:  # exact (R) unavailable -> MC
        p_ce = _nm([r["p_ce_mc"] for r in tc])

    # sigma = 0 anchor (noise-free endpoint): the redundancy regime. Reported
    # to 3 d.p. rather than via _fmt_pce, which would round 0.9976 to "1.00".
    zero = at(FIXED_NOBS, 0.0)
    if not zero:
        raise RuntimeError(
            "sigma_obs = 0 anchor missing at n_obs = %s; assim_numbers.tex "
            "needs it for AssimPCESigZero." % FIXED_NOBS
        )
    g_sig_zero = rom(zero)
    p_ce_zero = _nm([r["p_ce"] for r in zero])
    if p_ce_zero != p_ce_zero:  # exact (R) unavailable -> MC
        p_ce_zero = _nm([r["p_ce_mc"] for r in zero])

    # Per-seed reversals: cells where the constrained arm scored WORSE. These
    # are per-repetition ratios, NOT the ratio-of-means gain reported above.
    sig_cells = [r for r in at(n_filter=FIXED_NOBS) if r["sigma_obs"] > 0]
    n_cells = at(s_filter=FIXED_SIGMA)

    def _neg(cells):
        return sum(
            1 for r in cells if r["gain_rel"] == r["gain_rel"] and r["gain_rel"] < 0
        )

    macros = {
        "AssimSeeds": f"{len(SEEDS)}",
        "AssimNobsFixed": f"{FIXED_NOBS}",
        "AssimSigFixed": f"{FIXED_SIGMA:g}",
        "AssimSigLo": f"{sig_lo:g}",
        "AssimGainSigLo": f"{g_sig_lo:.0f}",
        "AssimSigHi": f"{sig_hi:g}",
        "AssimGainSigHi": f"{g_sig_hi:.0f}",
        "AssimNLo": f"{n_lo}",
        "AssimGainNLo": f"{gains_n[n_lo]:.0f}",
        "AssimNHi": f"{n_hi}",
        "AssimGainNHi": f"{gains_n[n_hi]:.0f}",
        "AssimRmseUnc": f"{rmse_unc:.3f}",
        "AssimRmseCon": f"{rmse_con:.3f}",
        "AssimGain": f"{gain:.0f}",
        "AssimBandUnc": f"{band_unc:.2f}",
        "AssimBandCon": f"{band_con:.2f}",
        "AssimMaxUnc": f"{max_unc:.2f}",
        "AssimMaxCon": f"{max_con:.2f}",
        "AssimMinSlope": f"{min_slope:.2f}",
        "AssimPCE": _fmt_pce(p_ce),
        "AssimGainSigZero": f"{g_sig_zero:.1f}",
        "AssimPCESigZero": f"{p_ce_zero:.3f}",
        "AssimNegCellsSig": f"{_neg(sig_cells)}",
        "AssimCellsSig": f"{len(sig_cells)}",
        "AssimNegCellsN": f"{_neg(n_cells)}",
        "AssimCellsN": f"{len(n_cells)}",
    }
    path = os.path.join(out_dir, "assim_numbers.tex")
    with open(path, "w") as fh:
        fh.write(
            "% Auto-generated by sweep.py — seed-averaged over "
            f"{len(SEEDS)} noise realisations. Do not edit by hand.\n"
        )
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"Wrote LaTeX macros -> {path}")


def _configs(grid: str):
    """Yield (axis_label, n_obs, sigma_obs) operating points (deduplicated)."""
    seen = set()
    if grid == "full":
        for n_obs in NOBS_VALUES:
            for sig in SIGMA_VALUES:
                key = (n_obs, sig)
                if key not in seen:
                    seen.add(key)
                    yield "grid", n_obs, sig
        return
    # 'cross'
    for sig in SIGMA_VALUES:
        key = (FIXED_NOBS, sig)
        if key not in seen:
            seen.add(key)
            yield "sigma", FIXED_NOBS, sig
    for n_obs in NOBS_VALUES:
        key = (n_obs, FIXED_SIGMA)
        if key not in seen:
            seen.add(key)
            yield "nobs", n_obs, FIXED_SIGMA


def _rows_from_csv(path: str) -> list[dict]:
    """Re-read a sweep CSV into the in-memory row format ``write_numbers_tex``
    expects. Used by ``--from-csv`` to refresh the macros without re-solving."""
    int_fields = {"n_obs", "obs_seed", "constrained_ok"}
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                if k == "axis":
                    row[k] = v
                elif k in int_fields:
                    row[k] = int(float(v))
                else:
                    row[k] = float(v) if v not in ("", "nan") else float("nan")
            rows.append(row)
    return rows


def main() -> None:
    quick = "--quick" in sys.argv
    uniform = "--uniform" in sys.argv  # default is the starved-corner headline
    grid = "full" if "--grid" in sys.argv and "full" in sys.argv else "cross"
    obs_hi = OBS_HI_FULL if uniform else OBS_HI_STARVED
    out_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = "_uniform" if uniform else ""  # headline (starved) has no suffix
    out_csv = os.path.join(out_dir, f"poisson_2d_data_assimilation_sweep{suffix}.csv")

    if "--from-csv" in sys.argv:
        # Macro-only refresh: re-derive assim_numbers.tex from the committed
        # CSV. No solves, no R. Every macro is a pure function of the CSV rows,
        # so this reproduces the existing values exactly.
        if uniform:
            print("NOTE: --uniform run; assim_numbers.tex left untouched.")
            return
        rows = _rows_from_csv(out_csv)
        print(f"Read {len(rows)} rows <- {out_csv} (no solves; R not required).")
        write_numbers_tex(rows, out_dir)
        return

    configs = list(_configs(grid))
    total = len(configs) * len(SEEDS)
    print(
        f"Sweep: design={grid}, obs_box=[0,{obs_hi:.2f}]^2 "
        f"({'full domain (contrast)' if uniform else 'STARVED corner (headline)'}), "
        f"{len(configs)} operating points x {len(SEEDS)} seeds = {total} solves. "
        f"quick={quick} (constrained/R {'OFF' if quick else 'ON'}).",
        flush=True,
    )

    rows = []
    t0 = time.time()
    done = 0
    for axis, n_obs, sig in configs:
        # Compute the (expensive, rtmvnorm-backed) constrained credible band ONLY
        # at the table operating point (FIXED_NOBS, FIXED_SIGMA); the gain curves
        # need only the constrained mean, so every other cell stays mean-only to
        # keep the R cost down. This yields a seed-averaged band_con for the
        # table without paying for samples across the whole sweep.
        is_table_cell = n_obs == FIXED_NOBS and abs(sig - FIXED_SIGMA) < 1e-12
        for seed in SEEDS:
            data = solve_assimilation(
                n_obs=n_obs,
                sigma_obs=sig,
                obs_seed=seed,
                obs_hi=obs_hi,
                compute_constrained=not quick,
                compute_band=(is_table_cell and not quick),
            )
            diag = data["diagnostics"]
            rows.append(_row(axis, diag, data["constrained_ok"]))
            done += 1
            gr = diag.get("gain_rel", float("nan"))
            print(
                f"  [{done:>3}/{total}] axis={axis:<5} n_obs={n_obs:<3} "
                f"sigma={sig:<5} seed={seed}  "
                f"rmse_unc={diag['rmse_unc']:.4f} "
                f"rmse_con={diag.get('rmse_con', float('nan')):.4f} "
                f"gain_rel={gr:.3f}  p_ce_mc={diag['p_ce_mc']:.3f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    # --- Compact seed-averaged summary to stdout. ---
    def _avg(axis, key_x):
        sel = [r for r in rows if r["axis"] == axis]
        xs = sorted(set(r[key_x] for r in sel))
        print(f"\n{axis} sweep (seed-averaged):")
        hdr = f"  {key_x:>10}  gain_rel(mean+/-std)   rmse_unc   rmse_con   p_ce_mc"
        print(hdr)
        for x in xs:
            grp = [r for r in sel if r[key_x] == x]
            gm = _nm([r["gain_rel"] for r in grp])
            gs = _ns([r["gain_rel"] for r in grp])
            ru = _nm([r["rmse_unc"] for r in grp])
            rc = _nm([r["rmse_con"] for r in grp])
            pc = _nm([r["p_ce_mc"] for r in grp])
            print(
                f"  {x:>10}  {gm:>7.3f} +/- {gs:>5.3f}    "
                f"{ru:>8.4f}   {rc:>8.4f}   {pc:>6.3f}"
            )

    if grid == "cross":
        _avg("sigma", "sigma_obs")
        _avg("nobs", "n_obs")
    else:
        # For the full grid, summarise gain by each axis marginally.
        _avg("grid", "sigma_obs")
        _avg("grid", "n_obs")

    print(
        "\nHypothesis check: gain_rel should INCREASE with sigma_obs and "
        "DECREASE with n_obs (the inequality helps more as data degrade)."
    )

    # Refresh the LaTeX macros for the section (headline starved sweep only, so
    # the --uniform contrast does not overwrite them).
    if not uniform:
        write_numbers_tex(rows, out_dir)
    else:
        print(
            "NOTE: --uniform run; assim_numbers.tex (headline macros) left "
            "untouched."
        )


if __name__ == "__main__":
    main()
