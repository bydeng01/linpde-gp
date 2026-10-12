"""
Real-data (SPX) thinned-chain sweep: the headline SPX experiment.

The dense SPX chain is a *negative control* --- thousands of tightly quoted
strikes pin the surface so well that the no-arbitrage inequalities are inert
(quotes+E+C == quotes+E; see run.py --dense). The publishable claim is that
the model-free no-arbitrage information recovers its value as the market gets
illiquid. We therefore THIN the chain to ``n_obs`` training quotes (seeded
random subsets of the bulk, so each seed is a different sparse market) with a
quote-noise floor, and trace the held-out RMSE of quotes+E vs quotes+E+C as a
function of ``n_obs``. The held-out set is fixed (the full wing / long-T block)
so the score target does not move.

This mirrors the synthetic ``sweep.py`` but on real market data and with the
constant-sigma Dupire PDE demoted to a soft prior (``sigma_pde > 0``), which is
the honest treatment when the model is misspecified: the equality side carries
a model assumption, the inequalities carry model-free truth.

Gain is reported as RATIO-OF-MEANS, 1 - mean(rmse_con)/mean(rmse_unc), over
SEEDS subsets (per-seed ratios divide by a noisy denominator; same estimator
decision as the synthetic sweep).

Run from the repo root inside the container (needs the R backend; ~minutes per
constrained solve):

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/black_scholes_surface/sweep_spx.py \\
        --data data/spx_chain_<date>.csv
    # R-free regime scout (no constrained column):
    PYTHONPATH=... python .../sweep_spx.py --data <csv> --quick

Output: ``black_scholes_spx_sweep.csv`` + ``bs_spx_numbers.tex`` in this
directory.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
from run import SPX_NOISE_FLOOR, load_chain_csv, solve_bs

# Retained-quote counts: from very illiquid up toward the dense control.
NOBS_VALUES = [12, 20, 35, 60, 120, 300]
# Number of thinned subsets (independent sparse markets) per n_obs. Each seed is
# a fresh random subset (run.py thins via default_rng(2000 + obs_seed)), so more
# seeds = more independent draws, not repeats. Override at launch with --seeds N
# (15-25 recommended; the 5-seed default was too noisy to pin the trend). At
# ~535 s/constrained solve the wall-clock is ~len(NOBS_VALUES)*N*535 s, i.e.
# ~13 h at N=15, ~18 h at N=20, ~22 h at N=25.
SEEDS = list(range(20))
NOISE_FLOOR = SPX_NOISE_FLOOR  # mimic wide / stale quotes
SIGMA_PDE = 1e-3  # misspecified Dupire as a soft prior
FIXED_NOBS = 20  # the headline illiquid operating point

_FIELDS = [
    "n_obs",
    "obs_seed",
    "rmse_unc",
    "rmse_con",
    "gain_abs",
    "gain_rel",
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


def _row(n_obs: int, seed: int, out: dict) -> dict:
    dE = out["results"]["quotes+E"]["diagnostics"]
    dEC = out["results"]["quotes+E+C"]["diagnostics"]
    ru, rc = dE["rmse_test"], dEC.get("rmse_test", float("nan"))
    return {
        "n_obs": n_obs,
        "obs_seed": seed,
        "rmse_unc": ru,
        "rmse_con": rc,
        "gain_abs": ru - rc,
        "gain_rel": (1.0 - rc / ru) if ru > 0 else float("nan"),
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
        "constrained_ok": int(bool(dEC["ok"])),
    }


def _nm(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _nmed(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _fmt_pce(p: float) -> str:
    if p != p:
        return "\\mathrm{n/a}"
    if p <= 0.0:
        return "<10^{-3}"
    if p < 0.01:
        import math

        e = math.floor(math.log10(p))
        return f"{p / 10 ** e:.0f}\\times10^{{{e}}}"
    return f"{p:.2f}"


def write_numbers_tex(rows, dense, out_dir):
    """Emit ``bs_spx_numbers.tex`` (seed-averaged ratio-of-means). Needs the
    constrained column; --quick leaves it untouched."""
    if not any(r["gain_rel"] == r["gain_rel"] for r in rows):
        print("NOTE: no constrained results — bs_spx_numbers.tex not refreshed.")
        return

    def at(n):
        return [r for r in rows if r["n_obs"] == n]

    def rom(grp):
        ru, rc = _nm([r["rmse_unc"] for r in grp]), _nm([r["rmse_con"] for r in grp])
        return 100.0 * (1.0 - rc / ru) if ru > 0 else float("nan")

    def gmed(grp):
        return 100.0 * _nmed([r["gain_rel"] for r in grp])

    def gpos(grp):
        gs = [r["gain_rel"] for r in grp if r["gain_rel"] == r["gain_rel"]]
        return 100.0 * sum(1 for x in gs if x > 0) / len(gs) if gs else float("nan")

    n_pts = sorted(set(r["n_obs"] for r in rows))
    n_lo, n_hi = n_pts[0], n_pts[-1]
    tc = at(FIXED_NOBS) or at(n_lo)
    ru, rc = _nm([r["rmse_unc"] for r in tc]), _nm([r["rmse_con"] for r in tc])
    # Dip point: the n_obs with the weakest *median* gain, i.e. where the
    # ratio-of-means improvement is least representative of a typical subset
    # (effect carried by a few high-error seeds rather than the median).
    dip = min(n_pts, key=lambda n: gmed(at(n)))
    macros = {
        "BsSpxSeeds": f"{len(SEEDS)}",
        "BsSpxNoise": f"{NOISE_FLOOR:g}",
        "BsSpxNobs": f"{tc[0]['n_obs']}",
        "BsSpxNLo": f"{n_lo}",
        "BsSpxGainNLo": f"{rom(at(n_lo)):.0f}",
        "BsSpxNHi": f"{n_hi}",
        "BsSpxGainNHi": f"{rom(at(n_hi)):.0f}",
        "BsSpxRmseUnc": f"{ru:.3f}",
        "BsSpxRmseCon": f"{rc:.3f}",
        "BsSpxGain": f"{100 * (1 - rc / ru):.0f}",
        "BsSpxGainMed": f"{gmed(tc):.0f}",
        "BsSpxPosFrac": f"{gpos(tc):.0f}",
        "BsSpxGainMedNLo": f"{gmed(at(n_lo)):.0f}",
        "BsSpxPosFracNLo": f"{gpos(at(n_lo)):.0f}",
        "BsSpxNDip": f"{dip}",
        "BsSpxGainDip": f"{rom(at(dip)):.0f}",
        "BsSpxGainMedDip": f"{gmed(at(dip)):.0f}",
        "BsSpxPosFracDip": f"{gpos(at(dip)):.0f}",
        "BsSpxBandUnc": f"{_nm([r['band_unc'] for r in tc]):.3f}",
        "BsSpxBandCon": f"{_nm([r['band_con'] for r in tc]):.3f}",
        "BsSpxViolCalUnc": f"{100 * _nm([r['viol_cal_unc'] for r in tc]):.0f}",
        "BsSpxViolCalCon": f"{100 * _nm([r['viol_cal_con'] for r in tc]):.0f}",
        "BsSpxPCE": _fmt_pce(_nm([r["p_ce"] for r in tc])),
        "BsSpxDenseRmseQuotes": f"{dense['rmse_quotes']:.3f}",
        "BsSpxDenseRmseUnc": f"{dense['rmse_unc']:.3f}",
        "BsSpxDenseRmseCon": f"{dense['rmse_con']:.3f}",
        "BsSpxDenseGain": f"{0.0 if round(dense['gain']) == 0 else dense['gain']:.0f}",
        "BsSpxDenseNobs": f"{dense['n_obs']}",
    }
    path = os.path.join(out_dir, "bs_spx_numbers.tex")
    with open(path, "w") as fh:
        fh.write(
            "% Auto-generated by sweep_spx.py on the SPX chain — "
            f"seed-averaged over {len(SEEDS)} thinned subsets. Do not edit.\n"
        )
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"Wrote LaTeX macros -> {path}")


def main() -> None:
    global SEEDS
    argv = sys.argv[1:]
    quick = "--quick" in argv
    if "--seeds" in argv:
        SEEDS = list(range(int(argv[argv.index("--seeds") + 1])))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    if "--data" not in argv:
        raise SystemExit("sweep_spx.py requires --data <chain.csv>.")
    csv_path = argv[argv.index("--data") + 1]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(out_dir, csv_path)
    data = load_chain_csv(csv_path)
    print(
        f"SPX sweep: pool {data['n_train']} train / {data['n_test']} test, "
        f"S0={data['S0']:.0f}, sigma={data['sigma']:.3f}, r={data['r']:.3f}; "
        f"thinning over {NOBS_VALUES} x {len(SEEDS)} seeds, noise floor "
        f"{NOISE_FLOOR:g}, sigma_pde={SIGMA_PDE:g}. quick={quick}.",
        flush=True,
    )

    rows = []
    t0 = time.time()
    total = len(NOBS_VALUES) * len(SEEDS)
    done = 0
    for n_obs in NOBS_VALUES:
        is_headline = n_obs == FIXED_NOBS
        for seed in SEEDS:
            out = solve_bs(
                obs_seed=seed,
                arms=("quotes+E", "quotes+E+C"),
                data=data,
                thin_n_obs=n_obs,
                sigma_obs_floor=NOISE_FLOOR,
                sigma_pde=SIGMA_PDE,
                compute_constrained=not quick,
                compute_band=(is_headline and not quick),
            )
            rows.append(_row(n_obs, seed, out))
            done += 1
            r = rows[-1]
            print(
                f"  [{done:>3}/{total}] n_obs={n_obs:<4} seed={seed}  "
                f"rmse_unc={r['rmse_unc']:.4f} rmse_con={r['rmse_con']:.4f} "
                f"gain_rel={r['gain_rel']:.3f}  p_ce_mc={r['p_ce_mc']:.3f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    # Dense negative control (one solve, all retained quotes). The quotes-only
    # arm is included so the misspecified-PDE-alone RMSE is computed in-pipeline
    # (was a hand-copied literal in the section before); the whole block is
    # persisted to black_scholes_spx_dense.json as the auditable raw record
    # behind the \BsSpxDense* macros (parallels black_scholes_spx_sweep.csv).
    dense = dict(
        n_obs=int(data["n_train"]),
        rmse_quotes=float("nan"),
        rmse_unc=float("nan"),
        rmse_con=float("nan"),
        gain=float("nan"),
    )
    if not quick:
        out = solve_bs(
            obs_seed=0,
            arms=("quotes", "quotes+E", "quotes+E+C"),
            data=data,
            sigma_pde=SIGMA_PDE,
            compute_constrained=True,
            compute_band=False,
        )
        dQ = out["results"]["quotes"]["diagnostics"]
        dE = out["results"]["quotes+E"]["diagnostics"]
        dEC = out["results"]["quotes+E+C"]["diagnostics"]
        ru, rc = dE["rmse_test"], dEC.get("rmse_test", float("nan"))
        dense.update(
            rmse_quotes=dQ["rmse_test"],
            rmse_unc=ru,
            rmse_con=rc,
            gain=100.0 * (1 - rc / ru) if ru > 0 else float("nan"),
        )
        with open(os.path.join(out_dir, "black_scholes_spx_dense.json"), "w") as fh:
            json.dump(dense, fh, indent=2)
        print(
            f"\nDENSE control (n_obs={dense['n_obs']}): "
            f"rmse_quotes={dense['rmse_quotes']:.4f} rmse_unc={ru:.4f} "
            f"rmse_con={rc:.4f} gain={dense['gain']:.1f}%"
        )

    suffix = "_quick" if quick else ""
    out_csv = os.path.join(out_dir, f"black_scholes_spx_sweep{suffix}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    print(f"\nthinned sweep (seed-averaged over {len(SEEDS)} subsets):")
    print(
        f"  {'n_obs':>6}  rmse_unc   rmse_con   gain_rom   gain_med   pos_frac   p_ce_mc"
    )
    for n in NOBS_VALUES:
        grp = [r for r in rows if r["n_obs"] == n]
        ru, rc = _nm([r["rmse_unc"] for r in grp]), _nm([r["rmse_con"] for r in grp])
        g = 1.0 - rc / ru if ru > 0 else float("nan")
        gains = [r["gain_rel"] for r in grp if r["gain_rel"] == r["gain_rel"]]
        gmed = _nmed([r["gain_rel"] for r in grp])
        pos = (sum(1 for x in gains if x > 0) / len(gains)) if gains else float("nan")
        print(
            f"  {n:>6}  {ru:>8.4f}   {rc:>8.4f}   {g:>8.3f}   {gmed:>8.3f}   "
            f"{pos:>8.2f}   {_nm([r['p_ce_mc'] for r in grp]):>7.3f}"
        )
    print(
        "\ngain_rom = ratio-of-means (1 - mean rc / mean ru); gain_med = median "
        "per-seed relative gain; pos_frac = fraction of seeds with gain > 0."
    )
    print(
        "Hypothesis: gain DECREASES toward the dense control as n_obs grows "
        "(constraints matter most where quotes are sparse). With more seeds, "
        "watch whether gain_med tracks gain_rom (robust effect) or stays ~0 "
        "while gain_rom is carried by a few high-error seeds (tail-clamping)."
    )

    if not quick:
        write_numbers_tex(rows, dense, out_dir)


if __name__ == "__main__":
    main()
