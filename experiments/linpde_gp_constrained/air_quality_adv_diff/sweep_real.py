r"""Pooled real-data sparsity sweep for the positivity constraint (publishable curve).

Turns several real OpenAQ snapshots into a defensible, cross-validated real-data
result. For each snapshot we vary the monitor count and, for each count, average
over random monitor subsets (each subset is one cross-validation fold, since
run.solve holds out the first quarter of the passed monitors). Snapshots differ
in day / city / pollutant AND in meteorology, so each one carries its OWN PDE
coefficients -- wind direction, diffusivity D, decay k -- via a small config
file. We then POOL every (snapshot, n_mon, fold) cell and report, as
RATIO-OF-MEANS over the pool at each monitor count:

  * sub-zero-tail (calibration) reduction of monitors+E+C over monitors+E,
  * held-out RMSE gain of monitors+E+C over monitors+E,

each averaged across snapshots, with a per-snapshot breakdown and bootstrap error
bars across snapshots+folds. A ZERO-CLIPPING BASELINE (clip the +E posterior mean
at 0, keep its variance) is carried alongside: clipping repairs the mean but not
the predictive sub-zero mass, whereas the constrained posterior repairs both.

No ground truth on real data, so RMSE is held-out cross-validation only and the
calibration (tail) reduction is the headline metric.

Run from the repo root inside the container (needs the gp_constr R backend for
the exact constrained arm; falls back to the R-free sampler otherwise):

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \
      python experiments/linpde_gp_constrained/air_quality_adv_diff/sweep_real.py \
        --config experiments/.../air_quality_adv_diff/snapshots.txt

Backward-compatible single-wind form (one --wind applied to all CSVs):

    ... sweep_real.py --data csv1,csv2 --wind=-0.869,1.992

snapshots.txt format -- one snapshot per line, comma-separated, '#' comments
(the snapshots.txt next to this script is the pooled configuration):

    # path, wind_east, wind_north [, D, k, wind_mag, source]
    data/openaq_maryland_pm25_2021-07-15.csv, 2.756, 1.975
    data/<snapshot_b>.csv,                    0.55, -1.40, 0.05, 0.30
    data/<snapshot_c>.csv,                    1.10,  0.30, , , , zero

wind_east/wind_north is the advection vector (TOWARD which the wind blows, the
opposite of the meteorological FROM convention). Empty optional fields fall back to
the global defaults (--D / --k / --wind-mag / --source, in turn the synthetic
regime). argparse needs the equals form for a negative global wind: --wind=-0.8,2.

Outputs (next to this script; SVG + PDF + TIFF@600, no PNG):
    air_quality_adv_diff_real_sweep.csv        ('snapshot' col + scoring columns)
    figure_air_quality_adv_diff_real.{svg,pdf,tiff}        sparsity / calibration
    figure_air_quality_adv_diff_real_calib.{svg,pdf,tiff}  CRPS gain + coverage
    aq_real_numbers.tex                    (pooled LaTeX macros, read from the CSV)

Figures + macros are recoverable from the CSV, so `--from-csv` regenerates them
without the (slow, R-backed) solve.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plume  # noqa: E402
import run as runmod  # noqa: E402

MON_GRID = [6, 8, 10, 14, 18, 22]
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 2000  # bootstrap resamples for the ratio-of-means error bar.

_FIELDS = [
    "snapshot",
    "n_mon",
    "seed",
    "p_ce_exact",
    "p_ce_mc",
    "rmse_held_E",
    "rmse_held_EC",
    "rmse_held_clip",
    "rmse_held_mon",
    "gain_held_EC_rel",
    "gain_held_clip_rel",
    "subzero_E",
    "subzero_EC",
    "subzero_clip",
    "tail_red_EC_rel",
    "tail_red_clip_rel",
    "band_E",
    "band_EC",
    "constrained_ok",
    # held-out proper scoring rules + calibration (per arm): lower crps/nll/iscore
    # are better, cov95 targets 0.95. Gaussian predictive from each arm's
    # (mean, std) at the held-out monitors (the +E+C std is the truncated-sample
    # spread, so this scores its sharpening; the Gaussian form is an approximation
    # of the truncated predictive).
    "crps_E",
    "crps_EC",
    "crps_clip",
    "nll_E",
    "nll_EC",
    "nll_clip",
    "cov_E",
    "cov_EC",
    "cov_clip",
    "iscore_E",
    "iscore_EC",
    "iscore_clip",
]


# -----------------------------------------------------------------------------
# Snapshot specification (per-snapshot wind / D / k / source).
# -----------------------------------------------------------------------------
def _label(path: str) -> str:
    """Short snapshot label: strip the openaq_ prefix and .csv suffix."""
    base = os.path.basename(path)
    base = base[:-4] if base.endswith(".csv") else base
    return base[len("openaq_") :] if base.startswith("openaq_") else base


def _apply_spec(base_params: dict, wind, wind_mag, D, k):
    """Copy base_params and orient v along `wind` (only its direction is
    identifiable; |v'| = wind_mag, default = synthetic |v|); override D, k."""
    p = dict(base_params)
    p["v"] = tuple(float(x) for x in p["v"])
    if wind is not None:
        w = np.asarray(wind, float)
        if w.shape != (2,) or np.allclose(w, 0.0):
            raise SystemExit(f"wind must be 'east,north' and nonzero, got {wind}.")
        mag = float(wind_mag) if wind_mag is not None else float(np.hypot(*p["v"]))
        v = mag * w / np.linalg.norm(w)
        p["v"] = (float(v[0]), float(v[1]))
    if D is not None:
        p["D"] = float(D)
    if k is not None:
        p["k"] = float(k)
    return p


def _src_fn(source: str):
    """S=0 (advection-diffusion-decay transport prior; the real-data default) or
    the synthetic Gaussian plume (indefensible on real data; testing only)."""
    if source == "zero":
        return lambda X: np.zeros(np.atleast_2d(X).shape[0])
    return lambda X: plume.source(X)


def _specs_from_config(path: str, base_params, g_wind_mag, g_D, g_k, g_source):
    """Parse snapshots.txt -> list of (csv_path, label, params, src_fn, source).

    Line: path, wind_east, wind_north [, D, k, wind_mag, source]. Empty optional
    fields fall back to the global defaults. Relative paths resolve against the
    config file's directory, then the experiment directory.
    """
    cfg_dir = os.path.dirname(os.path.abspath(path))
    here = os.path.dirname(os.path.abspath(__file__))
    specs = []
    with open(path) as fh:
        for ln, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                raise SystemExit(f"{path}:{ln}: need at least 'path, wind_e, wind_n'.")
            csv_rel = parts[0]
            csv_path = next(
                (
                    c
                    for c in (
                        csv_rel,
                        os.path.join(cfg_dir, csv_rel),
                        os.path.join(here, csv_rel),
                    )
                    if os.path.exists(c)
                ),
                csv_rel,
            )
            wind = [float(parts[1]), float(parts[2])]

            def opt(i, cast=float):
                return cast(parts[i]) if len(parts) > i and parts[i] != "" else None

            D = opt(3) if opt(3) is not None else g_D
            k = opt(4) if opt(4) is not None else g_k
            wmag = opt(5) if opt(5) is not None else g_wind_mag
            source = parts[6] if len(parts) > 6 and parts[6] != "" else g_source
            params = _apply_spec(base_params, wind, wmag, D, k)
            specs.append((csv_path, _label(csv_path), params, _src_fn(source), source))
    if not specs:
        raise SystemExit(f"{path}: no snapshot lines.")
    return specs


def _specs_from_data(data_arg, g_wind, base_params, g_wind_mag, g_D, g_k, g_source):
    """Backward-compatible path: comma-separated CSVs, one global --wind for all."""
    params = _apply_spec(base_params, g_wind, g_wind_mag, g_D, g_k)
    src_fn = _src_fn("plume" if g_source == "plume" else "zero")
    src_lbl = "plume" if g_source == "plume" else "zero"
    return [(p, _label(p), params, src_fn, src_lbl) for p in data_arg.split(",")]


# -----------------------------------------------------------------------------
# Zero-clipping baseline.
# -----------------------------------------------------------------------------
def _clip_baseline(data):
    """Zero-clip the monitors+E mean (variance unchanged); held RMSE + sub-zero tail.

    Returns (rmse_held_clip, subzero_clip). The clipped mean is max(m_E, 0); the
    sub-zero tail uses the ORIGINAL +E std, so it reflects that clipping leaves
    predictive mass below zero (unlike the constrained posterior).
    """
    n = int(data["n_eval_axis"])
    X_eval = runmod._full_grid(n)[0]
    mE, sE = data["fields"]["monitors+E"]
    clipped = np.clip(mE, 0.0, None)
    Xh, Yh = data["X_held"], data["Y_held"]
    rmse = float(np.sqrt(np.mean((runmod.mean_at(clipped, X_eval, Xh) - Yh) ** 2)))
    s = np.clip(sE, 1e-12, None)
    subzero = float(np.mean(norm.cdf(-clipped / s)))
    return rmse, subzero


# -----------------------------------------------------------------------------
# Pooling helpers: ratio-of-means over a group of rows + bootstrap error bar.
# -----------------------------------------------------------------------------
def _pair(rows, ku, kc):
    """Joint-finite (u, c) arrays with u > 0, for ratio-of-means over a pool."""
    u = np.array([r[ku] for r in rows], float)
    c = np.array([r[kc] for r in rows], float)
    m = np.isfinite(u) & np.isfinite(c) & (u > 0)
    return u[m], c[m]


def _rom_pool(rows, ku, kc):
    """Ratio-of-means gain [%] over a pool: 100*(1 - sum(c)/sum(u))."""
    u, c = _pair(rows, ku, kc)
    if u.size == 0 or u.sum() <= 0:
        return float("nan")
    return 100.0 * (1.0 - c.sum() / u.sum())


def _rom_boot(rows, ku, kc, n_boot=N_BOOT, seed=20240613):
    """Bootstrap std of the pooled ratio-of-means (resamples rows = snapshot x fold)."""
    u, c = _pair(rows, ku, kc)
    if u.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    n = u.size
    est = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        su = u[idx].sum()
        if su > 0:
            est.append(100.0 * (1.0 - c[idx].sum() / su))
    return float(np.std(est)) if est else float("nan")


def _scores(mu, sd, y):
    """Held-out proper scoring rules + calibration for a Gaussian predictive.

    Returns nll (log-score), crps (closed-form Gaussian CRPS, Gneiting & Raftery
    2007), cov (empirical 95% coverage; target 0.95) and iscore (95% interval
    score). All recoverable from the sweep CSV, so the figures regenerate without
    re-solving (see --from-csv).
    """
    mu = np.asarray(mu, float)
    y = np.asarray(y, float)
    sd = np.clip(np.asarray(sd, float), 1e-9, None)
    z = (y - mu) / sd
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * z**2))
    crps = float(
        np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))
    )
    lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
    cov = float(np.mean((y >= lo) & (y <= hi)))
    a = 0.05
    iscore = float(
        np.mean(
            (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)
        )
    )
    return dict(nll=nll, crps=crps, cov=cov, iscore=iscore)


def _held_pred(data, arm, clip0=False):
    """(mu, sd) of an arm at the held-out monitors. clip0 zero-clips the mean
    (the clipping baseline keeps the +E std)."""
    n = int(data["n_eval_axis"])
    X_eval = runmod._full_grid(n)[0]
    mean_g, std_g = data["fields"][arm]
    mu = runmod.mean_at(mean_g, X_eval, data["X_held"])
    sd = runmod.mean_at(std_g, X_eval, data["X_held"])
    if clip0:
        mu = np.clip(mu, 0.0, None)
    return mu, sd


def _nm(vals):
    a = np.array(vals, float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def _load_rows(csv_path):
    """Read a sweep CSV back into typed rows (for --from-csv figure/tex regen)."""
    int_cols = {"n_mon", "seed", "constrained_ok"}
    rows = []
    with open(csv_path, newline="") as fh:
        for d in csv.DictReader(fh):
            r = {}
            for k, v in d.items():
                if k == "snapshot":
                    r[k] = v
                elif k in int_cols:
                    r[k] = int(float(v)) if v not in ("", "nan") else 0
                else:
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        r[k] = float("nan")
            rows.append(r)
    return rows


# -----------------------------------------------------------------------------
# Sweep.
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        default=None,
        help="snapshots.txt: 'path, wind_e, wind_n [, D, k, wind_mag, "
        "source]' per line (per-snapshot PDE coefficients).",
    )
    ap.add_argument(
        "--data",
        default=None,
        help="OpenAQ snapshot CSV(s), comma-separated (one global --wind).",
    )
    ap.add_argument(
        "--wind", default=None, help="global wind 'east,north' (direction)."
    )
    ap.add_argument("--wind-mag", type=float, default=None)
    ap.add_argument("--D", type=float, default=None)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--source", default="zero", choices=("zero", "plume"))
    ap.add_argument("--n-samples", type=int, default=120)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="write outputs here (default: this script's directory).",
    )
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="skip the (slow) solve: reload air_quality_adv_diff_real_sweep.csv "
        "and regenerate the figures, macros and verification only.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "air_quality_adv_diff_real_sweep.csv")

    # Figures/macros are recoverable from the CSV, so decouple them from the solve.
    if args.from_csv:
        rows = _load_rows(out_csv)
        Ns = sorted(set(r["n_mon"] for r in rows))
        snaps = sorted(set(r["snapshot"] for r in rows))
        print(f"--from-csv: {len(rows)} rows, {len(snaps)} snapshots from {out_csv}")
        _write_tex(rows, Ns, snaps, out_dir)
        _plot(rows, Ns, snaps, out_dir)
        _plot_calib(rows, Ns, out_dir)
        _verify(rows, Ns)
        return

    if not (args.config or args.data):
        ap.error("pass --config snapshots.txt or --data csv[,csv...] (or --from-csv).")

    base = plume.adv_diff_params()
    g_wind = [float(t) for t in args.wind.split(",")] if args.wind else None
    if args.config:
        specs = _specs_from_config(
            args.config, base, args.wind_mag, args.D, args.k, args.source
        )
    else:
        specs = _specs_from_data(
            args.data, g_wind, base, args.wind_mag, args.D, args.k, args.source
        )

    print(f"Pooling {len(specs)} snapshot(s):")
    for path, lab, p, _fn, src in specs:
        print(
            f"  {lab:<28} D={p['D']:.4g} v=({p['v'][0]:.3g},{p['v'][1]:.3g}) "
            f"k={p['k']:.4g} source={src}"
        )

    rows = []
    t0 = time.time()
    for path, lab, params, src_fn, _src in specs:
        if not os.path.exists(path):
            print(f"  SKIP {lab}: {path} not found.", flush=True)
            continue
        pre = runmod._preprocess_openaq(path)
        n_all = pre["X_obs"].shape[0]
        for N in MON_GRID:
            if N > n_all:
                continue
            for S in SEEDS:
                rng = np.random.default_rng(7000 + S)
                idx = rng.permutation(n_all)[:N]
                rd = dict(pre, X_obs=pre["X_obs"][idx], Y_obs=pre["Y_obs"][idx])
                data = runmod.solve(
                    real_data=rd,
                    params=params,
                    source_fn=src_fn,
                    n_samples=args.n_samples,
                )
                d = data["diagnostics"]
                E, EC = d["arms"]["monitors+E"], d["arms"]["monitors+E+C"]
                mon = d["arms"]["monitors"]
                rmse_clip, sz_clip = _clip_baseline(data)
                ok = int(np.isfinite(EC["rmse_held"]))

                # Held-out proper scoring rules + calibration, per arm.
                Yh = data["Y_held"]
                muE, sdE = _held_pred(data, "monitors+E")
                muC, sdC = _held_pred(data, "monitors+E+C")
                sE_obj = _scores(muE, sdE, Yh)
                sC_obj = (
                    _scores(muC, sdC, Yh)
                    if np.all(np.isfinite(muC))
                    else dict(nll=np.nan, crps=np.nan, cov=np.nan, iscore=np.nan)
                )
                sK_obj = _scores(np.clip(muE, 0.0, None), sdE, Yh)  # clipping baseline

                rows.append(
                    {
                        "snapshot": lab,
                        "n_mon": N,
                        "seed": S,
                        "p_ce_exact": d["p_ce_exact"],
                        "p_ce_mc": d["p_ce_mc"],
                        "rmse_held_E": E["rmse_held"],
                        "rmse_held_EC": EC["rmse_held"],
                        "rmse_held_clip": rmse_clip,
                        "rmse_held_mon": mon["rmse_held"],
                        "gain_held_EC_rel": d["gain_rmse_held_rel"],
                        "gain_held_clip_rel": (
                            (1 - rmse_clip / E["rmse_held"])
                            if E["rmse_held"] > 0
                            else float("nan")
                        ),
                        "subzero_E": E["subzero_tail"],
                        "subzero_EC": EC["subzero_tail"],
                        "subzero_clip": sz_clip,
                        "tail_red_EC_rel": (
                            (1 - EC["subzero_tail"] / E["subzero_tail"])
                            if E["subzero_tail"] > 0
                            else float("nan")
                        ),
                        "tail_red_clip_rel": (
                            (1 - sz_clip / E["subzero_tail"])
                            if E["subzero_tail"] > 0
                            else float("nan")
                        ),
                        "band_E": E["band"],
                        "band_EC": EC["band"],
                        "constrained_ok": ok,
                        "crps_E": sE_obj["crps"],
                        "crps_EC": sC_obj["crps"],
                        "crps_clip": sK_obj["crps"],
                        "nll_E": sE_obj["nll"],
                        "nll_EC": sC_obj["nll"],
                        "nll_clip": sK_obj["nll"],
                        "cov_E": sE_obj["cov"],
                        "cov_EC": sC_obj["cov"],
                        "cov_clip": sK_obj["cov"],
                        "iscore_E": sE_obj["iscore"],
                        "iscore_EC": sC_obj["iscore"],
                        "iscore_clip": sK_obj["iscore"],
                    }
                )
                print(
                    f"  {lab} N={N:<3} seed={S}  P(C|E)={d['p_ce_mc']:.3f}  "
                    f"held +E+C={EC['rmse_held']:.3f} clip={rmse_clip:.3f} "
                    f"E={E['rmse_held']:.3f}  ({time.time()-t0:.0f}s)",
                    flush=True,
                )

    if not rows:
        raise SystemExit("no rows produced -- check snapshot paths in the config.")

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    snaps = sorted(set(r["snapshot"] for r in rows))
    Ns = sorted(set(r["n_mon"] for r in rows))

    # Pooled (ratio-of-means across all snapshot x fold cells) vs monitor count.
    print(
        f"\nPooled across {len(snaps)} snapshot(s), {len(SEEDS)} folds "
        "(ratio-of-means) vs monitor count:"
    )
    print(
        f"  {'n_mon':>6}  {'#cells':>6}  {'P(C|E)':>7}  {'tail_red_EC%':>12} "
        f"{'(+/-)':>6}  {'tail_red_clip%':>14}  {'held_gain_EC%':>13}  "
        f"{'held_gain_clip%':>15}"
    )
    for N in Ns:
        g = [r for r in rows if r["n_mon"] == N]
        print(
            f"  {N:>6}  {len(g):>6}  {_nm([r['p_ce_mc'] for r in g]):>7.3f}  "
            f"{_rom_pool(g, 'subzero_E', 'subzero_EC'):>12.1f} "
            f"{_rom_boot(g, 'subzero_E', 'subzero_EC'):>6.1f}  "
            f"{_rom_pool(g, 'subzero_E', 'subzero_clip'):>14.1f}  "
            f"{_rom_pool(g, 'rmse_held_E', 'rmse_held_EC'):>13.1f}  "
            f"{_rom_pool(g, 'rmse_held_E', 'rmse_held_clip'):>15.1f}"
        )

    # Per-snapshot breakdown (ratio-of-means tail reduction within each snapshot).
    print("\nPer-snapshot tail_red_EC% (ratio-of-means within snapshot):")
    print(f"  {'snapshot':<28}" + "".join(f"{('N='+str(N)):>9}" for N in Ns))
    for snap in snaps:
        cells = [
            _rom_pool(
                [r for r in rows if r["snapshot"] == snap and r["n_mon"] == N],
                "subzero_E",
                "subzero_EC",
            )
            for N in Ns
        ]
        print(
            f"  {snap:<28}"
            + "".join((f"{c:>9.1f}" if np.isfinite(c) else f"{'-':>9}") for c in cells)
        )

    # Pooled proper scoring rules + calibration (held-out): CRPS / NLL / interval
    # score as ratio-of-means gains of +E+C over +E (positive = better), coverage
    # as the empirical 95% level (target 0.95). These are NOT tautological -- unlike
    # the sub-zero tail, the constraint can win or lose here.
    print(
        "\nPooled held-out scoring rules + calibration vs monitor count "
        "(gain% = +E+C over +E; cov target 0.95):"
    )
    print(
        f"  {'n_mon':>6}  {'crps_gain%':>10} {'(+/-)':>6}  {'nll_gain%':>9}  "
        f"{'is_gain%':>8}  {'cov_E':>6}  {'cov_EC':>6}  {'cov_clip':>8}"
    )
    for N in Ns:
        g = [r for r in rows if r["n_mon"] == N]
        print(
            f"  {N:>6}  {_rom_pool(g, 'crps_E', 'crps_EC'):>10.1f} "
            f"{_rom_boot(g, 'crps_E', 'crps_EC'):>6.1f}  "
            f"{_rom_pool(g, 'nll_E', 'nll_EC'):>9.1f}  "
            f"{_rom_pool(g, 'iscore_E', 'iscore_EC'):>8.1f}  "
            f"{_nm([r['cov_E'] for r in g]):>6.2f}  "
            f"{_nm([r['cov_EC'] for r in g]):>6.2f}  "
            f"{_nm([r['cov_clip'] for r in g]):>8.2f}"
        )

    _write_tex(rows, Ns, snaps, out_dir)
    _plot(rows, Ns, snaps, out_dir)
    _plot_calib(rows, Ns, out_dir)
    _verify(rows, Ns)


# -----------------------------------------------------------------------------
# Verification: assert the pooled trend, do not eyeball it.
# -----------------------------------------------------------------------------
def _snapshot_count(rows, Ns):
    """Distinct snapshots contributing at each density (composition can shrink as
    n_mon grows, since sparser-network snapshots drop out at high counts)."""
    return {N: len(set(r["snapshot"] for r in rows if r["n_mon"] == N)) for N in Ns}


def _common_range(rows, Ns):
    """Densities reached by ALL snapshots (constant-composition sub-range)."""
    cnt = _snapshot_count(rows, Ns)
    mx = max(cnt.values())
    return [N for N in Ns if cnt[N] == mx], mx


def _verify(rows, Ns) -> None:
    """Hard-assert from the pooled CSV rows, controlling for composition:

    (1) pooled tail-reduction exceeds the zero-clipping baseline at EVERY
        density -- the constrained GP dominates the naive fix;
    (2) over the COMMON range (densities all snapshots reach, so composition is
        fixed) the pooled tail-reduction at the sparsest exceeds the densest --
        the calibration gain GROWS as monitors thin, free of the snapshot-set
        change that the full-range endpoints would otherwise confound;
    (3) within each snapshot's OWN density range (also composition-free), the
        sparsest tail-reduction exceeds the densest, for a strict majority.
    """

    def tr(N):
        return _rom_pool(
            [r for r in rows if r["n_mon"] == N], "subzero_E", "subzero_EC"
        )

    def cl(N):
        return _rom_pool(
            [r for r in rows if r["n_mon"] == N], "subzero_E", "subzero_clip"
        )

    # (1) clip dominance at every density.
    for N in Ns:
        assert tr(N) > cl(N) + 1e-9, (
            f"VERIFY FAIL: at N={N} constrained tail reduction {tr(N):.1f}% does not "
            f"exceed clipping baseline {cl(N):.1f}%."
        )

    # (2) composition-controlled trend over the common range.
    common, mx = _common_range(rows, Ns)
    c_lo, c_hi = common[0], common[-1]
    print(
        f"\n[verify] common range (all {mx} snapshots, N in {common}): "
        f"tail_red {tr(c_hi):.1f}% (N={c_hi}) -> {tr(c_lo):.1f}% (N={c_lo}); "
        f"full-range endpoints N={Ns[0]}:{tr(Ns[0]):.1f}% vs N={Ns[-1]}:{tr(Ns[-1]):.1f}% "
        "(dense end mixes fewer snapshots)."
    )
    assert tr(c_lo) > tr(c_hi), (
        f"VERIFY FAIL: over the common range tail reduction does not grow as monitors "
        f"thin (N={c_lo}:{tr(c_lo):.1f}% <= N={c_hi}:{tr(c_hi):.1f}%)."
    )

    # (3) per-snapshot, composition-free.
    snaps = sorted(set(r["snapshot"] for r in rows))
    mono = 0
    for s in snaps:
        sNs = sorted(set(r["n_mon"] for r in rows if r["snapshot"] == s))
        lo = _rom_pool(
            [r for r in rows if r["snapshot"] == s and r["n_mon"] == sNs[0]],
            "subzero_E",
            "subzero_EC",
        )
        hi = _rom_pool(
            [r for r in rows if r["snapshot"] == s and r["n_mon"] == sNs[-1]],
            "subzero_E",
            "subzero_EC",
        )
        ok = np.isfinite(lo) and np.isfinite(hi) and lo > hi
        mono += int(ok)
        print(
            f"[verify]   {s:<28} N={sNs[0]}:{lo:5.1f}%  >  N={sNs[-1]}:{hi:5.1f}%  "
            f"{'ok' if ok else 'NO'}"
        )
    assert 2 * mono >= len(snaps), (
        f"VERIFY FAIL: only {mono}/{len(snaps)} snapshots show sparse>dense tail "
        "reduction within their own range."
    )
    print(
        f"[verify] OK: clip-dominated at every density; tail reduction grows as "
        f"monitors thin over the common range AND within {mono}/{len(snaps)} "
        "snapshots (composition-controlled)."
    )


# -----------------------------------------------------------------------------
# LaTeX macros: every paper-bound number read from the pooled CSV.
# -----------------------------------------------------------------------------
def _write_tex(rows, Ns, snaps, out_dir) -> None:
    n_lo, n_hi = Ns[0], Ns[-1]
    g_lo = [r for r in rows if r["n_mon"] == n_lo]
    g_hi = [r for r in rows if r["n_mon"] == n_hi]

    def pct(g, ku, kc):
        x = _rom_pool(g, ku, kc)
        x = 0.0 if (np.isfinite(x) and round(x) == 0) else x  # avoid "-0"
        return f"{x:.0f}" if np.isfinite(x) else "nan"

    # Composition-controlled anchors: common range (all snapshots) + per-snapshot.
    common, mx = _common_range(rows, Ns)
    c_hi = common[-1]
    g_chi = [r for r in rows if r["n_mon"] == c_hi]
    clip_max = max(
        _rom_pool([r for r in rows if r["n_mon"] == N], "subzero_E", "subzero_clip")
        for N in Ns
    )
    snaps_mono = 0
    for s in snaps:
        sNs = sorted(set(r["n_mon"] for r in rows if r["snapshot"] == s))
        lo = _rom_pool(
            [r for r in rows if r["snapshot"] == s and r["n_mon"] == sNs[0]],
            "subzero_E",
            "subzero_EC",
        )
        hi = _rom_pool(
            [r for r in rows if r["snapshot"] == s and r["n_mon"] == sNs[-1]],
            "subzero_E",
            "subzero_EC",
        )
        snaps_mono += int(np.isfinite(lo) and np.isfinite(hi) and lo > hi)

    full_snaps = [
        s
        for s in snaps
        if max(r["n_mon"] for r in rows if r["snapshot"] == s) == Ns[-1]
    ]

    macros = {
        "AqRealSnaps": f"{len(snaps)}",
        "AqRealSnapNoun": "snapshot" if len(snaps) == 1 else "snapshots",
        "AqRealSnapsFull": f"{len(full_snaps)}",  # span the full monitor range
        "AqRealFolds": f"{len(SEEDS)}",
        "AqRealCells": f"{len(rows)}",
        "AqRealMonDense": f"{n_hi}",
        "AqRealMonSparse": f"{n_lo}",
        "AqRealTailRedDense": pct(g_hi, "subzero_E", "subzero_EC"),
        "AqRealTailRedSparse": pct(g_lo, "subzero_E", "subzero_EC"),
        "AqRealTailRedDenseErr": f"{_rom_boot(g_hi, 'subzero_E', 'subzero_EC'):.0f}",
        "AqRealTailRedSparseErr": f"{_rom_boot(g_lo, 'subzero_E', 'subzero_EC'):.0f}",
        "AqRealClipDense": pct(g_hi, "subzero_E", "subzero_clip"),
        "AqRealClipSparse": pct(g_lo, "subzero_E", "subzero_clip"),
        "AqRealClipMax": f"{clip_max:.0f}",
        "AqRealHeldGainDense": pct(g_hi, "rmse_held_E", "rmse_held_EC"),
        "AqRealHeldGainSparse": pct(g_lo, "rmse_held_E", "rmse_held_EC"),
        "AqRealPCEDense": f"{_nm([r['p_ce_mc'] for r in g_hi]):.2f}",
        "AqRealPCESparse": f"{_nm([r['p_ce_mc'] for r in g_lo]):.2f}",
        # Composition-controlled common-range anchor (all snapshots reach c_hi).
        "AqRealMonCommon": f"{c_hi}",
        "AqRealTailCommon": pct(g_chi, "subzero_E", "subzero_EC"),
        "AqRealSnapsMono": f"{snaps_mono}",
        # Held-out proper scoring rules + calibration (pooled, +E+C over +E).
        "AqRealCrpsGainSparse": pct(g_lo, "crps_E", "crps_EC"),
        "AqRealCrpsGainDense": pct(g_hi, "crps_E", "crps_EC"),
        "AqRealCrpsGainAll": f"{_rom_pool(rows, 'crps_E', 'crps_EC'):.0f}",
        "AqRealNllGainAll": f"{_rom_pool(rows, 'nll_E', 'nll_EC'):.0f}",
        "AqRealIscGainAll": f"{_rom_pool(rows, 'iscore_E', 'iscore_EC'):.0f}",
        "AqRealCovE": f"{_nm([r['cov_E'] for r in rows]):.2f}",
        "AqRealCovEC": f"{_nm([r['cov_EC'] for r in rows]):.2f}",
        "AqRealCovClip": f"{_nm([r['cov_clip'] for r in rows]):.2f}",
    }
    path = os.path.join(out_dir, "aq_real_numbers.tex")
    with open(path, "w") as fh:
        fh.write(
            "% Auto-generated by sweep_real.py from "
            "air_quality_adv_diff_real_sweep.csv.\n"
            "% Pooled real-data numbers (ratio-of-means across snapshots x "
            "folds). Do not edit by hand.\n"
        )
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"Wrote pooled LaTeX macros -> {path}")


# -----------------------------------------------------------------------------
# Figures: editable text, restrained palette (blue =
# the proposed constrained method, neutrals for baselines/per-snapshot; green/red
# reserved for directional cues, unused here), panel a as the hero, one shared
# frameless legend strip, y-limits tightened to the data, SVG (primary) + PDF +
# TIFF@600 (no PNG).
# -----------------------------------------------------------------------------
# Palette: blue_main / blue_secondary / neutral_mid / neutral_light.
_PAL = {
    "method": "#0F4D92",
    "method2": "#3775BA",
    "baseline": "#767676",
    "faint": "#CFCECE",
    "ref": "#9A9A9A",
    "ink": "#272727",
}


def _apply_pub_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )


def _save_pub(fig, base) -> None:
    """SVG (primary, editable text) + PDF + TIFF@600 (LZW). No PNG."""
    import matplotlib.pyplot as plt

    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(
        f"{base}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"Saved: {base}.svg / .pdf / .tiff")


def _composition_divider(axes, rows, Ns, label_ax):
    """Dotted divider where the sparser networks drop out, with flanking net-count
    labels at the top of `label_ax` (dense side pools fewer snapshots)."""
    cnt = _snapshot_count(rows, Ns)
    mx = max(cnt.values())
    full = [N for N in Ns if cnt[N] == mx]
    reduced = [N for N in Ns if cnt[N] < mx]
    if not (full and reduced):
        return
    bnd = 0.5 * (min(reduced) + max(full))
    for ax in axes:
        ax.axvline(bnd, color=_PAL["ref"], ls=":", lw=0.7, zorder=0)
    label_ax.text(
        0.30,
        0.96,
        f"{cnt[max(Ns)]}/{mx} nets",
        transform=label_ax.transAxes,
        fontsize=5,
        color=_PAL["baseline"],
        ha="center",
        va="top",
    )
    label_ax.text(
        0.84,
        0.96,
        f"all {mx} nets",
        transform=label_ax.transAxes,
        fontsize=5,
        color=_PAL["baseline"],
        ha="center",
        va="top",
    )


def _plot(rows, Ns, snaps, out_dir) -> None:
    """Pooled sparsity figure. To avoid drawing a line across a change in the pooled
    snapshot set (which produces a spurious jump where the sparser networks drop
    out), each curve holds its composition FIXED: a bold 'all networks' curve over
    the common range (every snapshot present) and a 'PM2.5 networks' curve over the
    full monitor range (only the snapshots that span it). No line crosses the
    divider; the two agree on their overlap."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    _apply_pub_style()
    C_M, C_M2, C_B = _PAL["method"], _PAL["method2"], _PAL["baseline"]

    common, _mx = _common_range(rows, Ns)  # densities all snaps reach
    full_snaps = [
        s
        for s in snaps
        if max(r["n_mon"] for r in rows if r["snapshot"] == s) == Ns[-1]
    ]
    n_all, n_full = len(snaps), len(full_snaps)

    def pool(ns_list, snap_set, ku, kc):
        keep = set(snap_set)
        xs = np.array(ns_list, float)
        m = np.array(
            [
                _rom_pool(
                    [r for r in rows if r["snapshot"] in keep and r["n_mon"] == N],
                    ku,
                    kc,
                )
                for N in ns_list
            ]
        )
        s = np.array(
            [
                _rom_boot(
                    [r for r in rows if r["snapshot"] in keep and r["n_mon"] == N],
                    ku,
                    kc,
                )
                for N in ns_list
            ]
        )
        return xs, m, s

    def draw(ax, ku, kcE, kcC, ylabel, zero_line=False, show_clip=True):
        if zero_line:
            ax.axhline(0, color=_PAL["ref"], lw=0.6, zorder=1)
        vals = []
        # PM2.5 networks (n=2), full monitor range -- composition-constant.
        x2, m2, s2 = pool(Ns, full_snaps, ku, kcE)
        ax.fill_between(x2, m2 - s2, m2 + s2, color=C_M2, alpha=0.12, lw=0, zorder=2)
        ax.plot(
            x2,
            m2,
            "-o",
            color=C_M2,
            mfc="white",
            mec=C_M2,
            mew=0.9,
            ms=3.0,
            lw=1.1,
            zorder=3,
        )
        vals += [m2 - s2, m2 + s2]
        # Clipping baseline over the same full range. Plotted only where it is
        # informative (panel a, the sub-zero tail). For held-out RMSE (panel b)
        # clipping never moves a held-out prediction, so the curve is flat 0 and
        # would just double the zero reference line -- omit it there.
        if show_clip:
            _x, mc, _s = pool(Ns, full_snaps, ku, kcC)
            ax.plot(x2, mc, "--", color=C_B, lw=1.1, zorder=2)
            vals.append(mc)
        # All networks (n=4), common range -- the composition-controlled headline.
        x1, m1, s1 = pool(common, snaps, ku, kcE)
        ax.fill_between(x1, m1 - s1, m1 + s1, color=C_M, alpha=0.18, lw=0, zorder=4)
        ax.plot(
            x1,
            m1,
            "-s",
            color=C_M,
            mfc="white",
            mec=C_M,
            mew=1.0,
            ms=3.8,
            lw=1.5,
            zorder=5,
        )
        vals += [m1 - s1, m1 + s1]
        ax.set_xlabel(r"sampled monitors $n_{\mathrm{mon}}$")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()  # sparser to the right
        return np.concatenate(vals)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(5.8, 2.5), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )

    # Panel a (HERO): Gaussian moment-based sub-zero-tail proxy reduction.
    va = draw(axa, "subzero_E", "subzero_EC", "subzero_clip", r"$Q_0$ reduction [%]")
    axa.set_ylim(min(0.0, float(np.nanmin(va))) - 4, float(np.nanmax(va)) + 14)
    axa.set_title("a", loc="left", fontweight="bold", fontsize=9)

    # Panel b: held-out RMSE gain (CV-only, regime-dependent). No clip line here
    # (flat 0, redundant with the zero reference).
    vb = draw(
        axb,
        "rmse_held_E",
        "rmse_held_EC",
        "rmse_held_clip",
        "held-out RMSE gain [%]",
        zero_line=True,
        show_clip=False,
    )
    pad = 0.12 * (float(np.nanmax(vb)) - float(np.nanmin(vb)) + 1e-9)
    axb.set_ylim(float(np.nanmin(vb)) - pad, float(np.nanmax(vb)) + pad)
    axb.set_title("b", loc="left", fontweight="bold", fontsize=9)

    # Integer monitor-count ticks (multiples of 2) on both panels -- the narrower
    # panel b would otherwise auto-pick fractional ticks (22.5, 17.5, ...).
    for ax in (axa, axb):
        ax.xaxis.set_major_locator(MultipleLocator(2))

    # Dotted divider: densest count all networks reach; denser still = PM2.5 only.
    # Flank it with the pooled composition on each side so the (legend-less) line
    # is self-explanatory: left of it (denser) only the PM2.5 nets remain, right
    # of it (sparser) all networks contribute.
    cnt = _snapshot_count(rows, Ns)
    n_dense = cnt[max(Ns)]  # nets reaching the densest count
    n_all_nets = max(cnt.values())
    reduced = [N for N in Ns if cnt[N] < n_all_nets]
    if reduced and common:
        bnd = 0.5 * (min(reduced) + max(common))
        for ax in (axa, axb):
            ax.axvline(bnd, color=_PAL["ref"], ls=":", lw=0.7, zorder=0)
            ax.text(
                0.27,
                0.965,
                f"Maryland PM$_{{2.5}}$\n({n_dense} snapshots)",
                transform=ax.transAxes,
                fontsize=4.6,
                color=_PAL["baseline"],
                ha="center",
                va="top",
                linespacing=0.95,
            )
            ax.text(
                0.80,
                0.965,
                f"all {n_all_nets}\nsnapshots",
                transform=ax.transAxes,
                fontsize=4.6,
                color=_PAL["baseline"],
                ha="center",
                va="top",
                linespacing=0.95,
            )

    # Per-panel frameless legends: panel a carries all three entries (incl. the
    # clip baseline drawn there); panel b lists only the two method curves, since
    # the clip line is omitted there (flat 0).
    h_all = Line2D(
        [0],
        [0],
        color=C_M,
        marker="s",
        mfc="white",
        mec=C_M,
        mew=1.0,
        ms=3.8,
        lw=1.5,
        label=f"+E+C, all {n_all} snapshots",
    )
    h_pm = Line2D(
        [0],
        [0],
        color=C_M2,
        marker="o",
        mfc="white",
        mec=C_M2,
        mew=0.9,
        ms=3.0,
        lw=1.1,
        label=f"+E+C, PM$_{{2.5}}$ ({n_full} snapshots)",
    )
    h_clip = Line2D([0], [0], color=C_B, lw=1.1, ls="--", label="clip mean at 0")
    axa.legend(
        handles=[h_all, h_pm, h_clip],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.15),
        fontsize=5,
        frameon=False,
        handlelength=1.7,
        labelspacing=0.3,
    )
    axb.legend(
        handles=[h_all, h_pm],
        loc="lower left",
        fontsize=5,
        frameon=False,
        handlelength=1.7,
        labelspacing=0.3,
        borderaxespad=0.4,
    )

    fig.tight_layout(w_pad=1.6)
    # Set the final labels after layout to preserve the published panel positions.
    axa.set_ylabel(r"$Q_0$ reduction [%]")
    axb.set_ylabel("held-out RMSE reduction [%]")
    _save_pub(fig, os.path.join(out_dir, "figure_air_quality_adv_diff_real"))


def _plot_calib(rows, Ns, out_dir) -> None:
    """Calibration figure (SVG/PDF/TIFF), both panels vs
    monitor count and fully recoverable from the sweep CSV: (a) held-out CRPS
    ratio-of-means gain of +E+C over +E (and clipping) -- a PROPER score, so the
    constraint can lose here; (b) empirical 95% coverage per arm against the 0.95
    target -- every arm undercovers on real held-out stations (the smooth prior is
    overconfident), and the constraint trades a little coverage for sharper bands."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    _apply_pub_style()
    C_M, C_B, C_E = _PAL["method"], _PAL["baseline"], _PAL["faint"]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(5.8, 2.5), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )

    # Panel a (HERO): held-out CRPS gain (+E+C vs +E), bootstrap band + clipping.
    m = np.array(
        [
            _rom_pool([r for r in rows if r["n_mon"] == N], "crps_E", "crps_EC")
            for N in Ns
        ]
    )
    s = np.array(
        [
            _rom_boot([r for r in rows if r["n_mon"] == N], "crps_E", "crps_EC")
            for N in Ns
        ]
    )
    mc = np.array(
        [
            _rom_pool([r for r in rows if r["n_mon"] == N], "crps_E", "crps_clip")
            for N in Ns
        ]
    )
    axa.axhline(0, color=_PAL["ref"], lw=0.6, zorder=1)
    axa.fill_between(Ns, m - s, m + s, color=C_M, alpha=0.15, lw=0, zorder=2)
    axa.plot(
        Ns,
        m,
        "-s",
        color=C_M,
        mfc="white",
        mec=C_M,
        mew=0.9,
        ms=3.4,
        zorder=4,
        label="+E+C (constrained)",
    )
    axa.plot(
        Ns,
        mc,
        "--o",
        color=C_B,
        mfc="white",
        mec=C_B,
        mew=0.9,
        ms=3.0,
        zorder=3,
        label="clip $\\geq 0$ (baseline)",
    )
    axa.set_xlabel("number of monitors")
    axa.set_ylabel("held-out CRPS gain [%]")
    axa.set_title("a", loc="left", fontweight="bold", fontsize=9)
    axa.legend(loc="best", frameon=False, handlelength=1.8)
    axa.invert_xaxis()

    # Panel b: empirical 95% coverage per arm vs monitor count; 0.95 target dashed.
    def cov(arm):
        return np.array(
            [_nm([r[f"cov_{arm}"] for r in rows if r["n_mon"] == N]) for N in Ns]
        )

    axb.axhline(0.95, color=_PAL["ink"], ls=":", lw=0.8, zorder=1)
    axb.plot(
        Ns,
        cov("E"),
        "-^",
        color=C_E,
        mfc="white",
        mec=C_E,
        mew=0.9,
        ms=3.2,
        zorder=2,
        label="+E",
    )
    axb.plot(
        Ns,
        cov("clip"),
        "--o",
        color=C_B,
        mfc="white",
        mec=C_B,
        mew=0.9,
        ms=3.0,
        zorder=3,
        label="clip $\\geq 0$",
    )
    axb.plot(
        Ns,
        cov("EC"),
        "-s",
        color=C_M,
        mfc="white",
        mec=C_M,
        mew=0.9,
        ms=3.4,
        zorder=4,
        label="+E+C",
    )
    axb.set_xlabel("number of monitors")
    axb.set_ylabel("95% coverage")
    axb.set_ylim(0.0, 1.0)
    axb.set_title("b", loc="left", fontweight="bold", fontsize=9)
    axb.invert_xaxis()
    axb.annotate(
        "0.95 target",
        xy=(Ns[len(Ns) // 2], 0.95),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=5,
        color=_PAL["ink"],
        ha="center",
        va="bottom",
    )
    axb.legend(loc="lower left", frameon=False, handlelength=1.8, ncol=1)

    fig.tight_layout(w_pad=1.6)
    _save_pub(fig, os.path.join(out_dir, "figure_air_quality_adv_diff_real_calib"))


if __name__ == "__main__":
    main()
