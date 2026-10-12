"""
Real-data (SPX) headline figure: no-arbitrage value vs market liquidity.

Main message:
    On a real SPX option chain the no-arbitrage inequalities add nothing when
    the chain is densely quoted (the data already pin the surface), but recover
    a large held-out-RMSE reduction as the chain is thinned toward an illiquid
    market --- the model-free constraints are worth most exactly where the
    quotes are sparse.

Panels (shared y-axis):
  * a: paired seed-averaged held-out RMSE of quotes+E vs quotes+E+C against the
       retained-quote count n_obs (thinned SPX), with the dense-chain control
       marked at the right edge where the two arms coincide;
  * b: the imposed-arbitrage violation rate (calendar) of each arm vs n_obs,
       showing the constraints suppress arbitrage precisely in the sparse regime.

Palette and editable-text rules are copied from figure.py.

Inputs (sweep_spx.py): black_scholes_spx_sweep.csv. The dense control point is
read from bs_spx_numbers.tex if present, else annotated as "coincident".

Run from the repo root inside the container (after sweep_spx.py):
    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/black_scholes_surface/figure_spx.py

Outputs: figure_black_scholes_spx.svg / .pdf / .tiff (+ _preview.png),
and a _column variant at 3.3 in.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "lines.linewidth": 1.1,
    }
)

C_UNC = "#4D4D4D"  # neutral_dark   — quotes+E
C_CON = "#0F4D92"  # blue_main      — quotes+E+C
C_GAIN = "#8BCF8B"  # green_3        — RMSE reduction
C_DENSE = "#767676"  # neutral_mid    — dense control marker


def _load(csv_path):
    with open(csv_path, newline="") as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for r in raw:
        d = {"n_obs": int(float(r["n_obs"]))}
        for k in ("rmse_unc", "rmse_con", "viol_cal_unc", "viol_cal_con"):
            try:
                d[k] = float(r[k])
            except (TypeError, ValueError, KeyError):
                d[k] = float("nan")
        rows.append(d)
    return rows


def _read_dense(out_dir):
    """Pull the dense-control RMSEs from bs_spx_numbers.tex if present."""
    path = os.path.join(out_dir, "bs_spx_numbers.tex")
    if not os.path.exists(path):
        return None
    txt = open(path).read()

    def g(name):
        m = re.search(rf"\\newcommand{{\\{name}}}{{([^}}]*)}}", txt)
        return float(m.group(1)) if m else float("nan")

    return dict(
        n_obs=g("BsSpxDenseNobs"),
        rmse_unc=g("BsSpxDenseRmseUnc"),
        rmse_con=g("BsSpxDenseRmseCon"),
    )


def _stats(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r["n_obs"]].append(r[key])
    xs = sorted(g)
    m = [float(np.nanmean(g[x])) for x in xs]
    s = [float(np.nanstd(g[x])) if len(g[x]) > 1 else 0.0 for x in xs]
    return np.array(xs, float), np.array(m), np.array(s)


def _band(rows, key):
    """Per-n_obs mean and interquartile (25-75%) range across the seeds.

    The IQR is the honest spread to show here: the unconstrained held-out RMSE
    is heavy-tailed across thinned subsets (a few sparse draws miss the wings
    badly), so a symmetric std band would overstate the lower edge. The two
    arms' IQR bands overlap at most n_obs (the subsets are paired, so a wide
    marginal spread coexists with a consistent per-subset gain); whether a point
    is within noise is read off the paired per-subset gain (median, fraction
    improving), not from band overlap.
    """
    g = defaultdict(list)
    for r in rows:
        g[r["n_obs"]].append(r[key])
    xs = sorted(g)
    mean = np.array([float(np.nanmean(g[x])) for x in xs])
    p25 = np.array([float(np.nanpercentile(g[x], 25)) for x in xs])
    p75 = np.array([float(np.nanpercentile(g[x], 75)) for x in xs])
    return np.array(xs, float), mean, p25, p75


def _rmse_panel(ax, rows, dense):
    x, mu, u25, u75 = _band(rows, "rmse_unc")
    _, mc, c25, c75 = _band(rows, "rmse_con")
    # Green fill between the mean curves = the headline ratio-of-means reduction.
    ax.fill_between(
        x,
        mc,
        mu,
        where=(mu >= mc),
        color=C_GAIN,
        alpha=0.45,
        linewidth=0,
        zorder=1,
        label="RMSE reduction (mean)",
    )
    # Interquartile (25-75%) bands across the thinned subsets. These overlap at
    # most n_obs (paired subsets, heavy-tailed marginal spread), so band overlap
    # is not the within-noise test; that is read off the paired per-subset gain.
    ax.fill_between(x, u25, u75, color=C_UNC, alpha=0.15, linewidth=0, zorder=0)
    ax.fill_between(x, c25, c75, color=C_CON, alpha=0.16, linewidth=0, zorder=0)
    ax.plot(
        x,
        mu,
        "-o",
        color=C_UNC,
        mfc="white",
        mec=C_UNC,
        mew=0.9,
        ms=3.2,
        zorder=3,
        label="quotes+E",
    )
    ax.plot(
        x,
        mc,
        "-s",
        color=C_CON,
        mfc="white",
        mec=C_CON,
        mew=0.9,
        ms=3.0,
        zorder=4,
        label="quotes+E+C",
    )
    i = int(np.argmin(x))
    if np.isfinite(mc[i]) and mu[i] > 0:
        gain = 1.0 - mc[i] / mu[i]
        ax.annotate(
            "",
            xy=(x[i], mu[i]),
            xytext=(x[i], mc[i]),
            arrowprops=dict(arrowstyle="<->", color=C_CON, lw=0.7),
        )
        ax.text(
            x[i],
            0.5 * (mu[i] + mc[i]),
            f"  {gain * 100:.0f}%",
            color=C_CON,
            fontsize=6.2,
            va="center",
            ha="left",
        )
    if dense is not None and np.isfinite(dense["rmse_unc"]):
        # Dense control plotted at its TRUE retained-quote count (the full,
        # un-thinned chain) far to the right on the *same* log axis. A faint
        # dashed connector bridges the gap to the thinned sweep, signalling
        # that this is the same axis at a much higher density --- not an
        # adjacent sweep point. The two arms coincide here: with the chain
        # densely quoted the data already pin the surface, so the constraints
        # add nothing.
        xd = float(dense["n_obs"])
        y_last = 0.5 * (mu[-1] + mc[-1])
        y_dense = 0.5 * (dense["rmse_unc"] + dense["rmse_con"])
        ax.plot(
            [float(x.max()), xd],
            [y_last, y_dense],
            ls=(0, (2, 2)),
            color=C_DENSE,
            lw=0.7,
            alpha=0.55,
            zorder=2,
        )
        ax.plot(
            [xd], [dense["rmse_unc"]], "o", color=C_DENSE, ms=3.2, mfc=C_DENSE, zorder=5
        )
        ax.plot(
            [xd],
            [dense["rmse_con"]],
            "s",
            color=C_DENSE,
            ms=3.0,
            mfc="white",
            mec=C_DENSE,
            zorder=5,
        )
        ax.annotate(
            "dense chain\n(arms coincide)",
            xy=(xd, dense["rmse_unc"]),
            fontsize=5.6,
            ha="right",
            va="bottom",
            color=C_DENSE,
            xytext=(xd * 0.92, dense["rmse_unc"] + 0.06 * ax.get_ylim()[1]),
        )
    ax.set_xscale("log")
    if dense is not None and np.isfinite(dense["rmse_unc"]):
        ax.set_xlim(float(x.min()) * 0.8, float(dense["n_obs"]) * 1.7)
    ax.set_xlabel(r"retained quotes $n_{\mathrm{obs}}$ (thinned SPX)")
    ax.set_ylabel("held-out RMSE")
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc="lower left",
        frameon=False,
        handlelength=1.4,
        labelspacing=0.3,
        borderaxespad=0.2,
    )


def _viol_panel(ax, rows):
    x, mu, _ = _stats(rows, "viol_cal_unc")
    _, mc, _ = _stats(rows, "viol_cal_con")
    ax.plot(
        x,
        100 * mu,
        "-o",
        color=C_UNC,
        mfc="white",
        mec=C_UNC,
        mew=0.9,
        ms=3.2,
        label="quotes+E",
    )
    ax.plot(
        x,
        100 * mc,
        "-s",
        color=C_CON,
        mfc="white",
        mec=C_CON,
        mew=0.9,
        ms=3.0,
        label="quotes+E+C",
    )
    ax.set_xscale("log")
    # Over a narrow (~1.4 decade) range matplotlib labels the log minor ticks
    # (2x10^1, 3x10^1, ...), which collide into an unreadable blur. Keep the
    # minor tick marks but show decade labels only, matching panel a.
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel(r"retained quotes $n_{\mathrm{obs}}$")
    ax.set_ylabel("calendar violations (% cells)")
    ax.set_ylim(bottom=0.0)
    # Upper-left is the empty quadrant here (the quotes+E curve rises into the
    # upper right), so the legend no longer overlaps the data.
    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.4,
        labelspacing=0.3,
        borderaxespad=0.2,
    )


def plot(rows, dense, fig_base, layout="wide"):
    if layout == "column":
        fig, (a, b) = plt.subplots(2, 1, figsize=(3.3, 4.6))
        fig_base = f"{fig_base}_column"
    else:
        fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 2.5))
    _rmse_panel(a, rows, dense)
    _viol_panel(b, rows)
    for ax, lbl in zip((a, b), "ab"):
        ax.set_title(lbl, loc="left", fontweight="bold", fontsize=9)
    fig.tight_layout(w_pad=1.4 if layout == "wide" else 0, h_pad=1.6)
    fig.savefig(f"{fig_base}.svg", bbox_inches="tight")
    fig.savefig(f"{fig_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{fig_base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{fig_base}_preview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_base}.svg / .pdf / .tiff (+ _preview.png)")


def main(out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "black_scholes_spx_sweep.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"SPX sweep CSV not found: {csv_path}\nRun sweep_spx.py "
            "(R backend) first; --quick fills only the unconstrained column."
        )
    rows = _load(csv_path)
    if not any(not math.isnan(r["rmse_con"]) for r in rows):
        raise SystemExit(
            "SPX sweep CSV has no constrained RMSE (--quick run?). "
            "Re-run sweep_spx.py with the R backend."
        )
    dense = _read_dense(out_dir)
    fig_base = os.path.join(out_dir, "figure_black_scholes_spx")
    plot(rows, dense, fig_base, layout="wide")
    plot(rows, dense, fig_base, layout="column")


if __name__ == "__main__":
    main(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
