"""
Publication figure for the noisy/sparse data-assimilation experiment.

Main message:
    Conditioning the PDE + noisy-scattered-data posterior on monotonicity lowers
    the reconstruction RMSE, and the improvement grows as the data degrade ---
    more observation noise or fewer observations --- so the shape constraint is
    most valuable exactly where the data are least informative.

Layout: two coupled trend panels.

Why paired RMSE curves rather than a single "relative gain" curve
-----------------------------------------------------------------
The per-realisation relative gain 1 - RMSE(con)/RMSE(unc) divides by a noisy
denominator and is therefore extremely high-variance (its ±1 std spans zero and
the curve is non-monotone). The *seed-averaged RMSE* of each posterior is, by
contrast, clean and monotone: the constrained curve sits below the unconstrained
one at every operating point, and the gap widens as the data degrade. We
therefore plot the two seed-averaged RMSE curves with ±1 std ribbons; the shaded
gap between them is the inequality's value, and we annotate the relative
reduction computed as a ratio of the seed-averaged RMSEs (a stable estimator).

That same ratio-of-means estimator is what the right-hand axis draws as a
curve. The objection above is to the *per-realisation* gain, whose noisy
denominator makes it unusable; the ratio of seed-averaged RMSEs is stable, so
the reader gets the trend in relative terms without a spurious ribbon. Left and
right axes therefore show the same quantity, absolute and relative.

Panels
------
  * Panel a: RMSE vs sigma_obs (n_obs fixed) -> gap widens with noise.
  * Panel b: RMSE vs n_obs (sigma_obs fixed) -> gap widens as data get sparse.
  * Right axis (both panels): the gap as a fraction of the baseline RMSE.
Both panels share a left y-axis, and a right y-axis on the same 0-40 % range, so
the two effects are directly comparable.

The gap annotation is a one-way arrow drawn from the baseline curve down to the
constrained curve: the glyph asserts a reduction rather than a neutral distance.

Input:
    poisson_2d_data_assimilation_sweep.csv  (sweep.py; SEEDS realisations).

Run from the repo root inside the container (after sweep.py):
    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/poisson_2d_data_assimilation/figure.py

Outputs (next to this script):
    figure_poisson_2d_data_assimilation.svg / .pdf / .tiff   (+ _preview.png for QA)
"""

from __future__ import annotations

from collections import defaultdict
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- Publication rcParams (editable text, journal sizing). ------------------
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "svg.fonttype": "none",  # editable text in SVG
        "pdf.fonttype": 42,  # editable TrueType text in PDF
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

# Palette: neutral baseline, blue signal, green = gain.
C_UNC = "#4D4D4D"  # neutral_dark   — PDE + noisy data (baseline)
C_CON = "#0F4D92"  # blue_main      — + monotonicity (ours)
C_GAIN = "#8BCF8B"  # green_3        — RMSE reduction (directional gain cue)
C_GAIN_L = "#2E6B2E"  # green_6        — same quantity as a relative curve

# Right-hand axis: the shaded gap read as a fraction of the baseline RMSE.
# Shared across panels so the two sweeps stay directly comparable, exactly as
# the left axis is.
GAIN_TOP = 40.0  # per cent, right-axis upper limit
HEADROOM = 1.30  # left-axis stretch that keeps the gain curve clear of
# the +/-1 std ribbons


def _load(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = dict(r)
        d["n_obs"] = int(float(r["n_obs"]))
        for k in ("sigma_obs", "rmse_unc", "rmse_con", "gain_rel"):
            try:
                d[k] = float(r[k])
            except (TypeError, ValueError):
                d[k] = float("nan")
        out.append(d)
    return out


def _stats(rows, keyfn, xs_filter):
    """Return (xs, mean_unc, std_unc, mean_con, std_con, gain_rom) for the cells
    selected by xs_filter, grouped by keyfn (the x value), seed-averaged.

    gain_rom is the ratio-of-means relative gain 1 - mean(rmse_con)/mean(rmse_unc)
    (a stable estimator; see the module docstring)."""
    g = defaultdict(list)
    for r in rows:
        if xs_filter(r):
            g[keyfn(r)].append(r)

    def m(v):
        v = [x for x in v if not math.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    def s(v):
        v = [x for x in v if not math.isnan(x)]
        return float(np.std(v)) if len(v) > 1 else 0.0

    xs = sorted(g)
    mu, su, mc, sc, gr = [], [], [], [], []
    for x in xs:
        ru = [r["rmse_unc"] for r in g[x]]
        rc = [r["rmse_con"] for r in g[x]]
        mu.append(m(ru))
        su.append(s(ru))
        mc.append(m(rc))
        sc.append(s(rc))
        gr.append(1.0 - m(rc) / m(ru) if m(ru) > 0 else float("nan"))
    return (
        np.array(xs, float),
        np.array(mu),
        np.array(su),
        np.array(mc),
        np.array(sc),
        np.array(gr),
    )


def _panel(
    ax,
    x,
    mu,
    su,
    mc,
    sc,
    gr,
    *,
    xlabel,
    fixed_label,
    annot_idx,
    label_lines=False,
    label_dx=0.0,
    label_dy=0.0,
    gain_axis=False,
    gain_ylabel=True,
    pad_right=0.0,
):
    # Shaded gain region between the two mean curves (green = RMSE reduction).
    ax.fill_between(
        x,
        mc,
        mu,
        where=(mu >= mc),
        color=C_GAIN,
        alpha=0.45,
        linewidth=0,
        zorder=1,
        label="RMSE reduction",
    )
    # +/-1 std ribbons.
    ax.fill_between(x, mu - su, mu + su, color=C_UNC, alpha=0.12, linewidth=0, zorder=0)
    ax.fill_between(x, mc - sc, mc + sc, color=C_CON, alpha=0.14, linewidth=0, zorder=0)
    # Mean curves.
    ax.plot(
        x,
        mu,
        "-o",
        color=C_UNC,
        mfc="white",
        mec=C_UNC,
        mew=0.9,
        ms=3.4,
        zorder=3,
        label="PDE + noisy data",
    )
    ax.plot(
        x,
        mc,
        "-s",
        color=C_CON,
        mfc="white",
        mec=C_CON,
        mew=0.9,
        ms=3.2,
        zorder=4,
        label="+ monotonicity",
    )

    # Annotate the ratio-of-means gain at the most-degraded operating point.
    # One-way arrow, baseline -> constrained, so the glyph itself reads as a
    # downward move in error rather than a neutral two-sided distance.
    i = annot_idx
    xa = x[i]
    ymid = 0.5 * (mu[i] + mc[i])
    ax.annotate(
        "",
        xy=(xa, mc[i]),
        xytext=(xa, mu[i]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_CON,
            lw=0.7,
            mutation_scale=6.5,
            shrinkA=1.0,
            shrinkB=1.0,
        ),
    )
    ax.text(
        xa + label_dx,
        ymid + label_dy,
        f"  {gr[i] * 100:.0f}%",
        color=C_CON,
        fontsize=6.2,
        va="center",
        ha="left",
    )

    ax.set_xlabel(xlabel)
    ax.text(
        0.03,
        0.97,
        fixed_label,
        transform=ax.transAxes,
        fontsize=6.2,
        va="top",
        ha="left",
        color="0.25",
    )
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.08)
    if pad_right:
        # extra room so the gain label is not squeezed against the right spine
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + pad_right * (x1 - x0))

    # Right-hand axis: the same reduction expressed as a fraction of the
    # baseline. This is the ratio-of-means estimator 1 - mean(con)/mean(unc),
    # not the high-variance per-realisation gain (see the module docstring), so
    # it is stable enough to draw as a curve.
    axr = None
    if gain_axis:
        axr = ax.twinx()
        axr.spines["right"].set_visible(True)
        axr.spines["right"].set_color(C_GAIN_L)
        axr.spines["top"].set_visible(False)
        axr.plot(
            x,
            100.0 * gr,
            ls=(0, (3.2, 1.6)),
            marker="^",
            ms=2.6,
            lw=0.9,
            color=C_GAIN_L,
            mfc="white",
            mec=C_GAIN_L,
            mew=0.7,
            zorder=5,
            label="relative reduction",
        )
        axr.set_ylim(0.0, GAIN_TOP)
        axr.set_yticks([0, 10, 20, 30, 40])
        axr.tick_params(axis="y", colors=C_GAIN_L, width=0.6, length=2.4)
        if gain_ylabel:
            axr.set_ylabel("RMSE reduction (%)", color=C_GAIN_L)
        axr.margins(x=0.08)

    if label_lines:
        handles, labels = ax.get_legend_handles_labels()
        if axr is not None:
            h2, l2 = axr.get_legend_handles_labels()
            handles += h2
            labels += l2
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.90 if gain_axis else 0.86),
            frameon=False,
            handlelength=1.4,
            labelspacing=0.3,
            borderaxespad=0.0,
        )
    return axr


def plot(rows, fig_base, layout="wide", gain_axis=True):
    """layout='wide'   : 1x2 panels, 5.6 in (double-column / figure*).
    layout='column' : 2x1 stacked panels, 3.3 in (single column),
    saved with a '_column' suffix. Font sizes are identical; only the
    panel arrangement changes, so the figure stays readable at 100%.

    gain_axis: draw the right-hand percentage-reduction axis."""
    # Panel a: vs sigma_obs at the modal fixed n_obs (the value most represented
    # among the sigma-axis rows).
    n_fixed = max(
        set(r["n_obs"] for r in rows), key=lambda n: sum(r["n_obs"] == n for r in rows)
    )
    xa, mua, sua, mca, sca, gra = _stats(
        rows, keyfn=lambda r: r["sigma_obs"], xs_filter=lambda r: r["n_obs"] == n_fixed
    )

    # Panel b: vs n_obs at the modal fixed sigma_obs (>0).
    sig_pool = [r["sigma_obs"] for r in rows if r["sigma_obs"] > 0]
    s_fixed = max(
        set(sig_pool), key=lambda s: sum(abs(r["sigma_obs"] - s) < 1e-12 for r in rows)
    )
    xb, mub, sub, mcb, scb, grb = _stats(
        rows,
        keyfn=lambda r: r["n_obs"],
        xs_filter=lambda r: abs(r["sigma_obs"] - s_fixed) < 1e-12,
    )

    if layout == "column":
        fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.3, 4.4))
        fig_base = f"{fig_base}_column"
    else:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.2 if gain_axis else 5.6, 2.5))

    # shared y-limit for honest cross-panel comparison; the extra headroom keeps
    # the right-axis gain curve out of the +/-1 std ribbons.
    ytop = 1.08 * max(np.nanmax(mua + sua), np.nanmax(mub + sub))
    if gain_axis:
        ytop *= HEADROOM

    _panel(
        axa,
        xa,
        mua,
        sua,
        mca,
        sca,
        gra,
        xlabel=r"observation noise $\sigma_{\mathrm{obs}}$",
        fixed_label=f"$n_{{\\mathrm{{obs}}}}={n_fixed}$",
        annot_idx=int(np.argmax(xa)),
        label_lines=True,
        gain_axis=gain_axis,
        pad_right=0.10 if gain_axis else 0.0,
    )
    axa.set_ylabel("RMSE vs. truth")
    axa.set_ylim(0.0, ytop)
    axa.set_title("a", loc="left", fontweight="bold", fontsize=9)

    _panel(
        axb,
        xb,
        mub,
        sub,
        mcb,
        scb,
        grb,
        xlabel=r"number of observations $n_{\mathrm{obs}}$",
        fixed_label=f"$\\sigma_{{\\mathrm{{obs}}}}={s_fixed:g}$",
        annot_idx=int(np.argmin(xb)),
        label_lines=False,
        # nudge the gain label up so it clears the "PDE + noisy data" curve at
        # the sparse (n_obs=6) point, while staying next to the arrow.
        label_dx=0.7,
        label_dy=0.026,
        gain_axis=gain_axis,
    )
    axb.set_ylim(0.0, ytop)
    axb.set_title("b", loc="left", fontweight="bold", fontsize=9)

    if layout == "column":
        axb.set_ylabel("RMSE vs. truth")
        fig.tight_layout(h_pad=1.6)
    else:
        fig.tight_layout(w_pad=2.6 if gain_axis else 1.4)

    fig.savefig(f"{fig_base}.svg", bbox_inches="tight")
    fig.savefig(f"{fig_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{fig_base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{fig_base}_preview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_base}.svg / .pdf / .tiff (+ _preview.png)")


def main(arg: str | None = None) -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    if arg and arg.endswith(".csv"):
        csv_path = arg
    else:
        if arg:
            out_dir = arg
        headline = os.path.join(out_dir, "poisson_2d_data_assimilation_sweep.csv")
        uniform = os.path.join(
            out_dir, "poisson_2d_data_assimilation_sweep_uniform.csv"
        )
        csv_path = headline if os.path.exists(headline) else uniform
    base = os.path.splitext(os.path.basename(csv_path))[0]
    name = base.replace(
        "poisson_2d_data_assimilation_sweep", "poisson_2d_data_assimilation"
    )
    fig_base = os.path.join(out_dir, f"figure_{name}")

    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Sweep CSV not found: {csv_path}\nRun sweep.py first (the default "
            "R-backed sweep gives rmse_unc/rmse_con; --quick only fills the "
            "unconstrained column)."
        )
    rows = _load(csv_path)
    if not any(not math.isnan(r["rmse_con"]) for r in rows):
        raise SystemExit(
            "The sweep CSV has no constrained RMSE (rmse_con all NaN); it was "
            "likely produced with --quick. Re-run sweep.py with the R backend."
        )
    plot(rows, fig_base, layout="wide")
    plot(rows, fig_base, layout="column")


if __name__ == "__main__":
    main(arg=sys.argv[1] if len(sys.argv) > 1 else None)
