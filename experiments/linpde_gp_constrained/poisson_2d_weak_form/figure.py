"""
Publication figure for the 2D Poisson *combination showcase* with incomplete
(partial) boundary information.

Main message:
    When boundary information is incomplete, the PDE-equality posterior is
    under-determined and non-monotone near the free corner; adding monotonicity
    inequality information restores an accurate, shape-correct solution and
    sharply lowers predictive uncertainty. Equality and inequality information
    are complementary --- neither alone suffices.

Main-paper layout (4 panels). The signed-error maps, the residual-uncertainty
map, and the metrics now live in the supplement (figure_full_7panel.py) and the
in-text table (table_metrics.tex); only the load-bearing panels are kept here:
    a  truth u* with the problem setup overlaid (partial boundary data on
       x=0/y=0, PDE collocation, monotonicity virtual points)
    b  equality only  u|E      (reverts toward 0 near the free corner (1,1))
    c  + inequality   u|(E,C)  (recovered)
    d  diagonal slice u(s,s) with 1.96-sigma bands: truth vs equality-only vs
       combined --- the MECHANISM (non-monotone dip removed, band shrinks,
       truth inside the combined band)

Rendered from fields.npz + diagnostics.json (written by run.py), so the figure
can be re-styled WITHOUT re-running the slow R solve.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# ---- Editable-text rules -----------------------------------------------------
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "legend.frameon": False,
    }
)

C_TRUTH = "#272727"
C_EQ = "#767676"  # equality-only (deficient baseline)
C_CON = "#0F4D92"  # combined (E,C) posterior (hero)
C_GAIN = "#2E9E44"  # improvement cue only
C_BC = "#B64342"  # boundary data
C_XV = "#FFD700"  # virtual points
CMAP_FIELD = "viridis"

HERE = os.path.dirname(os.path.abspath(__file__))


def _plabel(ax, s, x=-0.20, y=1.03):
    ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _square(ax):
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def load_data(out_dir):
    npz_path = os.path.join(out_dir, "fields.npz")
    diag_path = os.path.join(out_dir, "diagnostics.json")
    if os.path.exists(npz_path) and os.path.exists(diag_path):
        d = np.load(npz_path, allow_pickle=True)
        with open(diag_path) as fh:
            diag = json.load(fh)
        return {k: d[k] for k in d.files}, diag
    sys.path.insert(0, HERE)
    import run as run_module  # noqa: E402

    data = run_module.solve()
    diag = data["diagnostics"]
    fields = dict(
        EX=data["EX"],
        EY=data["EY"],
        n_eval_axis=data["n_eval_axis"],
        truth=data["truth"],
        unc_mean=data["unc_mean"],
        unc_std=data["unc_std"],
        con_mean=data["con_mean"],
        con_std=data["con_std"],
        X_pde=data["X_pde"],
        X_bc=data["X_bc"],
        Xv=data["Xv"],
        constrain_axes=np.asarray(data["constrain_axes"]),
        constrained_ok=np.asarray(data["constrained_ok"]),
    )
    return fields, diag


def plot(fields, diag, base):
    n = int(fields["n_eval_axis"])
    EX, EY = fields["EX"], fields["EY"]
    a = np.linspace(0.0, 1.0, n)
    truth = fields["truth"].reshape(n, n)
    unc = fields["unc_mean"].reshape(n, n)
    con = fields["con_mean"].reshape(n, n)
    unc_sd = fields["unc_std"].reshape(n, n)
    con_sd = fields["con_std"].reshape(n, n)
    X_bc, X_pde, Xv = fields["X_bc"], fields["X_pde"], fields["Xv"]

    vmin, vmax = float(truth.min()), float(truth.max())

    fig = plt.figure(figsize=(7.0, 4.5))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.06],
        height_ratios=[1, 0.82],
        wspace=0.50,
        hspace=0.85,
        left=0.07,
        right=0.92,
        top=0.94,
        bottom=0.11,
    )

    # ---- Row 1: solution fields, shared sequential scale -------------------
    ax_t = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])
    for ax, Z, ttl in [
        (ax_t, truth, "truth $u_\\star$"),
        (ax_e, unc, "equality-only mean"),
        (ax_c, con, "constrained mean"),
    ]:
        im = ax.pcolormesh(
            EX,
            EY,
            Z,
            shading="auto",
            cmap=CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(ttl, pad=3)
        _square(ax)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$u$", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    # setup overlay on (a)
    ax_t.scatter(
        X_bc[:, 0], X_bc[:, 1], s=10, c=C_BC, marker="s", linewidths=0, zorder=5
    )
    ax_t.scatter(
        X_pde[:, 0],
        X_pde[:, 1],
        s=13,
        facecolors="none",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
    )
    ax_t.scatter(
        Xv[:, 0],
        Xv[:, 1],
        s=8,
        c=C_XV,
        marker="o",
        edgecolors="black",
        linewidths=0.2,
        zorder=6,
    )
    ax_e.annotate(
        "free corner",
        xy=(0.965, 0.965),
        xytext=(0.30, 0.60),
        fontsize=5.6,
        color=C_GAIN,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color=C_GAIN, lw=0.8),
    )
    for s, ax in zip("abc", (ax_t, ax_e, ax_c)):
        _plabel(ax, s)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=C_BC,
            markersize=4,
            label="boundary data ($x{=}0,\\,y{=}0$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="0.35",
            markersize=4,
            label="PDE collocation",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=C_XV,
            markeredgecolor="black",
            markersize=4,
            label="monotonicity virtual points",
        ),
    ]
    ax_t.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.27),
        ncol=3,
        fontsize=5.4,
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    # ---- Row 2: credible-band slice (mechanism), spanning the field columns -
    # The quantitative summary is now the in-text LaTeX table (table_metrics
    # .tex, via make_table.py) and the signed-error / residual-uncertainty maps
    # are in the supplement (figure_full_7panel.py); the slice is the one panel
    # whose evidence is intrinsically graphical (the non-monotone equality dip,
    # the continuous credible bands, and the truth inside the combined band).
    ax_sl = fig.add_subplot(gs[1, 0:3])
    s = a
    td, ud, cd = np.diag(truth), np.diag(unc), np.diag(con)
    usd, csd = np.diag(unc_sd), np.diag(con_sd)
    ax_sl.fill_between(
        s, ud - 1.96 * usd, ud + 1.96 * usd, color=C_EQ, alpha=0.18, linewidth=0
    )
    ax_sl.fill_between(
        s, cd - 1.96 * csd, cd + 1.96 * csd, color=C_CON, alpha=0.18, linewidth=0
    )
    ax_sl.plot(s, ud, color=C_EQ, lw=1.4, label="equality-only mean")
    ax_sl.plot(s, cd, color=C_CON, lw=1.8, label="constrained mean")
    ax_sl.plot(s, td, color=C_TRUTH, lw=1.2, ls=(0, (4, 2)), label="truth $u_\\star$")
    ax_sl.axhline(0.0, color="0.7", lw=0.5, zorder=0)
    ax_sl.set_title("diagonal slice $u(t,t)$ with $m\\pm1.96s$ bands", pad=4)
    ax_sl.set_xlabel("$t$ (along $x=y$)")
    ax_sl.set_ylabel("$u$")
    ax_sl.set_xlim(0, 1)
    ax_sl.margins(y=0.08)
    ax_sl.legend(
        loc="upper left",
        fontsize=6,
        handlelength=1.9,
        labelspacing=0.3,
        ncol=3,
        columnspacing=1.3,
        borderaxespad=0.4,
    )
    _plabel(ax_sl, "d", x=-0.062)

    fig.savefig(f"{base}.svg")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=200)
    fig.savefig(f"{base}.tiff", dpi=600)
    plt.close(fig)
    print(f"Saved: {base}.svg / .pdf / .png / .tiff")


def main(out_dir=None):
    out_dir = out_dir or HERE
    fields, diag = load_data(out_dir)
    plot(fields, diag, os.path.join(out_dir, "figure_poisson_2d_partialBC"))


if __name__ == "__main__":
    main(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
