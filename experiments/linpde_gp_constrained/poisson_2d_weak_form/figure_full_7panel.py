"""
Publication figure for the 2D Poisson *combination showcase* with incomplete
(partial) boundary information.

Seven-panel supplementary version of figure.py (panels a--g). It writes the
``_supp`` basename.

Main message:
    When boundary information is incomplete, the PDE-equality posterior is
    under-determined and non-monotone near the free corner; adding monotonicity
    inequality information restores an accurate, shape-correct solution and
    sharply lowers predictive uncertainty. Equality and inequality information
    are complementary --- neither alone suffices.

Layout (advisor revision, Aug 2026): the mechanism panel is referenced first in
the text, so it leads the figure at full width; the six maps then form a 2x3
grid in which each column pairs a posterior with the quantity derived from it,
and every colour bar sits immediately to the right of the panel it scales.

    a  diagonal slice u(s,s) with 1.96-sigma bands: truth vs equality-only vs
       combined --- the MECHANISM (non-monotone dip removed, band shrinks,
       truth inside the combined band)                         [full width]
    ---------------------------------------------------------------------
    b  equality only  u|E      (reverts toward 0 near the free corner (1,1))
    c  + inequality   u|(E,C)  (recovered)                    | shared u scale
    d  combined posterior 1.96-sigma half-width (where uncertainty remains),
       with its OWN sequential colour bar
    ---------------------------------------------------------------------
    e  signed error  u|E - u*          | shared diverging scale: the error
    f  signed error  u|(E,C) - u*      | collapse is visible directly
    g  truth u* with the problem setup overlaid (partial boundary data on
       x=0/y=0, PDE collocation, monotonicity virtual points)

Columns therefore read b/e (equality only: posterior above, its error below),
c/f (combined: posterior above, its error below) and d/g (what is left over:
residual uncertainty above, the truth it should bracket below).

Quantitative summary: table_metrics.tex (via make_table.py).

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
CMAP_ERR = "RdBu_r"
CMAP_STD = "magma"

# ---- Explicit inch-based geometry -------------------------------------------
# Panels are placed by hand rather than by GridSpec so that (i) every map cell
# is exactly square, (ii) each colour bar has the same height as the panel it
# belongs to and sits immediately to its right, and (iii) the wide slice panel
# aligns with the left and right edges of the map grid.
FIG_W = 7.0  # 1:1 with a two-column \textwidth, so nothing is scaled
L_MARGIN = 0.42  # y label + tick labels of column 1
R_MARGIN = 0.38  # tick labels + label of the right-hand colour bars
CB_W = 0.055  # colour-bar width
CB_PAD = 0.045  # panel -> its colour bar
COL_GAP = 0.76  # colour-bar labels + the next panel's y label
TITLE_H = 0.16  # clearance above a panel for its title
ROW_GAP = 0.36  # clearance below a panel for x label + tick labels
T_MARGIN = 0.08
B_MARGIN = 0.95  # bottom row baseline (leaves room for the setup legend)
SLICE_H = 1.25
SLICE_GAP = 0.42

S = (FIG_W - L_MARGIN - R_MARGIN - 2 * COL_GAP - 3 * (CB_PAD + CB_W)) / 3
PITCH = S + CB_PAD + CB_W + COL_GAP
COL_X = [L_MARGIN + i * PITCH for i in range(3)]
ROW_LO = B_MARGIN  # maps, second row
ROW_HI = ROW_LO + S + TITLE_H + ROW_GAP  # maps, first row
SLICE_Y = ROW_HI + S + TITLE_H + SLICE_GAP
SLICE_W = 2 * PITCH + S
FIG_H = SLICE_Y + SLICE_H + TITLE_H + T_MARGIN

HERE = os.path.dirname(os.path.abspath(__file__))


def _rect(x, y, w, h):
    """Inches (from the lower-left of the canvas) -> figure-fraction rect."""
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def _panel(fig, col, row_y):
    return fig.add_axes(_rect(COL_X[col], row_y, S, S))


def _cbar_axes(fig, col, row_y):
    return fig.add_axes(_rect(COL_X[col] + S + CB_PAD, row_y, CB_W, S))


def _plabel(ax, s, x=-0.22, y=1.03):
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


def _style_cbar(cb, label):
    """Compact bar: few ticks, trimmed labels, so neighbouring panels stay clear."""
    cb.set_label(label, fontsize=6.5, labelpad=2)
    cb.locator = mpl.ticker.MaxNLocator(nbins=3)
    cb.formatter = mpl.ticker.FuncFormatter(lambda v, _: f"{v:.3g}")
    cb.update_ticks()
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=1.8, pad=1.5)
    cb.outline.set_linewidth(0.5)


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
    err_u, err_c = unc - truth, con - truth
    emax = float(np.nanmax(np.abs(err_u)))  # shared diverging scale
    half95_c = 1.96 * con_sd  # 1.96-sigma half-width

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---- (a) 1.96-sigma diagonal slice: the mechanism, referenced first -----
    # The one panel whose evidence is intrinsically graphical (the shape of the
    # non-monotone equality dip, the continuous uncertainty bands, and the truth
    # sitting inside the combined band), so it leads at full width.
    ax_sl = fig.add_axes(_rect(L_MARGIN, SLICE_Y, SLICE_W, SLICE_H))
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
    _plabel(ax_sl, "a", x=-0.050)

    # ---- Row 1 of the grid: (b) equality only, (c) + inequality, (d) sigma --
    ax_e = _panel(fig, 0, ROW_HI)
    ax_c = _panel(fig, 1, ROW_HI)
    ax_sd = _panel(fig, 2, ROW_HI)

    for ax, Z, ttl in [
        (ax_e, unc, "equality-only mean"),
        (ax_c, con, "constrained mean"),
    ]:
        im_u = ax.pcolormesh(
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
    # shared sequential scale for (b) and (c); (g) repeats it below
    _style_cbar(fig.colorbar(im_u, cax=_cbar_axes(fig, 1, ROW_HI)), "$u$")

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

    # (d) residual uncertainty: its OWN colour bar, adjacent to the panel, so
    # the diverging error scale is never read against this map.
    ims = ax_sd.pcolormesh(
        EX, EY, half95_c, shading="auto", cmap=CMAP_STD, rasterized=True
    )
    ax_sd.set_title("combined $1.96s$ half-width", pad=3)
    _square(ax_sd)
    _style_cbar(fig.colorbar(ims, cax=_cbar_axes(fig, 2, ROW_HI)), "$1.96s$")

    # ---- Row 2 of the grid: (e) error, (f) error, (g) truth + setup ---------
    ax_de = _panel(fig, 0, ROW_LO)
    ax_dc = _panel(fig, 1, ROW_LO)
    ax_t = _panel(fig, 2, ROW_LO)

    for ax, Z, ttl in [
        (ax_de, err_u, "equality error $m-u_\\star$"),
        (ax_dc, err_c, "constrained error $m-u_\\star$"),
    ]:
        ime = ax.pcolormesh(
            EX,
            EY,
            Z,
            shading="auto",
            cmap=CMAP_ERR,
            vmin=-emax,
            vmax=emax,
            rasterized=True,
        )
        ax.set_title(ttl, pad=3)
        _square(ax)
    # the diverging bar now sits beside the error maps it scales
    _style_cbar(fig.colorbar(ime, cax=_cbar_axes(fig, 1, ROW_LO)), "error")

    im_t = ax_t.pcolormesh(
        EX,
        EY,
        truth,
        shading="auto",
        cmap=CMAP_FIELD,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax_t.set_title("truth $u_\\star$ and setup", pad=3)
    _square(ax_t)
    _style_cbar(fig.colorbar(im_t, cax=_cbar_axes(fig, 2, ROW_LO)), "$u$")

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
        bbox_to_anchor=(-0.02, -0.26),
        ncol=1,
        fontsize=5.4,
        handletextpad=0.3,
        labelspacing=0.45,
        borderaxespad=0.0,
    )

    for lbl, ax in zip("bcdefg", (ax_e, ax_c, ax_sd, ax_de, ax_dc, ax_t)):
        _plabel(ax, lbl)

    fig.savefig(f"{base}.svg")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=200)
    fig.savefig(f"{base}.tiff", dpi=600)
    plt.close(fig)
    print(f"Saved: {base}.svg / .pdf / .png / .tiff")


def main(out_dir=None):
    out_dir = out_dir or HERE
    fields, diag = load_data(out_dir)
    plot(fields, diag, os.path.join(out_dir, "figure_poisson_2d_partialBC_supp"))


if __name__ == "__main__":
    main(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
