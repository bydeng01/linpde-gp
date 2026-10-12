"""
Publication figure for the 3-way ablation on the free-boundary
2D Poisson problem: equality-only vs inequality-only vs both.

Main message:
    The PDE-equality posterior and the scale-free monotonicity constraint each
    fail on a complementary axis --- equality fixes the magnitude but is
    non-monotone at the free corner, the inequality fixes the shape but not the
    magnitude --- so only their combination is accurate, monotone, and tight.

Layout: field row -> mechanism slice -> metric bars.
Panels:
    a  truth u*            (+ setup: partial-BC edges, PDE collocation, virtual pts)
    b  equality only       (failure 1: reverts toward 0 at the free corner; non-monotone)
    c  inequality only     (failure 2: monotone but wrong magnitude; widest band)
    d  both (E, C)         (the fix)                       [a-d share one colour scale]
    e  diagonal slice u(s,s): truth vs the three arms      [the MECHANISM]
    f  metrics relative to equality-only: RMSE / band / free-corner error

Data: rendered from ``fields.npz`` + ``diagnostics.json`` (written by run.py),
so the figure re-styles without re-running the R solve. If absent, falls back to
``run.main()`` (needs the docker stack).

Run (after run.py):
    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/poisson_2d_ablation/figure.py

Outputs (next to this script):
    figure_ablation_3way.svg / .pdf / .tiff
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# ---- MANDATORY editable-text rules (must precede any figure) ----------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"
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
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

# Restrained palette: one neutral family (reference / deficient
# baselines), one signal family (the combined hero), green reserved for the
# directional gain cue only.
C_TRUTH = "#272727"  # near-black reference
C_EQ = "#767676"  # neutral mid — equality only (deficient baseline)
C_IN = "#3775BA"  # blue — inequality only (the other deficient baseline)
C_BOTH = "#B64342"  # signal red — both (the hero)
C_GAIN = "#2E9E44"  # green — directional gain cue only
FIELD_CMAP = "viridis"

MM = 1.0 / 25.4
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load():
    npz = os.path.join(_THIS_DIR, "fields.npz")
    js = os.path.join(_THIS_DIR, "diagnostics.json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        import sys

        if _THIS_DIR not in sys.path:
            sys.path.insert(0, _THIS_DIR)
        import run

        run.main()
    data = dict(np.load(npz, allow_pickle=True))
    with open(js) as fh:
        diag = json.load(fh)
    return data, diag


def _panel_label(ax, s, x=-0.02, y=1.06, color="black"):
    ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="right",
        va="bottom",
        color=color,
    )


def _field(ax, EX, EY, Z, title, vmin, vmax, ok=True):
    if not ok:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(
            0.5,
            0.5,
            f"{title}\nunavailable (no R)",
            ha="center",
            va="center",
            fontsize=6.5,
            color=C_EQ,
        )
        ax.set_title(title)
        return None
    im = ax.pcolormesh(
        EX,
        EY,
        Z,
        shading="gouraud",
        cmap=FIELD_CMAP,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    return im


def main():
    data, diag = _load()
    n = int(data["n_eval_axis"])
    EX, EY = data["EX"], data["EY"]
    Xv, X_pde, X_bc = data["Xv"], data["X_pde"], data["X_bc"]
    truth = data["truth"].reshape(n, n)
    eqm = data["eq_mean"].reshape(n, n)
    inm = data["ineq_mean"].reshape(n, n)
    bom = data["both_mean"].reshape(n, n)
    ineq_ok = bool(np.asarray(data["ineq_ok"]))
    both_ok = bool(np.asarray(data["both_ok"]))
    vmin, vmax = 0.0, float(max(1.0, truth.max()))

    d_eq, d_in, d_bo = diag["equality_only"], diag["inequality_only"], diag["both"]

    fig = plt.figure(figsize=(183 * MM, 116 * MM), layout="constrained")
    # Two independent sub-figures so the bottom row's y-axis labels cannot
    # distort the top row's column widths (which must stay evenly spaced).
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.0, 0.74])

    # ---------------- Row 1: truth + three arms (shared colour scale) --------
    gst = sf_top.add_gridspec(1, 4, wspace=0.12)
    ax_t = sf_top.add_subplot(gst[0, 0])
    ax_e = sf_top.add_subplot(gst[0, 1])
    ax_i = sf_top.add_subplot(gst[0, 2])
    ax_b = sf_top.add_subplot(gst[0, 3])

    im = _field(ax_t, EX, EY, truth, "truth  $u_\\star$", vmin, vmax)
    _field(ax_e, EX, EY, eqm, "equality only", vmin, vmax)
    _field(ax_i, EX, EY, inm, "inequality only", vmin, vmax, ok=ineq_ok)
    _field(ax_b, EX, EY, bom, "both  $(E,\\,C)$", vmin, vmax, ok=both_ok)
    ax_t.set_ylabel("y")
    for ax in (ax_e, ax_i, ax_b):
        ax.set_yticklabels([])

    # Setup overlay on the truth panel only (defines the visual vocabulary).
    # Kept light so the field stays readable underneath.
    ax_t.plot(
        X_bc[:, 0],
        X_bc[:, 1],
        "s",
        ms=4.0,
        mfc="white",
        mec="black",
        mew=0.5,
        label="Dirichlet data ($x{=}0,\\,y{=}0$)",
    )
    ax_t.plot(
        X_pde[:, 0],
        X_pde[:, 1],
        "+",
        ms=3.0,
        mec="#FF2D2D",
        mew=1.0,
        label="PDE collocation",
    )
    ax_t.plot(
        Xv[:, 0],
        Xv[:, 1],
        "o",
        ms=1.5,
        mfc="none",
        mec="#FFD700",
        mew=0.5,
        label="monotonicity virtual points",
    )
    leg = ax_t.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.30),
        fontsize=4.9,
        handletextpad=0.25,
        labelspacing=0.25,
        columnspacing=1.0,
        borderaxespad=0.0,
        ncol=3,
    )
    # Keep the wide 2-row legend from widening panel a's column (which would
    # blow open the a-b gap); it is positioned manually, not by the layout.
    leg.set_in_layout(False)

    # Virtual points on the constrained arms (where C acts). Same gold as the
    # panel-a legend marker, so the monotonicity points read consistently
    # across every panel that shows them.
    for ax, ok in ((ax_i, ineq_ok), (ax_b, both_ok)):
        if ok:
            ax.plot(
                Xv[:, 0],
                Xv[:, 1],
                "o",
                ms=1.5,
                mfc="none",
                mec="#FFD700",
                mew=0.5,
                alpha=0.85,
            )

    # Inline metric per reconstruction (self-quantifying panels).
    def _metric_tag(ax, text):
        ax.text(
            0.05,
            0.07,
            text,
            transform=ax.transAxes,
            fontsize=5.0,
            color="white",
            ha="left",
            va="bottom",
            linespacing=1.15,
        )

    _metric_tag(
        ax_e,
        f"RMSE {d_eq['l2']:.2f}\nmin derivative {min(d_eq['min_d0'], d_eq['min_d1']):+.2f}",
    )
    if ineq_ok:
        _metric_tag(
            ax_i,
            f"RMSE {d_in['l2']:.2f}\nband {d_in['band'] / d_eq['band']:.1f}$\\times$",
        )
    if both_ok:
        _metric_tag(
            ax_b, f"RMSE {d_bo['l2']:.2f}\ncorner {d_bo['free_corner_err']:.2f}"
        )

    # Failure / fix call-outs (direct annotation at the free corner (1,1)).
    ax_e.annotate(
        "reverts to 0\n(non-monotone)",
        xy=(0.97, 0.97),
        xytext=(0.24, 0.66),
        fontsize=5.2,
        color="white",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="white", lw=0.6),
    )
    if ineq_ok:
        ax_i.annotate(
            "monotone but\noff-scale",
            xy=(0.92, 0.92),
            xytext=(0.22, 0.64),
            fontsize=5.2,
            color="white",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="white", lw=0.6),
        )
    if both_ok:
        ax_b.annotate(
            "recovered",
            xy=(0.97, 0.97),
            xytext=(0.30, 0.74),
            fontsize=5.2,
            color="white",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="white", lw=0.6),
        )

    for ax, lab in ((ax_t, "a"), (ax_e, "b"), (ax_i, "c"), (ax_b, "d")):
        _panel_label(ax, lab)

    if im is not None:
        # shrink so the bar spans the square panels' height (not the taller
        # grid cell); aspect bumped to keep the same bar width after shrinking.
        cb = sf_top.colorbar(
            im,
            ax=[ax_t, ax_e, ax_i, ax_b],
            location="right",
            fraction=0.022,
            pad=0.01,
            aspect=15,
            shrink=0.50,
        )
        cb.set_label("$u$", rotation=0, labelpad=4)
        cb.outline.set_linewidth(0.5)

    # ---------------- Row 2a: diagonal slice (the mechanism) -----------------
    gsb = sf_bot.add_gridspec(1, 2, wspace=0.32)
    ax_s = sf_bot.add_subplot(gsb[0, 0])
    s = np.linspace(0.0, 1.0, n)
    d = np.arange(n)
    ax_s.plot(s, truth[d, d], color=C_TRUTH, lw=1.6, label="truth", zorder=5)
    ax_s.plot(s, eqm[d, d], color=C_EQ, lw=1.1, ls=(0, (4, 1.5)), label="equality only")
    if ineq_ok:
        ax_s.plot(
            s, inm[d, d], color=C_IN, lw=1.1, ls=(0, (1, 1)), label="inequality only"
        )
    if both_ok:
        ax_s.plot(s, bom[d, d], color=C_BOTH, lw=1.5, label="both")
    ax_s.axvspan(0.85, 1.0, color="#000000", alpha=0.05, lw=0)
    ax_s.set_xlabel("position along diagonal  $t$  (free corner at $t=1$)")
    ax_s.set_ylabel("$u(t,\\,t)$")
    ax_s.set_xlim(0, 1)
    ax_s.legend(loc="upper left", fontsize=5.8, handlelength=1.6, labelspacing=0.25)
    _panel_label(ax_s, "e", x=-0.10, y=1.02)

    # ---------------- Row 2b: metrics relative to equality-only --------------
    ax_m = sf_bot.add_subplot(gsb[0, 1])
    labels = ["RMSE", "band width\n$B_{1.96}$", "free-corner\nerror"]
    eq = np.array([d_eq["l2"], d_eq["band"], d_eq["free_corner_err"]])
    inq = np.array([d_in["l2"], d_in["band"], d_in["free_corner_err"]])
    bo = np.array([d_bo["l2"], d_bo["band"], d_bo["free_corner_err"]])
    rel_in, rel_bo = inq / eq, bo / eq

    x = np.arange(len(labels))
    w = 0.36
    ax_m.axhline(1.0, color=C_EQ, lw=0.9, ls="--", zorder=1)
    ax_m.text(
        2.48, 1.02, "equality-only", color=C_EQ, fontsize=5.2, ha="right", va="bottom"
    )
    b_in = ax_m.bar(
        x - w / 2,
        rel_in,
        w,
        color=C_IN,
        edgecolor="black",
        linewidth=0.4,
        label="inequality only",
        zorder=3,
    )
    b_bo = ax_m.bar(
        x + w / 2,
        rel_bo,
        w,
        color=C_BOTH,
        edgecolor="black",
        linewidth=0.4,
        label="both",
        zorder=3,
    )

    # Absolute values annotated on bars (no information lost by normalising).
    # inequality-only: centered above the bar.
    for r, av in zip(b_in, inq):
        ax_m.text(
            r.get_x() + r.get_width() / 2,
            r.get_height() + 0.04,
            f"{av:.2f}",
            ha="center",
            va="bottom",
            fontsize=4.9,
            color="black",
        )
    # both: offset to the upper-right so the green gain arrow (drawn at the bar
    # centre) does not cover the value.
    for r, av in zip(b_bo, bo):
        ax_m.text(
            r.get_x() + r.get_width() + 0.03,
            r.get_height() + 0.04,
            f"{av:.2f}",
            ha="left",
            va="bottom",
            fontsize=4.9,
            color="black",
        )

    # Directional gain cue (green) on "both" where it beats the baseline.
    for r in b_bo:
        ax_m.annotate(
            "",
            xy=(r.get_x() + r.get_width() / 2, r.get_height()),
            xytext=(r.get_x() + r.get_width() / 2, 1.0),
            arrowprops=dict(arrowstyle="-|>", color=C_GAIN, lw=0.8),
        )

    ax_m.set_xticks(x)
    ax_m.set_xticklabels(labels)
    ax_m.set_ylabel("relative to equality-only")
    ax_m.set_ylim(0, max(3.4, rel_in.max() * 1.12))
    ax_m.legend(loc="upper right", fontsize=5.8, handlelength=1.1)
    _panel_label(ax_m, "f", x=-0.12, y=1.02)

    # Figure title intentionally omitted; the LaTeX \caption carries the message.

    for ext in ("svg", "pdf", "tiff"):
        out = os.path.join(_THIS_DIR, f"figure_ablation_3way.{ext}")
        kw = {"bbox_inches": "tight"}
        if ext == "tiff":
            kw["dpi"] = 600
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}  # journals cap TIFF size
        fig.savefig(out, **kw)
        print(f"Saved {out}")
    # Also a quick raster for on-screen QA.
    fig.savefig(
        os.path.join(_THIS_DIR, "figure_ablation_3way.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
