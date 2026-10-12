"""
Publication figure for the arbitrage-constrained call-surface experiment.

Main message:
    Conditioning the quote + Dupire-PDE posterior on the no-arbitrage
    inequalities sharply reduces the arbitrage in the extrapolated surface
    (residual finite-difference violations persist in the wings, since the
    inequalities are imposed softly at finite virtual points) and cuts the
    held-out error, most where the quotes pin the surface least.

Panels
------
  * a: setup — truth, training quotes (bulk) vs held-out quotes (wings,
       long-maturity band), virtual boxes;
  * b-e: the 2x2 finite-difference constraint block. Column = arm
       (quotes+E, quotes+E+C), row = constraint family (call spread d_x c,
       calendar d_T c). The DERIVATIVE FIELD is plotted, not a violation
       mask: red is the infeasible side, the solid red line is the
       constraint boundary (the old hatch outline), and the dashed blue line
       in b, c is the lower leg of the two-sided spread bound -e^{x-rT}.
       Plotting the field keeps the DEPTH of each violation, which the old
       binary hatch discarded (advisor comment, 2026-08-04);
  * f: held-out RMSE per arm;
  * g: arbitrage-violation rates per arm (incl. the never-imposed butterfly);
  * h: sweep — seed-averaged held-out RMSE of quotes+E vs quotes+E+C against
       n_obs; the gap is the inequalities' value (ratio-of-means gain
       annotated; per-seed ratios are too noisy to plot, cf.
       poisson_2d_data_assimilation/figure.py).

NOTE (scope of panel g): run.py:violation_rates scores only the UPPER leg of
the two-sided spread bound (d_x c > 0) and does not score the price-bound
block at all, though both are imposed by run.py:build_constraints. The dashed
contour in b, c is therefore the only place the lower leg is visible.

Inputs (all produced by run.py / sweep.py; no R, no re-solve here):
    fields.npz, diagnostics.json, black_scholes_sweep.csv

Run from the repo root inside the container (after run.py + sweep.py):
    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/black_scholes_surface/figure.py

Outputs (next to this script):
    figure_black_scholes_surface.svg / .pdf / .tiff   (+ _preview.png)
    figure_black_scholes_surface_column.*             (3.3 in single column)
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np

# --- Publication rcParams (editable text; copied from the Poisson figures). --
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

# Palette. The four arms form ONE "information added" family
# rather than four competing hues: neutral for the data-only baselines, the
# blue signal family for the arms that carry the no-arbitrage inequalities,
# darkening as information accrues (quotes < quotes+E; quotes+C < quotes+E+C).
# Green and red are reserved strictly for directional cues (RMSE reduction,
# arbitrage violations).
C_UNC = "#4D4D4D"  # neutral_dark   — quotes+E (equality baseline)
C_CON = "#0F4D92"  # blue_main      — quotes+E+C (full method, hero)
C_GAIN = "#8BCF8B"  # green_3        — RMSE reduction (gain cue)
C_VIOL = "#B64342"  # red_strong     — arbitrage violations (drop cue)
C_Q = "#CFCECE"  # neutral_light  — quotes only
C_QC = "#3775BA"  # blue_secondary — quotes+C
C_LBOUND = "#1F4E9C"  # blue_deep      — second bound of a two-sided constraint

ARMS = ("quotes", "quotes+E", "quotes+C", "quotes+E+C")
ARM_COLORS = {"quotes": C_Q, "quotes+E": C_UNC, "quotes+C": C_QC, "quotes+E+C": C_CON}

VIOL_TOL = 1e-4
HEADROOM_G = 1.38  # ylim slack in panel h so the 2x2 arm key clears the bars
PROF_GAP = 0.14  # inch gap between the two stacked profiles of panel d


def _viol_mask(M: np.ndarray, EX: np.ndarray, ET: np.ndarray) -> np.ndarray:
    """Interior mask (n_x-2, n_T-2): call-spread OR calendar FD violation."""
    h_x = EX[1, 0] - EX[0, 0]
    h_T = ET[0, 1] - ET[0, 0]
    d_x = (M[2:, :] - M[:-2, :]) / (2.0 * h_x)
    d_T = (M[:, 2:] - M[:, :-2]) / (2.0 * h_T)
    return (d_x[:, 1:-1] > VIOL_TOL) | (d_T[1:-1, :] < -VIOL_TOL)


def _surface_panel(
    ax, EX, ET, M, title, vmin, vmax, overlay_viol=False, xlabel=True, ylabel=True
):
    im = ax.pcolormesh(
        EX, ET, M, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True
    )
    if overlay_viol:
        mask = _viol_mask(M, EX, ET)
        if mask.any():
            cs = ax.contourf(
                EX[1:-1, 1:-1],
                ET[1:-1, 1:-1],
                mask.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["////"],
            )
            cs.set_edgecolor(C_VIOL)
            cs.set_linewidth(0.0)
            ax.contour(
                EX[1:-1, 1:-1],
                ET[1:-1, 1:-1],
                mask.astype(float),
                levels=[0.5],
                colors=[C_VIOL],
                linewidths=0.7,
            )
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(r"$x=\ln(K/S_0)$")
    if ylabel:
        ax.set_ylabel(r"$T$ (yr)")
    return im


# --- Finite-difference constraint fields -------------------------------------
# The no-arbitrage conditions are inequalities on DERIVATIVES of the surface,
# so the quantity that shows a violation is the finite-difference derivative
# itself, not a binary mask of it. Plotting the derivative keeps the depth of
# the violation (how far outside the feasible set the mean sits), which a
# hatch discards: between quotes+E and quotes+E+C the violating AREA falls
# 52.7% -> 30.1% of interior cells, but the worst calendar violation falls
# -0.072 -> -0.028 and the worst spread violation +0.81 -> +0.48. The zero
# contour drawn on each panel is exactly the old hatch boundary.
#
# STENCIL CAVEAT: those depths are on the SHARED interior grid used here,
# (n_x-2, n_T-2), because d_x and d_T must be plotted on common axes.
# run.py:violation_rates keeps each derivative on its own fuller grid
# (d_x over all T, d_T over all x), so its worst_* are slightly larger:
# spread 0.83 -> 0.49 and calendar 0.079 -> 0.033 in diagnostics.json.
# Quote the diagnostics.json values in the paper, since panel g's rates come
# from the same source; the two agree except on the boundary rows.


def _fd_fields(M: np.ndarray, EX: np.ndarray, ET: np.ndarray):
    """Central-difference d_x c and d_T c on the shared interior grid.

    Returns (d_x, d_T, Xi, Ti), all (n_x-2, n_T-2). Same scheme and stencil
    as ``_viol_mask`` and as ``run.py:violation_rates``, so the sign of each
    field reproduces the reported violation rates cell for cell.
    """
    h_x = EX[1, 0] - EX[0, 0]
    h_T = ET[0, 1] - ET[0, 0]
    d_x = ((M[2:, :] - M[:-2, :]) / (2.0 * h_x))[:, 1:-1]
    d_T = ((M[:, 2:] - M[:, :-2]) / (2.0 * h_T))[1:-1, :]
    return d_x, d_T, EX[1:-1, 1:-1], ET[1:-1, 1:-1]


def _fd_panel(
    ax, Xi, Ti, F, vmin, vmax, cmap, title, lower_bound=None, xlabel=True, ylabel=True
):
    """One finite-difference constraint field, diverging about the bound.

    ``cmap`` is oriented so the INFEASIBLE side is red. The solid red line is
    the constraint boundary (F = 0); the dashed blue line, when given, is the
    second bound of a two-sided constraint (F = lower_bound).
    """
    norm = TwoSlopeNorm(vmin=min(vmin, -1e-12), vcenter=0.0, vmax=max(vmax, 1e-12))
    im = ax.pcolormesh(Xi, Ti, F, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    ax.contour(Xi, Ti, F, levels=[0.0], colors=[C_VIOL], linewidths=0.8)
    if lower_bound is not None:
        ax.contour(
            Xi,
            Ti,
            F - lower_bound,
            levels=[0.0],
            colors=[C_LBOUND],
            linewidths=0.8,
            linestyles="dashed",
        )
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(r"$x=\ln(K/S_0)$")
    if ylabel:
        ax.set_ylabel(r"$T$ (yr)")
    return im


def _fd_full(M: np.ndarray, EX: np.ndarray, ET: np.ndarray):
    """Each derivative on its OWN full grid, exactly as run.py scores them.

    ``_fd_fields`` trims both to a shared interior so d_x and d_T can be drawn
    on common axes; that trim lowers the extremes. The profile panel reports
    magnitudes on a readable axis, so it must match run.py:violation_rates
    (and therefore diagnostics.json) cell for cell instead.
    """
    h_x = EX[1, 0] - EX[0, 0]
    h_T = ET[0, 1] - ET[0, 0]
    d_x = (M[2:, :] - M[:-2, :]) / (2.0 * h_x)  # (n_x-2, n_T)
    d_T = (M[:, 2:] - M[:, :-2]) / (2.0 * h_T)  # (n_x, n_T-2)
    return d_x, EX[1:-1, 0], d_T, EX[:, 0]


def _profile_panel(ax_top, ax_bot, xs, dxE, xt, dTE, dxC, dTC, ec_ok):
    """Worst-over-T constraint margin against log-moneyness, both arms.

    Colour maps magnitude badly, so panels b, c, e, f cannot be subtracted by
    eye. These two profiles put the same information on a position scale: the
    zero crossing is the edge of the violating band (identical to the red
    contour on the maps) and the curve height is how deep the violation runs.
    The horizontal shift between the arms is the narrowing, the vertical gap
    is the shallowing.
    """

    def _arms(ax, x, fE, fC, bound_above):
        # bound_above: infeasible half lies ABOVE zero (call spread) or below.
        vals = [fE] + ([fC] if ec_ok else [])
        lo = min(float(v.min()) for v in vals)
        hi = max(float(v.max()) for v in vals)
        pad = 0.14 * (hi - lo)
        if bound_above:
            ax.axhspan(0.0, hi + pad, color=C_VIOL, alpha=0.10, lw=0)
        else:
            ax.axhspan(lo - pad, 0.0, color=C_VIOL, alpha=0.10, lw=0)
        ax.axhline(0.0, color=C_VIOL, lw=0.8)
        ax.plot(x, fE, color=C_UNC, label="q+E")
        if ec_ok:
            ax.plot(x, fC, color=C_CON, label="q+E+C")
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.tick_params(labelsize=5.6)

    # Call spread: constraint d_x c <= 0, so the worst cell is the max over T.
    _arms(ax_top, xs, dxE.max(1), dxC.max(1) if ec_ok else None, True)
    ax_top.set_ylabel(r"$\max_T\,\partial_x m$", fontsize=6.2)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(
        frameon=False,
        handlelength=1.0,
        labelspacing=0.2,
        borderaxespad=0.1,
        loc="upper left",
        fontsize=5.8,
    )
    # Calendar: constraint d_T c >= 0, so the worst cell is the min over T.
    _arms(ax_bot, xt, dTE.min(1), dTC.min(1) if ec_ok else None, False)
    ax_bot.set_ylabel(r"$\min_T\,\partial_T m$", fontsize=6.2)
    ax_bot.set_xlabel(r"$x=\ln(K/S_0)$")


def _stats(rows, key_x, x_filter):
    g = defaultdict(list)
    for r in rows:
        if x_filter(r):
            g[r[key_x]].append(r)

    def m(v):
        v = [x for x in v if not math.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    def s(v):
        v = [x for x in v if not math.isnan(x)]
        return float(np.std(v)) if len(v) > 1 else 0.0

    xs = sorted(g)
    mu, su, mc, sc = [], [], [], []
    for x in xs:
        mu.append(m([r["rmse_unc"] for r in g[x]]))
        su.append(s([r["rmse_unc"] for r in g[x]]))
        mc.append(m([r["rmse_con"] for r in g[x]]))
        sc.append(s([r["rmse_con"] for r in g[x]]))
    return (np.array(xs, float), np.array(mu), np.array(su), np.array(mc), np.array(sc))


def _sweep_panel(ax, rows):
    sig_fixed = max(
        set(r["sigma_obs"] for r in rows),
        key=lambda s: sum(abs(r["sigma_obs"] - s) < 1e-12 for r in rows),
    )
    x, mu, su, mc, sc = _stats(
        rows, "n_obs", lambda r: abs(r["sigma_obs"] - sig_fixed) < 1e-12
    )
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
    ax.fill_between(x, mu - su, mu + su, color=C_UNC, alpha=0.12, linewidth=0)
    ax.fill_between(x, mc - sc, mc + sc, color=C_CON, alpha=0.14, linewidth=0)
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
    gain = 1.0 - mc[i] / mu[i] if mu[i] > 0 else float("nan")
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
    ax.set_xlabel(r"number of quotes $n_{\mathrm{obs}}$")
    ax.set_ylabel("held-out RMSE")
    ax.text(
        0.97,
        0.97,
        f"$\\sigma_{{\\mathrm{{obs}}}}={sig_fixed:g}$",
        transform=ax.transAxes,
        fontsize=6.2,
        va="top",
        ha="right",
        color="0.25",
    )
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.08)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.88),
        frameon=False,
        handlelength=1.4,
        labelspacing=0.3,
        borderaxespad=0.0,
    )


def _bars_panel(ax, diag, key_by_arm, ylabel, title=None, scale=1.0):
    arms = [a for a in ARMS if diag["arms"][a].get(key_by_arm) is not None]
    vals = [scale * diag["arms"][a][key_by_arm] for a in arms]
    xpos = np.arange(len(arms))
    ax.bar(xpos, vals, 0.62, color=[ARM_COLORS[a] for a in arms])
    ax.set_xticks(xpos)
    ax.set_xticklabels([a.replace("quotes", "q") for a in arms], fontsize=6)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def _viol_bars_panel(ax, diag, legend=True):
    """Grouped violation rates. ``legend=False`` in the wide layout: the arm
    key is carried once, in the shared legend cell above, because an in-axes
    legend lands on the butterfly group (the tallest bars on the right)."""
    kinds = [
        ("viol_spread_rate", "spread"),
        ("viol_cal_rate", "calendar"),
        ("viol_bfly_rate", "butterfly"),
    ]
    arms = [a for a in ARMS if diag["arms"][a].get("viol_spread_rate") is not None]
    w = 0.8 / len(arms)
    for j, a in enumerate(arms):
        vals = [100.0 * diag["arms"][a][k] for k, _ in kinds]
        ax.bar(
            np.arange(3) + (j - (len(arms) - 1) / 2) * w,
            vals,
            w,
            color=ARM_COLORS[a],
            label=a.replace("quotes", "q"),
        )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(
        [lbl for _, lbl in kinds],
        fontsize=5.8,
        rotation=12,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("violating grid cells (%)")
    if legend:
        # 2x2 key inside the panel. The tallest bar is spread/q on the far
        # left, so headroom plus an upper-right anchor clears every group;
        # matplotlib fills ncol=2 column-wise, giving q / q+E | q+C / q+E+C.
        top = max(100.0 * diag["arms"][a][k] for a in arms for k, _ in kinds)
        ax.set_ylim(0.0, HEADROOM_G * top)
        ax.legend(
            frameon=False,
            handlelength=0.9,
            handleheight=0.9,
            labelspacing=0.28,
            ncol=2,
            columnspacing=0.9,
            borderaxespad=0.2,
            loc="upper right",
        )


# --- Inch-based layout constants (wide / figure* variant) --------------------
# Panels are placed with fig.add_axes from these constants rather than by
# GridSpec, so each colour bar matches the height of the panel it scales and
# the 2x2 finite-difference block stacks exactly (column = arm, row =
# constraint). Change the constants, not the rects. FIG_W/FIG_H are derived.
MAP_W, MAP_H = 1.80, 1.35  # constraint / surface map cell
CB_W, CB_PAD = 0.095, 0.055  # colour bar width and its gap from the panel
L_MARG, R_MARG = 0.50, 0.10
GAP_CB = CB_W + CB_PAD + 0.55  # column gap that must hold a bar + a ylabel
GAP_PLAIN = 0.22  # gap between two maps sharing one scale
TOP_MARG, TITLE_H = 0.24, 0.20
ROW_GAP, BOT_ROW_GAP = 0.26, 0.62
PLOT_H, BOT_MARG = 1.15, 0.42
PLOT_GAP = 0.62  # gap between the three bottom-row plots

_X1 = L_MARG
_X2 = _X1 + MAP_W + GAP_CB
_X3 = _X2 + MAP_W + GAP_PLAIN
FIG_W = _X3 + MAP_W + CB_PAD + CB_W + R_MARG
FIG_H = (
    TOP_MARG
    + TITLE_H
    + MAP_H
    + ROW_GAP
    + TITLE_H
    + MAP_H
    + BOT_ROW_GAP
    + PLOT_H
    + BOT_MARG
)
_PLOT_W = (FIG_W - L_MARG - R_MARG - 2 * PLOT_GAP) / 3.0
_ROW1_Y = FIG_H - TOP_MARG - TITLE_H - MAP_H
_ROW2_Y = _ROW1_Y - ROW_GAP - TITLE_H - MAP_H
_ROW3_Y = _ROW2_Y - BOT_ROW_GAP - PLOT_H


def _ax(fig, x, y, w, h):
    """add_axes from inches."""
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def _cbar(fig, im, x, y, label, nticks=5):
    cax = _ax(fig, x, y, CB_W, MAP_H)
    cb = fig.colorbar(im, cax=cax)
    # Unit sits above the bar, not beside it: at 0.095 in wide there is no
    # room for a rotated label outside the tick labels.
    cb.ax.set_title(label, fontsize=6.4, pad=3.0)
    cb.ax.tick_params(labelsize=5.6, width=0.5, length=1.8, pad=1.2)
    cb.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nticks))
    cb.outline.set_linewidth(0.5)
    return cb


def plot(fields, diag, rows, fig_base, layout="wide"):
    """layout='wide'  : 8 panels, ~7.0 in (double-column / figure*), placed
    with inch-based add_axes; the 2x2 finite-difference block sits in columns
    2-3 (column = arm, row = constraint family).
    layout='column': 4x2 panels, 3.3 in (single column), '_column'
    suffix. Font sizes identical; only the arrangement changes."""
    EX, ET = fields["EX"], fields["ET"]
    nx, nT = int(fields["n_eval_x"]), int(fields["n_eval_T"])
    truth = fields["truth"].reshape(nx, nT)
    mE = fields["quotes_E_mean"].reshape(nx, nT)
    mEC = fields["quotes_E_C_mean"].reshape(nx, nT)
    ec_ok = bool(fields["quotes_E_C_ok"])
    vmin, vmax = float(np.nanmin(truth)), float(np.nanmax(truth))

    # Finite-difference constraint fields. The lower leg of the two-sided
    # call-spread bound, -e^{x-rT} <= d_x c, is drawn as a dashed contour: it
    # is imposed at the virtual points (run.py:build_constraints) and is what
    # stops the one-sided blocks biasing the ITM wing upward.
    r_rate = float(diag["params"].get("r", 0.02))
    dxE, dTE, Xi, Ti = _fd_fields(mE, EX, ET)
    lb_dx = -np.exp(Xi - r_rate * Ti)
    if ec_ok:
        dxC, dTC, _, _ = _fd_fields(mEC, EX, ET)
    else:
        dxC = dTC = np.full_like(dxE, np.nan)
    dx_lo = float(np.nanmin([dxE.min(), dxC.min()]))
    dx_hi = float(np.nanmax([dxE.max(), dxC.max()]))
    dT_lo = float(np.nanmin([dTE.min(), dTC.min()]))
    dT_hi = float(np.nanmax([dTE.max(), dTC.max()]))

    if layout == "column":
        fig, axes = plt.subplots(5, 2, figsize=(3.3, 10.6))
        ax_a, ax_b = axes[0]
        ax_c, ax_dtop = axes[1]
        ax_dbot = ax_dtop.inset_axes([0.0, -0.62, 1.0, 0.46])
        ax_e, ax_f = axes[2]
        ax_g, ax_h = axes[3]
        ax_i, ax_spare = axes[4]
        ax_spare.axis("off")
        fig_base = f"{fig_base}_column"
    else:
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        ax_a = _ax(fig, _X1, _ROW1_Y, MAP_W, MAP_H)
        ax_b = _ax(fig, _X2, _ROW1_Y, MAP_W, MAP_H)
        ax_c = _ax(fig, _X3, _ROW1_Y, MAP_W, MAP_H)
        # (d) is a two-row cell occupying one map footprint: the call-spread
        # profile above the calendar profile, in the same top-to-bottom order
        # as the map rows (b, c above; e, f alongside).
        _sub_h = (MAP_H - PROF_GAP) / 2.0
        ax_dtop = _ax(fig, _X1, _ROW2_Y + _sub_h + PROF_GAP, MAP_W, _sub_h)
        ax_dbot = _ax(fig, _X1, _ROW2_Y, MAP_W, _sub_h)
        ax_e = _ax(fig, _X2, _ROW2_Y, MAP_W, MAP_H)
        ax_f = _ax(fig, _X3, _ROW2_Y, MAP_W, MAP_H)
        ax_g = _ax(fig, L_MARG, _ROW3_Y, _PLOT_W, PLOT_H)
        ax_h = _ax(fig, L_MARG + _PLOT_W + PLOT_GAP, _ROW3_Y, _PLOT_W, PLOT_H)
        ax_i = _ax(fig, L_MARG + 2 * (_PLOT_W + PLOT_GAP), _ROW3_Y, _PLOT_W, PLOT_H)

    wide = layout != "column"
    # In the 3.3 in column variant the bold panel letter and a long centred
    # title collide, so the arm prefix is dropped there.
    arm_E = "quotes+E" if wide else "+E"
    arm_EC = "quotes+E+C" if wide else "+E+C"

    # a: truth + setup.
    im_a = _surface_panel(ax_a, EX, ET, truth, "truth", vmin, vmax)
    ax_a.scatter(
        fields["X_train"][:, 0],
        fields["X_train"][:, 1],
        s=7,
        c="white",
        edgecolors="k",
        linewidths=0.4,
        zorder=5,
        label="train",
    )
    ax_a.scatter(
        fields["X_test"][:, 0],
        fields["X_test"][:, 1],
        s=7,
        marker="^",
        c=C_VIOL,
        edgecolors="k",
        linewidths=0.3,
        zorder=5,
        label="held out",
    )
    ax_a.scatter(
        fields["Xv"][:, 0],
        fields["Xv"][:, 1],
        s=3,
        c="k",
        alpha=0.5,
        zorder=4,
        label="virtual",
    )
    if wide:
        _cbar(fig, im_a, _X1 + MAP_W + CB_PAD, _ROW1_Y, r"$c$")
    # The setup key lives inside (a). No corner of this panel is empty (every
    # quadrant carries 8-10 markers), so the frame is semi-transparent white:
    # it keeps the text legible over the bright yellow/green corner without
    # hiding the markers underneath.
    leg_a = ax_a.legend(
        loc="upper left",
        frameon=True,
        handlelength=0.8,
        labelspacing=0.2,
        borderaxespad=0.3,
    )
    leg_a.get_frame().set_facecolor("white")
    leg_a.get_frame().set_edgecolor("none")
    leg_a.get_frame().set_alpha(0.65)

    # d: worst-over-T margin profiles, the position-scale companion to b, c,
    # e, f (colour cannot be subtracted by eye).
    dxEf, xs, dTEf, xt = _fd_full(mE, EX, ET)
    if ec_ok:
        dxCf, _, dTCf, _ = _fd_full(mEC, EX, ET)
    else:
        dxCf = dTCf = None
    _profile_panel(ax_dtop, ax_dbot, xs, dxEf, xt, dTEf, dxCf, dTCf, ec_ok)

    # b, c: call-spread derivative. Feasible band -e^{x-rT} <= d_x c <= 0, so
    # positive (red) is the upper-leg violation the reported rate counts.
    im_b = _fd_panel(
        ax_b,
        Xi,
        Ti,
        dxE,
        dx_lo,
        dx_hi,
        "RdBu_r",
        f"{arm_E}:  $\\partial_x m$",
        lower_bound=lb_dx,
        xlabel=not wide,
        ylabel=not wide,
    )
    if ec_ok:
        _fd_panel(
            ax_c,
            Xi,
            Ti,
            dxC,
            dx_lo,
            dx_hi,
            "RdBu_r",
            f"{arm_EC}:  $\\partial_x m$",
            lower_bound=lb_dx,
            xlabel=not wide,
            ylabel=not wide,
        )
    else:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "quotes+E+C\nunavailable (no R)", ha="center", va="center")
    if wide:
        _cbar(fig, im_b, _X3 + MAP_W + CB_PAD, _ROW1_Y, r"$\partial_x m$")

    # e, f: calendar derivative. Feasible d_T c >= 0, so negative (red) is the
    # calendar-arbitrage violation.
    im_e = _fd_panel(
        ax_e,
        Xi,
        Ti,
        dTE,
        dT_lo,
        dT_hi,
        "RdBu",
        f"{arm_E}:  $\\partial_T m$",
        xlabel=True,
        ylabel=not wide,
    )
    if ec_ok:
        _fd_panel(
            ax_f,
            Xi,
            Ti,
            dTC,
            dT_lo,
            dT_hi,
            "RdBu",
            f"{arm_EC}:  $\\partial_T m$",
            xlabel=True,
            ylabel=not wide,
        )
    else:
        ax_f.axis("off")
    if wide:
        _cbar(fig, im_e, _X3 + MAP_W + CB_PAD, _ROW2_Y, r"$\partial_T m$")
        # The 2x2 block shares one x axis per column and one y axis per row:
        # tick labels are drawn once, on the outer edges of the block.
        for ax in (ax_b, ax_c):
            ax.tick_params(labelbottom=False)
        for ax in (ax_c, ax_f):
            ax.tick_params(labelleft=False)
    else:
        # Column variant: no inch-based rects, so each scale gets an attached
        # bar on the last panel that uses it.
        for im_, ax_, lbl_ in (
            (im_a, ax_a, r"$c$"),
            (im_b, ax_c if ec_ok else ax_b, r"$\partial_x m$"),
            (im_e, ax_f if ec_ok else ax_e, r"$\partial_T m$"),
        ):
            cb = fig.colorbar(im_, ax=ax_, fraction=0.046, pad=0.03)
            cb.set_label(lbl_, fontsize=6.2)
            cb.ax.tick_params(labelsize=5.6)

    # g: held-out RMSE per arm.
    _bars_panel(ax_g, diag, "rmse_test", "held-out RMSE")

    # h: violation rates per arm.
    _viol_bars_panel(ax_h, diag)

    # i: sweep.
    if rows:
        _sweep_panel(ax_i, rows)
    else:
        ax_i.axis("off")
        ax_i.text(
            0.5, 0.5, "sweep CSV missing\n(run sweep.py)", ha="center", va="center"
        )

    for ax, lbl in zip(
        (ax_a, ax_b, ax_c, ax_dtop, ax_e, ax_f, ax_g, ax_h, ax_i), "abcdefghi"
    ):
        ax.set_title(lbl, loc="left", fontweight="bold", fontsize=9)

    if layout == "column":
        fig.tight_layout(h_pad=1.2, w_pad=1.0)

    fig.savefig(f"{fig_base}.svg", bbox_inches="tight")
    fig.savefig(f"{fig_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{fig_base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{fig_base}_preview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_base}.svg / .pdf / .tiff (+ _preview.png)")


def _load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for r in raw:
        d = dict(r)
        d["n_obs"] = int(float(r["n_obs"]))
        for k in ("sigma_obs", "rmse_unc", "rmse_con"):
            try:
                d[k] = float(r[k])
            except (TypeError, ValueError):
                d[k] = float("nan")
        rows.append(d)
    if not any(not math.isnan(r["rmse_con"]) for r in rows):
        raise SystemExit(
            "Sweep CSV has no constrained RMSE (produced with --quick?). "
            "Re-run sweep.py with the R backend."
        )
    return rows


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    fields = np.load(os.path.join(out_dir, "fields.npz"))
    # Guard: this is the *synthetic* surface figure; panel a and the shared
    # colour scale need a ground-truth field. A non-synthetic (e.g. SPX)
    # fields.npz stores `truth` as all-NaN, which would silently break the
    # truth panel and yield vmin=vmax=NaN. Fail fast and point to the right
    # script rather than emit a corrupted figure. (See REPRO_figure_surface.md.)
    if not bool(fields["synthetic"]) or not np.isfinite(fields["truth"]).any():
        raise SystemExit(
            "fields.npz is from a non-synthetic run (truth unavailable); "
            "figure.py requires a synthetic run. Re-run run.py with no --data "
            "flag, then this script. For the SPX real-data figure use "
            "figure_spx.py."
        )
    with open(os.path.join(out_dir, "diagnostics.json")) as fh:
        diag = json.load(fh)
    csv_path = os.path.join(out_dir, "black_scholes_sweep.csv")
    rows = _load_rows(csv_path) if os.path.exists(csv_path) else []
    if not rows:
        print(f"NOTE: {csv_path} not found; panel f will be a placeholder.")

    fig_base = os.path.join(out_dir, "figure_black_scholes_surface")
    plot(fields, diag, rows, fig_base, layout="wide")
    plot(fields, diag, rows, fig_base, layout="column")


if __name__ == "__main__":
    main(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
