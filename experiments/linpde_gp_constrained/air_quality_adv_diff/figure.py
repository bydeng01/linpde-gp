"""
Publication figure for the positivity-constrained advection-diffusion experiment.

Main message:
    Conditioning the steady advection-diffusion posterior on nonnegativity
    reduces the spurious sub-zero mass the PDE + sparse-monitor posterior places
    in the clean upwind air, and the amount of infeasible probability it reclaims
    grows as the monitors degrade.

Layout: a clean row of reconstruction maps (main figure). The noise sweep
is a separate, well-proportioned appendix figure (figure_..._sweep), and the
per-arm scalar metrics are a companion table (table_air_quality_arms.tex),
where exact multi-metric values read better than bars and the off-scale +C
ablation no longer distorts a shared axis.

Main-figure panels
------------------
  a  monitor map over the truth plume (monitors, held-out sites, virtual points).
  b  equality-only (monitors+E) mean with its negative region overlaid.
  c  combined (monitors+E+C) mean -- nonnegativity (enforced at 9 virtual
     points, not pointwise) shrinks the sub-zero region but does not erase
     it; each map reports the fraction of the field still below zero.

Appendix figure (figure_..._sweep): sub-zero-tail proxy reduction and held-RMSE
gain vs observation noise -- the proxy reduction grows
as the data degrade.

Companion table (table_air_quality_arms.tex): held-out RMSE, mean<0 fraction,
sub-zero-tail proxy and 1.96-sigma band width per arm at the seed-0 headline point.

Inputs (rendered without re-running R):
    fields.npz            (run.py)         -- arm means/stds, truth, point sets.
    diagnostics.json      (run.py)         -- per-arm scalar metrics (table).
    air_quality_adv_diff_sweep.csv (sweep.py) -- seed-averaged sweep (appendix fig).

Run from the repo root inside the container (after run.py + sweep.py):
    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \
      python experiments/linpde_gp_constrained/air_quality_adv_diff/figure.py

Outputs (next to this script):
    figure_air_quality_adv_diff.{pdf,svg,tiff}        -- maps a,b,c (main figure),
    figure_air_quality_adv_diff_column.{pdf,svg,tiff} -- single-column maps,
    figure_air_quality_adv_diff_sweep.{pdf,svg,tiff}  -- noise sweep (appendix),
    table_air_quality_arms.tex                        -- per-arm metrics.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
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

C_BASE = "#4D4D4D"  # neutral baseline (monitors+E)
C_COMB = "#0F4D92"  # blue: combined (ours)
C_GAIN = "#8BCF8B"  # green: calibration gain
C_NEG = "#C0392B"  # red: infeasible / negative
ARMS = ("monitors", "monitors+E", "monitors+C", "monitors+E+C")
ARM_SHORT = {
    "monitors": "mon",
    "monitors+E": "+E",
    "monitors+C": "+C",
    "monitors+E+C": "+E+C",
}


def _load_fields(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _load_sweep(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = dict(r)
        d["n_obs"] = int(float(r["n_obs"]))
        for k in (
            "sigma_obs",
            "rmse_held_E",
            "rmse_held_EC",
            "subzero_E",
            "subzero_EC",
        ):
            try:
                d[k] = float(r[k])
            except (TypeError, ValueError):
                d[k] = float("nan")
        out.append(d)
    return out


def _rom(group, ku, kc):
    u = np.array([r[ku] for r in group], float)
    c = np.array([r[kc] for r in group], float)
    u, c = u[~np.isnan(u)], c[~np.isnan(c)]
    return 1.0 - c.mean() / u.mean() if u.size and u.mean() > 0 else np.nan


# -----------------------------------------------------------------------------
# Map panels.
# -----------------------------------------------------------------------------
def _map(ax, EX, EY, Z, n, title, vmin, vmax, cmap="viridis"):
    im = ax.pcolormesh(
        EX,
        EY,
        Z.reshape(n, n),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_title(title, fontsize=7)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    return im


def _panel_truth(ax, F, n, vmax):
    im = _map(ax, F["EX"], F["EY"], F["truth"], n, "truth + monitors", 0.0, vmax)
    ax.scatter(
        F["X_obs"][:, 0],
        F["X_obs"][:, 1],
        s=10,
        c="red",
        edgecolors="k",
        linewidths=0.3,
        label="monitor",
        zorder=3,
    )
    ax.scatter(
        F["X_held"][:, 0],
        F["X_held"][:, 1],
        s=14,
        marker="^",
        c="white",
        edgecolors="k",
        linewidths=0.4,
        label="held-out",
        zorder=3,
    )
    ax.scatter(
        F["Xv"][:, 0],
        F["Xv"][:, 1],
        s=5,
        c=[(0, 0, 0, 0.5)],
        edgecolors="white",
        linewidths=0.5,
        label="virtual",
        zorder=2,
    )
    leg = ax.legend(
        loc="upper left",
        fontsize=5.0,
        frameon=True,
        handletextpad=0.2,
        borderaxespad=0.1,
        labelspacing=0.2,
    )
    # Semi-transparent white panel so the labels stay legible over the dark
    # (low-c) upper-left corner of the heatmap.
    fr = leg.get_frame()
    fr.set_facecolor("white")
    fr.set_alpha(0.7)
    fr.set_edgecolor("none")
    fr.set_linewidth(0.0)
    return im


def _panel_mean_with_neg(ax, F, n, vmax, arm, title, ref_arm=None):
    mean = F[f"mean__{arm}"]
    M = mean.reshape(n, n)
    has_neg = bool(np.any(M < 0))
    # Report the residual infeasible area in the title so b -> c reads honestly:
    # nonnegativity shrinks the sub-zero region but does not erase it.
    if has_neg:
        title = f"{title}  ({100.0 * (M < 0).mean():.1f}% $m<0$)"
    im = _map(ax, F["EX"], F["EY"], mean, n, title, 0.0, vmax)
    # Ghost the looser arm's (<0) boundary as a dashed outline, so the red fill
    # shrinking inside it reads as the constraint's effect -- the top row then
    # carries the result rather than looking near-identical to panel b.
    if ref_arm is not None:
        R = F[f"mean__{ref_arm}"].reshape(n, n)
        if np.any(R < 0):
            ax.contour(
                F["EX"],
                F["EY"],
                R,
                levels=[0.0],
                colors="white",
                linewidths=0.7,
                linestyles="dashed",
                zorder=4,
            )
            ax.text(
                0.97,
                0.045,
                r"$m=0$ before $C$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=4.8,
                color="white",
                zorder=5,
            )
    # Overlay the infeasible (mean < 0) region as a red contour.
    if has_neg:
        ax.contour(F["EX"], F["EY"], M, levels=[0.0], colors=C_NEG, linewidths=0.9)
        ax.contourf(
            F["EX"],
            F["EY"],
            (M < 0).astype(float),
            levels=[0.5, 1.5],
            colors=[C_NEG],
            alpha=0.28,
        )
    return im


# -----------------------------------------------------------------------------
# Quantitative panels.
# -----------------------------------------------------------------------------
ARM_TABLE_LABEL = {
    "monitors": "monitors",
    "monitors+E": "$+E$",
    "monitors+C": "$+C$",
    "monitors+E+C": "$+E+C$",
}


def write_arm_table(diag, path):
    """Per-arm scalar metrics -> a booktabs LaTeX table (replaces the former
    bar panels d, e). One representative noise seed (seed 0, the headline point
    n_obs=14 / sigma_obs=0.1); lower is better for every column, and the best
    (lowest) value in each column is bold-faced. Seed-averaged feasibility-proxy
    numbers live in the prose macros and in panel d."""
    # sub-zero tail is intentionally omitted: it is reported (seed-averaged) in
    # the prose, and panels b/c already show the negative region, so listing a
    # second seed-0 value here would invite a spurious prose/table mismatch.
    cols = [
        ("rmse_held", "held-out RMSE"),
        ("neg_frac", r"fraction $m<0$"),
        ("band", r"$B_{1.96}$"),
    ]
    arms = list(ARMS)
    best = {k: min(arms, key=lambda a: diag["arms"][a][k]) for k, _ in cols}
    body = []
    for a in arms:
        cells = []
        for k, _ in cols:
            s = f"{diag['arms'][a][k]:.3f}"
            cells.append(f"\\textbf{{{s}}}" if a == best[k] else s)
        body.append(f"  {ARM_TABLE_LABEL[a]:<9} & " + " & ".join(cells) + r" \\")
    out = [
        "% Auto-generated by figure.py -- do not edit by hand.",
        "% Per-arm metrics at the seed-0 headline point (n_obs=14, sigma=0.1).",
        "% Bold = best (lowest) in each column; lower is better throughout.",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "arm & " + " & ".join(name for _, name in cols) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"Saved: {path}")


def _panel_sweep(ax, sweep, fixed_nobs):
    """vs sigma_obs at fixed n_obs: tail-reduction (robust) + held-RMSE gain."""
    rows = [r for r in sweep if r["n_obs"] == fixed_nobs]
    by_s = defaultdict(list)
    for r in rows:
        by_s[r["sigma_obs"]].append(r)
    xs = sorted(by_s)
    tail = [100 * _rom(by_s[s], "subzero_E", "subzero_EC") for s in xs]
    rmse = [100 * _rom(by_s[s], "rmse_held_E", "rmse_held_EC") for s in xs]
    ax.axhline(0.0, color="0.7", lw=0.6, zorder=0)
    ax.plot(
        xs,
        tail,
        "-o",
        color=C_GAIN,
        mec="k",
        mew=0.5,
        ms=3.4,
        label=r"tail proxy $Q_0$",
        zorder=3,
    )
    ax.plot(
        xs,
        rmse,
        "-s",
        color=C_COMB,
        mfc="white",
        mec=C_COMB,
        mew=0.9,
        ms=3.2,
        label="held-out RMSE",
        zorder=3,
    )
    ax.set_xlabel(r"observation noise $\sigma_{\mathrm{obs}}$")
    ax.set_ylabel(r"relative gain $G_M$ [%]")
    # The caption supplies the panel description and fixed n_obs value
    # (\AqNobsFixed).
    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.16, 0.5),
        fontsize=5.6,
        frameon=False,
        handlelength=1.2,
        labelspacing=0.25,
    )


# -----------------------------------------------------------------------------
# Assembly.
# -----------------------------------------------------------------------------
def plot(F, fig_base, layout="wide"):
    """Main figure: the reconstruction-map row a,b,c. The noise sweep is a
    separate appendix figure (see plot_sweep)."""
    n = int(F["n_eval_axis"])
    vmax = (
        float(np.nanmax(F["truth"]))
        if "truth" in F
        else float(np.nanmax(F["mean__monitors+E"]))
    )

    if layout == "column":
        fig = plt.figure(figsize=(3.3, 1.5))
        gs = fig.add_gridspec(1, 3, wspace=0.34)
        fig_base = f"{fig_base}_column"
    else:
        fig = plt.figure(figsize=(7.2, 2.6))
        gs = fig.add_gridspec(1, 3, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    im = _panel_truth(ax_a, F, n, vmax)
    _panel_mean_with_neg(ax_b, F, n, vmax, "monitors+E", "+E mean")
    _panel_mean_with_neg(
        ax_c, F, n, vmax, "monitors+E+C", "+E+C mean", ref_arm="monitors+E"
    )
    # Compact colorbar inset on the rightmost map.
    cax = ax_c.inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("c", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        ax.annotate(
            lab,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-14, 2),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )

    fig.savefig(f"{fig_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{fig_base}.svg", bbox_inches="tight")
    fig.savefig(f"{fig_base}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_base}.pdf / .svg / .tiff")


def plot_sweep(sweep, fixed_nobs, fig_base):
    """Standalone appendix figure: the observation-noise sweep (sub-zero-tail
    proxy reduction vs regime-dependent held-RMSE gain), at a sensible
    single-panel aspect rather than squashed into the map figure."""
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    _panel_sweep(ax, sweep, fixed_nobs)
    out = f"{fig_base}_sweep"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}.pdf / .svg / .tiff")


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    F = _load_fields(os.path.join(out_dir, "fields.npz"))
    with open(os.path.join(out_dir, "diagnostics.json")) as fh:
        diag = json.load(fh)
    sweep_path = os.path.join(out_dir, "air_quality_adv_diff_sweep.csv")
    if not os.path.exists(sweep_path):
        raise SystemExit(f"Sweep CSV not found: {sweep_path}. Run sweep.py first.")
    sweep = _load_sweep(sweep_path)
    # Modal fixed n_obs among the sigma-axis rows (matches sweep.FIXED_NOBS).
    fixed_nobs = max(
        set(r["n_obs"] for r in sweep),
        key=lambda nn: sum(r["n_obs"] == nn for r in sweep),
    )

    write_arm_table(diag, os.path.join(out_dir, "table_air_quality_arms.tex"))

    base = os.path.join(out_dir, "figure_air_quality_adv_diff")
    plot(F, base, layout="wide")
    plot(F, base, layout="column")
    plot_sweep(sweep, fixed_nobs, base)


if __name__ == "__main__":
    main(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
