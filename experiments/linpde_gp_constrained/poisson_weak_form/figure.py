"""
Figure 1: Inequality conditioning on monotonicity reduces both prediction
error and predictive uncertainty for a 1D Poisson BVP with weak-form PDE
residual on the equality side.

Main message:
    For a 1D Poisson BVP -u'' = f, u(0)=0, u(1)=1 whose equality information
    is a low-rank weak-form PDE residual on a 2-mode hat-basis test space
    plus two Dirichlet BCs, conditioning on the monotonicity inequality
    u' >= 0 at virtual points simultaneously reduces the posterior mean's
    L^2 error and the 95% credible band width.

Panels:
  Panel a (unconstrained equality posterior):
    - sample paths drawn from u | (weak-form PDE, BCs) — most are not monotone
    - 95% credible band is wide; mean is visibly biased away from truth.
  Panel b (constrained posterior):
    - sample paths from u | (weak-form PDE, BCs, u' >= 0 at virtual points)
      are monotone by construction.
    - band is narrower; mean is closer to truth.
  Numerical drops are reported in-panel as small annotations:
    L2 error, average 95% band width, and P(C | E) (panel b only).

Design choices:
  - Reproducibility: seed pinned; deterministic regeneration of all data.
  - Honest comparison: identical y-axis range, identical sample-path count,
    identical credible-band definition (mean +/- 1.96 * std) on both panels.
  - Constraint cost: the panel b annotation includes P(C | E) so the reader
    can assess how much of the unconstrained posterior was discarded.

Layout: two stacked panels of equal weight, 3.25 in wide x 4.2 in tall
(single-column width; stacked top-to-bottom so the reader's eye traverses
unconstrained -> constrained as "before -> after"). Arial 7 pt; editable
SVG and PDF; TIFF at 600 dpi.

Run from the repo root inside the container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/poisson_weak_form/figure.py

Outputs (next to this script):
    figure_poisson_weak_form.svg     master vector (editable text)
    figure_poisson_weak_form.pdf     Illustrator-friendly vector
    figure_poisson_weak_form.tiff    600 dpi raster, journal submission
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import probnum as pn

from linpde_gp.functions import bases as linpde_bases
from linpde_gp.linfuncops.diffops import Derivative
from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad
from linpde_gp_constrained import (
    ConstrainedLinPDEPosterior,
    InequalityConstraint,
    make_pde_equality_posterior,
)

# -----------------------------------------------------------------------------
# Publication rcParams. Editable text in SVG/PDF, no top/right spines, thin
# lines. 7 pt is the journal target body size.
# -----------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",  # editable text in SVG
        "pdf.fonttype": 42,  # TrueType in PDF
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
    }
)


# -----------------------------------------------------------------------------
# Palette: one neutral family + one signal family + one accent. We deliberately
# avoid the saturated red/blue clash and use a warm coral for the
# "uncorrected" panel and a cool slate for the "improved" panel — a
# directional cue that reads even at thumbnail size.
# -----------------------------------------------------------------------------
COLOR_UNC = "#C97062"  # warm desaturated coral, "before"
COLOR_CON = "#4F7CAC"  # cool desaturated slate,  "after"
COLOR_TRUTH = "#222222"
COLOR_PATH_ALPHA = 0.10
COLOR_BAND_ALPHA = 0.22


# -----------------------------------------------------------------------------
# Manufactured solution (matches run.py).
# -----------------------------------------------------------------------------
_E_MINUS_ONE = float(np.exp(1.0) - 1.0)


def u_true(x):
    return (np.exp(np.asarray(x)) - 1.0) / _E_MINUS_ONE


def f_rhs(x):
    return -np.exp(np.asarray(x)) / _E_MINUS_ONE


SEED = 0
N_PATHS_PLOT = 25  # how many sample paths to overlay per panel


# -----------------------------------------------------------------------------
# Compute everything from scratch (deterministic with SEED).
# -----------------------------------------------------------------------------
def compute() -> dict:
    rng = np.random.default_rng(SEED)

    prior = pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(), output_shape=()),
        cov=1.0 * LinpdeExpQuad(input_shape=(), lengthscales=0.25),
    )
    test_grid = np.linspace(0.0, 1.0, 4)  # 2 interior hats
    test_basis = linpde_bases.UnivariateLinearInterpolationBasis(
        test_grid, zero_boundary=False
    )

    sigma_pde = 0.0
    sigma_bc = 1e-6
    posterior = make_pde_equality_posterior(
        prior,
        test_basis,
        f_rhs,
        sigma_pde=sigma_pde,
        X_bc=np.array([0.0, 1.0]),
        Y_bc=np.array([0.0, 1.0]),
        sigma_bc=sigma_bc,
        pde_jitter=1e-8,
        bc_jitter=1e-12,
    )

    n_v = 30
    Xv = np.linspace(0.0, 1.0, n_v).reshape(-1, 1)
    cwrap = ConstrainedLinPDEPosterior(
        posterior,
        [
            InequalityConstraint(
                operator=Derivative(order=1),
                Xv=Xv,
                bounds=(0.0, np.inf),
            )
        ],
        constr_likelihood=1e-6,
    )

    x_test = np.linspace(0.0, 1.0, 250)
    X_test = x_test.reshape(-1, 1)

    unc_mean = np.asarray(posterior.mean(x_test)).reshape(-1)
    unc_cov = np.asarray(posterior.cov(x_test[:, None], x_test[None, :]))
    unc_cov_sym = 0.5 * (unc_cov + unc_cov.T)
    unc_var = np.clip(np.diag(unc_cov_sym), 0.0, None)
    unc_std = np.sqrt(unc_var)
    # Sample from the unconstrained posterior for path overlays. Add a tiny
    # jitter to keep the multivariate normal sampler from complaining.
    unc_cov_for_samples = unc_cov_sym + 1e-10 * np.eye(unc_cov_sym.shape[0])
    unc_paths = rng.multivariate_normal(
        unc_mean,
        unc_cov_for_samples,
        size=N_PATHS_PLOT,
        check_valid="ignore",
    )

    con_mean = cwrap.posterior_mean(X_test)
    con_paths = cwrap.posterior_samples(X_test, n_samples=N_PATHS_PLOT)
    con_std = con_paths.std(axis=0, ddof=1)

    truth = u_true(x_test)
    diagnostics = dict(
        l2_unc=float(np.linalg.norm(unc_mean - truth)),
        l2_con=float(np.linalg.norm(con_mean - truth)),
        band_unc=float(np.mean(2 * 1.96 * unc_std)),
        band_con=float(np.mean(2 * 1.96 * con_std)),
    )
    try:
        diagnostics["p_ce"] = float(cwrap.constraint_satisfaction_probability)
    except Exception:
        diagnostics["p_ce"] = np.nan

    return dict(
        x_test=x_test,
        truth=truth,
        unc_mean=unc_mean,
        unc_std=unc_std,
        unc_paths=unc_paths,
        con_mean=con_mean,
        con_std=con_std,
        con_paths=con_paths,
        diagnostics=diagnostics,
    )


# -----------------------------------------------------------------------------
# Plot.
# -----------------------------------------------------------------------------
def plot(data: dict, fig_path_base: str) -> None:
    x = data["x_test"]
    truth = data["truth"]
    diag = data["diagnostics"]

    # Single column: width = \columnwidth ~= 3.25 in. Stacking the two
    # panels top-to-bottom lets each panel keep a readable aspect ratio
    # (3.25 x 1.7 in per panel) at the cost of a taller overall figure.
    fig_w_in = 3.25
    fig_h_in = 4.20
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_w_in, fig_h_in),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(hspace=0.32),
    )

    # --- Shared y-range so the panels read as a direct comparison.
    y_min = min(
        data["unc_paths"].min(),
        data["con_paths"].min(),
        truth.min(),
        (data["unc_mean"] - 1.96 * data["unc_std"]).min(),
    )
    y_max = max(
        data["unc_paths"].max(),
        data["con_paths"].max(),
        truth.max(),
        (data["unc_mean"] + 1.96 * data["unc_std"]).max(),
    )
    y_pad = 0.05 * (y_max - y_min)

    panels = [
        dict(
            ax=axes[0],
            label="a",
            short="Unconstrained",
            title="Equality only:  weak-form PDE + BCs",
            mean=data["unc_mean"],
            std=data["unc_std"],
            paths=data["unc_paths"],
            color=COLOR_UNC,
            annotation=(
                f"L$^2$ err   = {diag['l2_unc']:.3f}\n"
                f"avg 95% band = {diag['band_unc']:.3f}"
            ),
        ),
        dict(
            ax=axes[1],
            label="b",
            short="Constrained",
            title="+ inequality:  $u'(x) \\geq 0$ at 30 virtual pts",
            mean=data["con_mean"],
            std=data["con_std"],
            paths=data["con_paths"],
            color=COLOR_CON,
            annotation=(
                f"L$^2$ err   = {diag['l2_con']:.3f}\n"
                f"avg 95% band = {diag['band_con']:.3f}\n"
                f"$P(C\\,|\\,E)$    = {diag['p_ce']:.3f}"
            ),
        ),
    ]

    for p in panels:
        ax = p["ax"]

        # Sample paths in the panel's accent color, very faint.
        for path in p["paths"]:
            ax.plot(
                x,
                path,
                color=p["color"],
                lw=0.4,
                alpha=COLOR_PATH_ALPHA,
                rasterized=False,
            )

        # Credible band.
        ax.fill_between(
            x,
            p["mean"] - 1.96 * p["std"],
            p["mean"] + 1.96 * p["std"],
            color=p["color"],
            alpha=COLOR_BAND_ALPHA,
            linewidth=0,
            label="95% credible band",
        )

        # Posterior mean (thicker, on top of band).
        ax.plot(x, p["mean"], color=p["color"], lw=1.4, label="posterior mean")

        # Truth and BCs.
        ax.plot(
            x,
            truth,
            color=COLOR_TRUTH,
            lw=0.7,
            ls=(0, (4, 2)),
            label="true $u(x)$",
            dash_capstyle="butt",
        )
        ax.scatter(
            [0.0, 1.0],
            [0.0, 1.0],
            color=COLOR_TRUTH,
            marker="^",
            s=18,
            zorder=5,
            label="Dirichlet BC",
            facecolors="white",
            edgecolors=COLOR_TRUTH,
            linewidth=0.7,
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("$u(x)$")

        # Panel label (a, b) in bold at the top-left corner of each axes.
        ax.text(
            -0.13,
            1.05,
            p["label"],
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Subtitle inside the panel area.
        ax.text(
            0.5,
            1.02,
            p["title"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.6,
        )

        # In-panel diagnostics annotation, lower right.
        ax.text(
            0.97,
            0.04,
            p["annotation"],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            family="monospace",
            bbox=dict(
                facecolor="white",
                edgecolor="0.7",
                linewidth=0.4,
                boxstyle="round,pad=0.25",
                alpha=0.92,
            ),
        )

    # x-axis label only on bottom panel.
    axes[-1].set_xlabel("$x$")

    # Single shared legend, upper-left of top panel, on two columns to keep
    # it within the panel's horizontal extent at the single-column width.
    handles, labels = axes[0].get_legend_handles_labels()
    order = [
        labels.index(s)
        for s in ("true $u(x)$", "Dirichlet BC", "posterior mean", "95% credible band")
    ]
    axes[0].legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.00),
        fontsize=5.8,
        ncol=2,
        columnspacing=0.9,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.13, right=0.985, top=0.95, bottom=0.085)

    save_pub(fig, fig_path_base)
    plt.close(fig)


def save_pub(fig, base_path: str, dpi: int = 600) -> None:
    fig.savefig(f"{base_path}.svg")
    fig.savefig(f"{base_path}.pdf")
    fig.savefig(f"{base_path}.tiff", dpi=dpi)
    print(f"Saved: {base_path}.svg")
    print(f"Saved: {base_path}.pdf")
    print(f"Saved: {base_path}.tiff (dpi={dpi})")


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "figure_poisson_weak_form")

    print("Computing equality + constrained posteriors ...")
    data = compute()
    print("Diagnostics:")
    for k, v in data["diagnostics"].items():
        print(f"  {k:>10s} = {v:.4f}")
    print("Rendering figure ...")
    plot(data, base)


if __name__ == "__main__":
    main(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
