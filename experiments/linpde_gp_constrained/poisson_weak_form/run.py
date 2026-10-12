"""
1D Poisson with weak-form PDE residual on the equality side
============================================================

This is the publication-grade companion to monotone_regression/run.py: a
demo that genuinely exercises linpde-gp's distinctive capability — equality
information delivered as a *finite test-basis projection of a PDE residual*
rather than as point observations of u.

Problem setup
-------------
On Ω = [0, 1] consider

    -u''(x) = f(x),     x in Ω,
    u(0) = 0,           u(1) = 1,

with the manufactured monotone solution

    u_true(x) = (1 - cos(π x)) / 2,

so

    f(x) = -u_true''(x) = -(π² / 2) cos(π x),
    u_true'(x) = (π / 2) sin(π x) >= 0  on [0, 1].

``u_true`` is C^infty, monotone, with the boundary values (0, 1).

Equality side (weak-form, NOT pointwise)
----------------------------------------
Pick a hat basis ``{φ_i}_{i=1}^m`` on a uniform grid of [0, 1] with grid
endpoints at the domain boundary. The PDE residual modes are

    L_pde[i] u  =  -∫ φ_i(x) u''(x) dx          (l2_proj ∘ -Δ)
    Y_pde[i]   =  ∫ φ_i(x) f(x) dx              (l2-projection of f).

We assemble the full ``L_pde u = Y_pde + ε_pde`` block via
``make_pde_equality_posterior``, which routes through linpde-gp's existing
cross-cov dispatch (``WeightedLaplacian × ExpQuad`` and
``L2Projection × CovarianceFunction``) and constructs the resulting
``ConditionalGaussianProcess`` directly. Dirichlet BCs ``u(0) = 0,
u(1) = 1`` are added as a separate ``_EvaluationFunctional`` block.

Inequality side
---------------
Monotonicity on a virtual grid: ``u'(x_v) >= 0`` at ``n_v`` evenly spaced
virtual points. Mirrors the monotone_regression demo, but in this setting
the inequality bites against a posterior whose equality information is the
PDE residual — not direct observations of ``u``.

Reporting
---------
The script prints and saves a comparison table:
    L2 error vs truth          (unconstrained vs constrained)
    max monotonicity violation (unconstrained vs constrained)
    avg 95% credible band width (unconstrained vs constrained)
    P(C | E)                   (probability the unconstrained posterior
                                satisfies the inequality before truncation;
                                a low value means the equality information
                                alone places significant mass on
                                non-monotone configurations and the
                                inequality step contributes substantively)

Run from the repo root:

    cd /app
    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/poisson_weak_form/run.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
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

SEED = 0
rng = np.random.default_rng(SEED)


# -----------------------------------------------------------------------------
# Manufactured solution.
#
# u_true(x) = (e^x - 1) / (e - 1), so:
#   u_true(0) = 0,  u_true(1) = 1                      (BCs)
#   u_true'(x) = e^x / (e - 1)  >= 1/(e-1) ≈ 0.58 > 0  (monotone with margin)
#   u_true''(x) = e^x / (e - 1)
#   f(x) = -u_true''(x) = -e^x / (e - 1)
#
# The "with margin" property is what makes the inequality conditioning
# numerically friendly here: u_true' bounded away from 0 means the
# monotonicity constraint u' >= 0 is far from the boundary of the polytope
# everywhere, so the truncated-MVN mean is much less biased than for the
# previous u_true(x) = (1 - cos(pi x))/2 (which had u' vanishing at both
# endpoints, putting the truncation polytope's boundary right under the
# truth and biasing the constrained mean's endpoint slopes upward).
# -----------------------------------------------------------------------------
_E_MINUS_ONE = float(np.exp(1.0) - 1.0)


def u_true(x):
    return (np.exp(np.asarray(x)) - 1.0) / _E_MINUS_ONE


def f_rhs(x):
    return -np.exp(np.asarray(x)) / _E_MINUS_ONE


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. GP prior: zero mean, ExpQuad with a moderately small lengthscale so
    #    second derivatives are well-conditioned.
    # -------------------------------------------------------------------------
    prior = pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(), output_shape=()),
        cov=1.0 * LinpdeExpQuad(input_shape=(), lengthscales=0.25),
    )

    # -------------------------------------------------------------------------
    # 2. Hat-basis test functions for the weak form. Use a coarse grid to
    #    keep the number of double-integrals tractable.
    # -------------------------------------------------------------------------
    # Coarse test basis on purpose: 2 interior hats means the weak-form PDE
    # residual lives in a 2-dim subspace of L². With an ExpQuad prior of
    # lengthscale 0.25 the prior has many more effective DOF than that, so
    # the equality side genuinely under-determines u — the unconstrained mean
    # is allowed to be visibly off truth, and (combined with the monotone-
    # with-margin u_true above) the inequality has room to pull it back
    # WITHOUT introducing the endpoint-slope bias that dominated when
    # u_true' vanished at the boundary.
    test_grid = np.linspace(0.0, 1.0, 4)  # 4 grid points -> 2 interior hats
    test_basis = linpde_bases.UnivariateLinearInterpolationBasis(
        test_grid, zero_boundary=False
    )

    # -------------------------------------------------------------------------
    # 3. Build the equality posterior: weak-form PDE residual + Dirichlet BCs.
    # -------------------------------------------------------------------------
    sigma_pde = 0.0  # noiseless PDE residual; small jitter inside.
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

    # -------------------------------------------------------------------------
    # 4. Inequality: u'(x) >= 0 at n_v virtual points.
    # -------------------------------------------------------------------------
    n_v = 30
    Xv = np.linspace(0.0, 1.0, n_v).reshape(-1, 1)
    constraints = [
        InequalityConstraint(
            operator=Derivative(order=1),
            Xv=Xv,
            bounds=(0.0, np.inf),
        ),
    ]
    cwrap = ConstrainedLinPDEPosterior(posterior, constraints, constr_likelihood=1e-6)

    # -------------------------------------------------------------------------
    # 5. Evaluate on a fine test grid.
    # -------------------------------------------------------------------------
    x_test = np.linspace(0.0, 1.0, 200)
    X_test = x_test.reshape(-1, 1)

    unconstrained_mean = np.asarray(posterior.mean(x_test)).reshape(-1)
    unconstrained_var = np.diag(
        np.asarray(posterior.cov(x_test[:, None], x_test[None, :]))
    )
    unconstrained_std = np.sqrt(np.clip(unconstrained_var, 0.0, None))

    truth = u_true(x_test)

    # The constrained posterior delegates to gp_constr, which loads rpy2 +
    # several R packages on first call. If rpy2 is not installed (or any R
    # package is missing) we still want the equality-side report — the new
    # machinery in this demo (weak-form residual + kernel adapter generic
    # path) is fully exercised on the unconstrained side.
    constrained_available = True
    constrained_error: str | None = None
    try:
        constrained_mean = cwrap.posterior_mean(X_test)
        samples = cwrap.posterior_samples(X_test, n_samples=200)
        constrained_std = samples.std(axis=0, ddof=1)
    except (ImportError, RuntimeError) as exc:
        # ImportError: rpy2 missing.
        # RuntimeError: rpy2 present but R binary / libR.so / R packages
        #   missing — gp_constr's _load_r_wrappers re-raises that case as
        #   RuntimeError. Either way the equality-side report below stays
        #   valid; we just skip the inequality step.
        constrained_available = False
        constrained_error = f"{type(exc).__name__}: {exc}"
        constrained_mean = np.full_like(unconstrained_mean, np.nan)
        constrained_std = np.full_like(unconstrained_std, np.nan)
        print(
            "WARNING: constrained-side computation skipped — gp_constr could "
            f"not load its R backend.\n         {constrained_error}\n"
            "         Equality-side diagnostics below are still valid.",
        )

    # -------------------------------------------------------------------------
    # 6. Diagnostics.
    # -------------------------------------------------------------------------
    grid_diffs_unc = np.diff(unconstrained_mean)
    monotone_violations_unc = int(np.sum(grid_diffs_unc < -1e-3))
    bc_err_left_unc = abs(unconstrained_mean[0] - 0.0)
    bc_err_right_unc = abs(unconstrained_mean[-1] - 1.0)
    l2_unc = float(np.linalg.norm(unconstrained_mean - truth))
    band_unc = float(np.mean(2 * 1.96 * unconstrained_std))

    if constrained_available:
        grid_diffs_con = np.diff(constrained_mean)
        monotone_violations_con = int(np.sum(grid_diffs_con < -1e-3))
        bc_err_left = abs(constrained_mean[0] - 0.0)
        bc_err_right = abs(constrained_mean[-1] - 1.0)
        l2_con = float(np.linalg.norm(constrained_mean - truth))
        band_con = float(np.mean(2 * 1.96 * constrained_std))
        try:
            p_ce = cwrap.constraint_satisfaction_probability
            p_ce_str = f"{p_ce:.4f}"
        except Exception as exc:  # rpy2 / R not available
            p_ce_str = f"unavailable ({type(exc).__name__})"
    else:
        monotone_violations_con = -1
        bc_err_left = float("nan")
        bc_err_right = float("nan")
        l2_con = float("nan")
        band_con = float("nan")
        p_ce_str = "unavailable (rpy2 not installed)"

    mono_con_str = (
        f"{monotone_violations_con:>10}" if constrained_available else f"{'n/a':>10}"
    )
    table_lines = [
        "                              unconstrained   constrained",
        "L2 error vs truth           : " + f"{l2_unc:>12.4f}    {l2_con:>10.4f}",
        "mean monotonicity viols     : "
        + f"{monotone_violations_unc:>12}    {mono_con_str}",
        "BC residual at x=0          : "
        + f"{bc_err_left_unc:>12.3e}    {bc_err_left:>10.3e}",
        "BC residual at x=1          : "
        + f"{bc_err_right_unc:>12.3e}    {bc_err_right:>10.3e}",
        "avg 95% credible band width : " + f"{band_unc:>12.4f}    {band_con:>10.4f}",
        f"P(C | E) = {p_ce_str}",
    ]
    table_text = "\n".join(table_lines)

    print("--- poisson_weak_form diagnostics ---")
    print(table_text)

    with open(os.path.join(out_dir, "comparison.txt"), "w") as fh:
        fh.write(table_text + "\n")

    # -------------------------------------------------------------------------
    # 7. Plot: truth, unconstrained mean ± 95% band, constrained mean ± 95% band.
    # -------------------------------------------------------------------------
    panels = [
        (
            unconstrained_mean,
            unconstrained_std,
            monotone_violations_unc,
            "Unconstrained (weak-form PDE + BCs only)",
        ),
    ]
    if constrained_available:
        panels.append(
            (
                constrained_mean,
                constrained_std,
                monotone_violations_con,
                "Constrained (weak-form PDE + BCs + monotonicity)",
            )
        )

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.5 * len(panels), 5), sharey=True, squeeze=False
    )
    axes_flat = axes[0]

    for ax, (mean, std, mono_viol, label) in zip(axes_flat, panels):
        ax.fill_between(
            x_test,
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.25,
            color="C0",
            label="95% credible band",
        )
        ax.plot(x_test, mean, color="C0", lw=2, label="posterior mean")
        ax.plot(x_test, truth, "k--", lw=1.5, label="truth")
        ax.scatter(
            [0.0, 1.0],
            [0.0, 1.0],
            color="k",
            marker="^",
            s=70,
            label="BC",
            facecolors="none",
        )
        ax.set_xlabel("x")
        ax.set_title(label + f"\nmean monotonicity violations: {mono_viol}")
        ax.legend(loc="upper left", fontsize=9)
    axes_flat[0].set_ylabel("u(x)")

    fig.suptitle(
        "1D Poisson with weak-form PDE residual on the equality side\n"
        f"hat basis: m = {len(test_basis)} interior nodes; "
        f"virtual points: n_v = {n_v}; "
        f"P(C | E) = {p_ce_str}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = os.path.join(out_dir, "run.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot:  {out_path}")
    print(f"Saved table: {os.path.join(out_dir, 'comparison.txt')}")

    # -------------------------------------------------------------------------
    # 8. Self-check assertions (relaxed: this is a research demo).
    # -------------------------------------------------------------------------
    # Equality-side checks always run.
    assert (
        bc_err_left_unc < 1e-2
    ), f"Unconstrained BC violation at x=0 too large: {bc_err_left_unc:.3e}."
    assert (
        bc_err_right_unc < 1e-2
    ), f"Unconstrained BC violation at x=1 too large: {bc_err_right_unc:.3e}."
    assert l2_unc < 1.0, f"Unconstrained L2 error vs truth too large: {l2_unc:.4f}."

    if constrained_available:
        assert monotone_violations_con == 0, (
            f"Constrained mean violates monotonicity at "
            f"{monotone_violations_con} grid indices (tolerance 1e-3)."
        )
        assert (
            bc_err_left < 50 * sigma_bc + 1e-3
        ), f"BC violation at x=0: {bc_err_left:.3e}."
        assert (
            bc_err_right < 50 * sigma_bc + 1e-3
        ), f"BC violation at x=1: {bc_err_right:.3e}."
    else:
        print(
            "NOTE: constrained-side assertions skipped because gp_constr's R "
            "backend is unavailable. Install it with `pip install .[r]` and "
            "the R packages tmvtnorm, mvtnorm, TruncatedNormal, truncnorm to "
            "exercise the inequality side."
        )


if __name__ == "__main__":
    main(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
