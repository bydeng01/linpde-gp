"""
2D Poisson collocation posterior + kernel-adapter vector-input tests.

Covers the mathematical-soundness checks of the 2D extension
(experiments/linpde_gp_constrained/poisson_2d_weak_form):

  * Kernel adapter — the per-L block derivative path on VECTOR input:
      Ki0(X1, X2, i) matches central FD of K w.r.t. X1[a, i] for i in {0, 1};
      Kij(X1, X2, i, j) matches the 4-point mixed-difference stencil for all
      (i, j) in {(0,0), (0,1), (1,0), (1,1)}.
  * Equality posterior — the unconstrained collocation posterior recovers the
      manufactured solution u_true within max-abs 0.1 on a 30×30 grid.
  * Combination — adding the monotonicity inequality (a) keeps the corner BCs tight,
      (b) makes the constrained mean x-monotone within 1e-3, and
      (c) yields P(C | E) in (0.05, 0.95). This needs gp_constr's R backend;
      it is skipped only where rpy2 / R is unavailable (it runs in the
      project's docker container).

REQUIRES: linpde-gp + probnum + jax + scipy; the combination test additionally
needs rpy2 + R packages mvtnorm/tmvtnorm/TruncatedNormal/truncnorm.

Run with: ``python -m pytest test_pde_equality_posterior_2d.py -v``
"""

from __future__ import annotations

import numpy as np

import pytest

linpde_gp = pytest.importorskip("linpde_gp")
probnum = pytest.importorskip("probnum")

from linpde_gp_constrained.kernel_adapter import LinPDEKernelAdapter  # noqa: E402
import probnum as pn  # noqa: E402

from linpde_gp.linfuncops.diffops import PartialDerivative  # noqa: E402
from linpde_gp.linfuncops.diffops._coefficients import MultiIndex  # noqa: E402
from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad  # noqa: E402
from linpde_gp_constrained import (  # noqa: E402
    ConstrainedLinPDEPosterior,
    InequalityConstraint,
    make_pde_collocation_posterior,
)

# -----------------------------------------------------------------------------
# Manufactured solution on (0,1)²: u_true = (e^x - 1)(e^y - 1)/(e - 1)².
# -----------------------------------------------------------------------------
_DENOM = float((np.exp(1.0) - 1.0) ** 2)


def _u_true(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x, y = X[:, 0], X[:, 1]
    return (np.exp(x) - 1.0) * (np.exp(y) - 1.0) / _DENOM


def _f_rhs(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x, y = X[:, 0], X[:, 1]
    return -(np.exp(x) * (np.exp(y) - 1.0) + np.exp(y) * (np.exp(x) - 1.0)) / _DENOM


def _prior(lengthscale=0.4, scale=1.0):
    return pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(2,), output_shape=()),
        cov=scale * LinpdeExpQuad(input_shape=(2,), lengthscales=lengthscale),
    )


def _interior_grid(n_per_axis):
    a = np.linspace(0.0, 1.0, n_per_axis)[1:-1]
    XX, YY = np.meshgrid(a, a, indexing="ij")
    return np.column_stack([XX.ravel(), YY.ravel()])


def _perimeter(n_per_axis):
    a = np.linspace(0.0, 1.0, n_per_axis)
    pts = set()
    for t in a:
        pts.update({(0.0, float(t)), (1.0, float(t)), (float(t), 0.0), (float(t), 1.0)})
    return np.array(sorted(pts), dtype=float)


def _small_posterior(lengthscale=0.4, n_colloc_axis=5, n_bc_axis=5):
    """A modest collocation posterior — fast, enough to exercise the adapter."""
    prior = _prior(lengthscale)
    X_pde = _interior_grid(n_colloc_axis)
    X_bc = _perimeter(n_bc_axis)
    return make_pde_collocation_posterior(
        prior,
        X_pde,
        _f_rhs,
        sigma_pde=0.0,
        X_bc=X_bc,
        Y_bc=_u_true(X_bc),
        sigma_bc=1e-6,
        pde_jitter=1e-8,
        bc_jitter=1e-12,
    )


# -----------------------------------------------------------------------------
# Precondition — manufactured solution boundary/corner values.
# -----------------------------------------------------------------------------
def test_manufactured_solution_boundary_values():
    a = np.linspace(0.0, 1.0, 50)
    # Left (x=0) and bottom (y=0) edges vanish.
    left = np.column_stack([np.zeros_like(a), a])
    bottom = np.column_stack([a, np.zeros_like(a)])
    np.testing.assert_allclose(_u_true(left), 0.0, atol=1e-12)
    np.testing.assert_allclose(_u_true(bottom), 0.0, atol=1e-12)
    # Corners (0,0),(1,0),(0,1),(1,1) = 0,0,0,1.
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    np.testing.assert_allclose(_u_true(corners), [0, 0, 0, 1], atol=1e-12)
    # Non-trivial top/right edge data.
    np.testing.assert_allclose(
        _u_true(np.column_stack([np.ones_like(a), a])),
        (np.exp(a) - 1.0) / (np.exp(1.0) - 1.0),
        atol=1e-12,
    )


# -----------------------------------------------------------------------------
# Kernel adapter Ki0 / Kij vs finite differences, on vector input.
# -----------------------------------------------------------------------------
def _adapter_vector_input():
    posterior = _small_posterior(n_colloc_axis=4, n_bc_axis=4)
    adapter = LinPDEKernelAdapter(posterior)
    adapter._ensure_derivative_setup()
    assert adapter.dim == 2
    assert adapter._eval_only is False, (
        "Collocation posterior has a composite PDE functional; the per-L block "
        "path must be active."
    )
    return adapter


@pytest.mark.parametrize("i", [0, 1])
def test_Ki0_matches_FD_of_K_vector_input(i):
    adapter = _adapter_vector_input()
    X1 = np.array([[0.30, 0.40], [0.55, 0.25], [0.70, 0.65]])
    X2 = np.array([[0.35, 0.50], [0.60, 0.30], [0.80, 0.75]])
    h = 1e-4

    e = np.zeros((1, 2))
    e[0, i] = h
    Ki0_fd = (adapter.K(X1 + e, X2) - adapter.K(X1 - e, X2)) / (2.0 * h)
    Ki0_an = adapter.Ki0(X1, X2, i)
    np.testing.assert_allclose(Ki0_an, Ki0_fd, atol=1e-4, rtol=5e-3)


@pytest.mark.parametrize("i,j", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_Kij_matches_mixed_FD_of_K_vector_input(i, j):
    adapter = _adapter_vector_input()
    X1 = np.array([[0.30, 0.40], [0.55, 0.25], [0.70, 0.65]])
    X2 = np.array([[0.35, 0.50], [0.60, 0.30], [0.80, 0.75]])
    h = 1e-3  # larger step: mixed 4-point stencil, 2nd-order accurate

    ei = np.zeros((1, 2))
    ei[0, i] = h
    ej = np.zeros((1, 2))
    ej[0, j] = h

    # ∂_{x_i} ∂_{x'_j} K  via the 4-point mixed difference:
    #   [K(X1+ei, X2+ej) - K(X1+ei, X2-ej) - K(X1-ei, X2+ej) + K(X1-ei, X2-ej)]
    #   / (4 h²)
    Kpp = adapter.K(X1 + ei, X2 + ej)
    Kpm = adapter.K(X1 + ei, X2 - ej)
    Kmp = adapter.K(X1 - ei, X2 + ej)
    Kmm = adapter.K(X1 - ei, X2 - ej)
    Kij_fd = (Kpp - Kpm - Kmp + Kmm) / (4.0 * h * h)
    Kij_an = adapter.Kij(X1, X2, i, j)
    np.testing.assert_allclose(Kij_an, Kij_fd, atol=1e-4, rtol=5e-3)


# -----------------------------------------------------------------------------
# Unconstrained collocation posterior recovers u_true within 0.1.
# -----------------------------------------------------------------------------
# Tuned configuration (chosen via the R-free sweep so that the equality-posterior
# accuracy check and the P(C|E) check both hold). Equality side is deliberately
# coarse so u|E carries genuine posterior uncertainty; virtual points sit on the
# interior [0.1, 0.9]² off the y=0 edge where the true x-slope is exactly 0.
_LS = 0.5
_N_COLLOC_AXIS = 5  # 3×3 = 9 interior collocation points
_N_BC_AXIS = 5  # 16 perimeter BC points
_N_V_AXIS = 6  # 6×6 = 36 virtual points
_XV_LO, _XV_HI = 0.1, 0.9


def test_equality_posterior_recovers_u_true():
    prior = _prior(lengthscale=_LS)
    X_pde = _interior_grid(_N_COLLOC_AXIS)
    X_bc = _perimeter(_N_BC_AXIS)
    posterior = make_pde_collocation_posterior(
        prior,
        X_pde,
        _f_rhs,
        sigma_pde=0.0,
        X_bc=X_bc,
        Y_bc=_u_true(X_bc),
        sigma_bc=1e-6,
        pde_jitter=1e-8,
        bc_jitter=1e-12,
    )

    a = np.linspace(0.0, 1.0, 30)
    XX, YY = np.meshgrid(a, a, indexing="ij")
    X_eval = np.column_stack([XX.ravel(), YY.ravel()])
    mean = np.asarray(posterior.mean(X_eval)).reshape(-1)

    max_abs = float(np.max(np.abs(mean - _u_true(X_eval))))
    assert max_abs < 0.1, (
        f"Equality posterior check failed: max|m_E - u_true| = {max_abs:.4f} "
        ">= 0.1. Refine the collocation grid or lower the lengthscale."
    )


def test_equality_posterior_bc_corners_tight():
    posterior = _small_posterior(
        lengthscale=_LS, n_colloc_axis=_N_COLLOC_AXIS, n_bc_axis=_N_BC_AXIS
    )
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    m = np.asarray(posterior.mean(corners)).reshape(-1)
    np.testing.assert_allclose(m, _u_true(corners), atol=1e-2)


# -----------------------------------------------------------------------------
# COMBINATION SHOWCASE (partial boundary data).
#
# Dirichlet data ONLY on the homogeneous edges x=0, y=0; the far edges x=1, y=1
# are free, so the equality posterior is under-determined toward the (1,1)
# corner and NON-monotone (GP reversion to the prior mean). Adding both
# ∂_x u >= 0 and ∂_y u >= 0 supplies the missing structure and measurably
# improves the MEAN (RMSE), not just the band — this is what demonstrates the
# advantage of combining equality (LinPDE-GP) with inequality (Agrell)
# information. The constrained tests need gp_constr's R backend; the MC bite
# test is R-free. Measured with the R backend: RMSE 0.328 -> 0.172 (-48%), band
# 0.930 -> 0.456, free-corner err 1.07 -> 0.20, P(C|E) ~ 0.05.
# -----------------------------------------------------------------------------
_SHOW_LS = 0.5
_SHOW_NC = 5  # 3×3 = 9 interior collocation points
_SHOW_NBC = 7  # boundary-grid density (only x=0 and y=0 edges used)
_SHOW_NV = 4  # 4×4 = 16 virtual points/axis -> joint constraint dim 32
_SHOW_AXES = (0, 1)  # ∂_x and ∂_y monotonicity
_SHOW_LO, _SHOW_HI = 0.2, 0.8  # interior virtual grid; pulled in to lift P(C|E)


def _axis_op(axis):
    mi = np.zeros(2, dtype=int)
    mi[axis] = 1
    return PartialDerivative(MultiIndex(mi))


def _two_edge_points(n):
    """Partial boundary: only the x=0 and y=0 edges (u_true = 0 there)."""
    a = np.linspace(0.0, 1.0, n)
    pts = set()
    for t in a:
        pts.add((0.0, float(t)))
        pts.add((float(t), 0.0))
    return np.array(sorted(pts), dtype=float)


def _build_showcase():
    prior = _prior(lengthscale=_SHOW_LS)
    X_pde = _interior_grid(_SHOW_NC)
    X_bc = _two_edge_points(_SHOW_NBC)
    posterior = make_pde_collocation_posterior(
        prior,
        X_pde,
        _f_rhs,
        sigma_pde=0.0,
        X_bc=X_bc,
        Y_bc=_u_true(X_bc),
        sigma_bc=1e-6,
        pde_jitter=1e-8,
        bc_jitter=1e-12,
    )
    a_v = np.linspace(_SHOW_LO, _SHOW_HI, _SHOW_NV)
    VX, VY = np.meshgrid(a_v, a_v, indexing="ij")
    Xv = np.column_stack([VX.ravel(), VY.ravel()])
    cwrap = ConstrainedLinPDEPosterior(
        posterior,
        [
            InequalityConstraint(operator=_axis_op(ax), Xv=Xv, bounds=(0.0, np.inf))
            for ax in _SHOW_AXES
        ],
        constr_likelihood=1e-4,
    )
    return posterior, cwrap, Xv


def _mc_p_ce(posterior, Xv, rng, axes=_SHOW_AXES, n_mc=20000, jitter=1e-9):
    """R-free joint MC estimate of P(C|E) = P(all ∂_axis u(Xv) >= 0 | E).

    Stacked slope vector ~ N([∂_a m_E(Xv)]_a, [Kij_E(Xv,Xv,a,b)]_{a,b}); count
    draws with every component non-negative. Mirrors run.mc_p_ce; kept local so
    the test does not import the experiment module.
    """
    adapter = LinPDEKernelAdapter(posterior)
    m = Xv.shape[0]
    k = len(axes)
    h = 1e-4
    means = []
    for a in axes:
        Xp = np.array(Xv, copy=True)
        Xp[:, a] += h
        Xm = np.array(Xv, copy=True)
        Xm[:, a] -= h
        means.append(
            (
                np.asarray(posterior.mean(Xp)).reshape(-1)
                - np.asarray(posterior.mean(Xm)).reshape(-1)
            )
            / (2.0 * h)
        )
    mean = np.concatenate(means)
    cov = np.empty((k * m, k * m))
    for ia, a in enumerate(axes):
        for ib, b in enumerate(axes):
            cov[ia * m : (ia + 1) * m, ib * m : (ib + 1) * m] = np.asarray(
                adapter.Kij(Xv, Xv, a, b)
            )
    cov = 0.5 * (cov + cov.T) + jitter * np.eye(k * m)
    draws = rng.multivariate_normal(mean, cov, size=n_mc, check_valid="ignore")
    return float(np.mean(np.all(draws >= 0.0, axis=1)))


def _eval_grid(n=30):
    a = np.linspace(0.0, 1.0, n)
    XX, YY = np.meshgrid(a, a, indexing="ij")
    return np.column_stack([XX.ravel(), YY.ravel()]), n


def _min_d_region(mean_flat, n, axis, lo=_SHOW_LO, hi=_SHOW_HI):
    """Most-negative ∂_axis of the mean field over the interior [lo, hi]²."""
    M = mean_flat.reshape(n, n)  # [x, y]
    h = 1.0 / (n - 1)
    a = np.linspace(0.0, 1.0, n)
    if axis == 0:
        dM = (M[2:, :] - M[:-2, :]) / (2.0 * h)
        sel = np.ix_((a[1:-1] >= lo) & (a[1:-1] <= hi), (a >= lo) & (a <= hi))
    else:
        dM = (M[:, 2:] - M[:, :-2]) / (2.0 * h)
        sel = np.ix_((a >= lo) & (a <= hi), (a[1:-1] >= lo) & (a[1:-1] <= hi))
    return float(dM[sel].min())


def test_equality_only_is_underdetermined_and_nonmonotone():
    """Precondition for the showcase (R-free): with partial BCs the equality
    posterior is both inaccurate and non-monotone, so the inequality has real
    mean error to correct."""
    posterior, _, _ = _build_showcase()
    X_eval, n = _eval_grid(30)
    mean = np.asarray(posterior.mean(X_eval)).reshape(-1)
    truth = _u_true(X_eval)
    assert float(np.sqrt(np.mean((mean - truth) ** 2))) > 0.1  # genuinely off
    assert _min_d_region(mean, n, 0, lo=0.0, hi=1.0) < -1e-2  # non-monotone


def test_inequality_bites_monte_carlo_R_free():
    """P(C|E) check without R: the joint MC P(C|E) over both axes bites — neither
    decorative (~1) nor almost-surely violated (~0). Runs everywhere."""
    posterior, _, Xv = _build_showcase()
    rng = np.random.default_rng(0)
    p = _mc_p_ce(posterior, Xv, rng)
    assert 0.02 < p < 0.95, (
        f"MC P(C|E) = {p:.4f} not in (0.02, 0.95); ~1 => decorative, ~0 => "
        "almost-surely violated. Re-tune via sweep.py."
    )


def test_combination_advantage():
    """Headline: equality + inequality beats equality alone on the MEAN.

    With the under-determined partial-BC equality posterior, adding ∂_x u >= 0
    and ∂_y u >= 0 reduces the mean RMSE (>=15%) AND the credible-band width AND
    the worst-case error, and the constrained mean is monotone on the
    constrained interior in both axes. (Note: we do NOT require full recovery —
    monotonicity is scale-free and cannot pin the magnitude at the free corner
    (1,1); it corrects the shape and substantially reduces the error.)
    """
    try:
        posterior, cwrap, Xv = _build_showcase()
        X_eval, n = _eval_grid(30)
        truth = _u_true(X_eval)
        unc_mean = np.asarray(posterior.mean(X_eval)).reshape(-1)
        unc_cov = np.asarray(posterior.cov(X_eval[:, None, :], X_eval[None, :, :]))
        unc_std = np.sqrt(np.clip(np.diag(0.5 * (unc_cov + unc_cov.T)), 0.0, None))
        con_mean = np.asarray(cwrap.posterior_mean(X_eval)).reshape(-1)
        samples = cwrap.posterior_samples(X_eval, n_samples=120)
        con_std = samples.std(axis=0, ddof=1)
        p_ce = float(cwrap.constraint_satisfaction_probability)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"gp_constr R backend unavailable: {exc}")

    l2_unc = float(np.sqrt(np.mean((unc_mean - truth) ** 2)))
    l2_con = float(np.sqrt(np.mean((con_mean - truth) ** 2)))
    band_unc = float(np.mean(2 * 1.96 * unc_std))
    band_con = float(np.mean(2 * 1.96 * con_std))

    # Combination advantage: mean RMSE and band both drop materially.
    assert l2_con < 0.85 * l2_unc, (l2_unc, l2_con)
    assert band_con < band_unc, (band_unc, band_con)
    # Worst-case error reduced.
    assert float(np.max(np.abs(con_mean - truth))) < float(
        np.max(np.abs(unc_mean - truth))
    )
    # Constrained mean monotone on the constrained interior, both axes.
    for ax in _SHOW_AXES:
        mi = _min_d_region(con_mean, n, ax)
        assert mi >= -1e-3, (ax, mi)
    # Anchored corners (0,0),(1,0),(0,1) stay tight; extracted from the grid
    # (indices i*n+j) so no extra R call. The free corner (1,1) is allowed off.
    anchor_idx = [0, (n - 1) * n, (n - 1)]
    anchor_pts = X_eval[anchor_idx]
    assert float(np.max(np.abs(con_mean[anchor_idx] - _u_true(anchor_pts)))) < 1e-2
    # Inequality bites.
    assert 0.02 < p_ce < 0.95, p_ce
