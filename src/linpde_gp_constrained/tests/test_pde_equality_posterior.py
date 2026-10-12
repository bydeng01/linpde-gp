"""
Integration tests for ``make_pde_equality_posterior`` and the kernel adapter's
non-evaluation ``L_E`` path.

REQUIRES: linpde-gp + probnum + jax + scipy. Skipped in environments without
linpde-gp installed.

Mathematical claims tested
--------------------------
1. **Weak-form posterior matches the manufactured solution** when given exact
   PDE data + exact BCs. With u_true(x) = (1 - cos π x)/2, f = -u_true'' =
   -(π²/2) cos(π x), and a hat basis on [0, 1]:

       posterior_mean(x) ≈ u_true(x)   on [0, 1]    (within discretisation error)

   This is a smoke test of the entire equality assembly: bivariate kernel
   construction (Δ_s Δ_t k), dispatched L2 projections, gram solve,
   PriorPredictiveCrossCovariance evaluation, and ConditionalGaussianProcess
   construction. Tolerances are loose because the hat-basis discretisation is
   coarse.

2. **Kernel adapter generic path agrees with eval-only path** on a posterior
   conditioned only on point evaluations. We force ``_eval_only = False`` to
   route the per-L block code, and compare Ki0 and Kij to the fast path
   element-wise. Catches block-assembly / shape bugs without requiring the
   weak-form machinery to be correct.

3. **Per-L kLa block** equals ``posterior._kLas`` evaluation: the adapter's
   ``_eval_kLa(X)`` should reproduce ``posterior_gp._kLas(X)`` reshaped, since
   they are mathematically the same object (concatenated cross-covariances
   on the prior).

4. **Ki0 FD self-consistency on the weak-form posterior**: the adapter's
   ``Ki0(X1, X2, 0)`` matches ``d/dX1[a, 0] K(X1, X2)`` by central
   differencing. This is the *correctness* test for the new path.

Run with: ``python -m pytest test_pde_equality_posterior.py -v``
"""

from __future__ import annotations

import numpy as np

import pytest

linpde_gp = pytest.importorskip("linpde_gp")
probnum = pytest.importorskip("probnum")

from linpde_gp_constrained.kernel_adapter import LinPDEKernelAdapter  # noqa: E402
import probnum as pn  # noqa: E402  (after importorskip)

from linpde_gp.functions import bases as linpde_bases  # noqa: E402
from linpde_gp.linfuncops.diffops import WeightedLaplacian  # noqa: E402
from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad  # noqa: E402
from linpde_gp_constrained import make_pde_equality_posterior  # noqa: E402

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _build_zero_mean_expquad(lengthscale: float = 0.25, scale: float = 1.0):
    return pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(), output_shape=()),
        cov=scale * LinpdeExpQuad(input_shape=(), lengthscales=lengthscale),
    )


def _u_true(x):
    return 0.5 * (1.0 - np.cos(np.pi * np.asarray(x)))


def _f_rhs(x):
    return -(np.pi**2) * 0.5 * np.cos(np.pi * np.asarray(x))


# -----------------------------------------------------------------------------
# Test 1 — Weak-form posterior recovers manufactured solution.
# -----------------------------------------------------------------------------


def test_weak_form_posterior_recovers_manufactured_solution():
    """The mean of u | (weak-form -u''=f, BCs) is close to the truth."""
    prior = _build_zero_mean_expquad(lengthscale=0.25)
    grid = np.linspace(0.0, 1.0, 11)
    test_basis = linpde_bases.UnivariateLinearInterpolationBasis(
        grid, zero_boundary=False
    )

    posterior = make_pde_equality_posterior(
        prior,
        test_basis,
        _f_rhs,
        sigma_pde=0.0,
        X_bc=np.array([0.0, 1.0]),
        Y_bc=np.array([0.0, 1.0]),
        sigma_bc=1e-6,
        pde_jitter=1e-8,
        bc_jitter=1e-12,
    )

    x_test = np.linspace(0.0, 1.0, 100)
    mean = np.asarray(posterior.mean(x_test)).reshape(-1)

    # Loose: hat-basis weak-form discretisation with 9 interior nodes is
    # coarse; we expect ~1e-2 absolute error band.
    assert np.max(np.abs(mean - _u_true(x_test))) < 5e-2


def test_weak_form_posterior_satisfies_bcs():
    """Dirichlet BCs are tight (within sigma_bc * O(10))."""
    prior = _build_zero_mean_expquad(lengthscale=0.25)
    grid = np.linspace(0.0, 1.0, 11)
    test_basis = linpde_bases.UnivariateLinearInterpolationBasis(
        grid, zero_boundary=False
    )

    sigma_bc = 1e-6
    posterior = make_pde_equality_posterior(
        prior,
        test_basis,
        _f_rhs,
        X_bc=np.array([0.0, 1.0]),
        Y_bc=np.array([0.0, 1.0]),
        sigma_bc=sigma_bc,
    )

    m_at_bc = np.asarray(posterior.mean(np.array([0.0, 1.0]))).reshape(-1)
    # BC residual should be at most a few orders of magnitude above sigma_bc.
    assert abs(m_at_bc[0] - 0.0) < 1e-3
    assert abs(m_at_bc[1] - 1.0) < 1e-3


def test_weak_form_posterior_no_bc_branch_runs():
    """The X_bc=None branch should construct without error and predict."""
    prior = _build_zero_mean_expquad(lengthscale=0.4)
    grid = np.linspace(0.0, 1.0, 7)
    test_basis = linpde_bases.UnivariateLinearInterpolationBasis(
        grid, zero_boundary=False
    )
    posterior = make_pde_equality_posterior(
        prior,
        test_basis,
        _f_rhs,
        sigma_pde=0.0,
        X_bc=None,
    )
    x_test = np.linspace(0.0, 1.0, 20)
    mean = np.asarray(posterior.mean(x_test)).reshape(-1)
    # With no BCs the mean is unanchored at the boundary — we just check that
    # the prediction is finite and not all-zero (i.e. the PDE info bit).
    assert np.all(np.isfinite(mean))
    assert not np.allclose(mean, 0.0)


# -----------------------------------------------------------------------------
# Test 2 — kernel adapter generic path agrees with eval-only path.
#
# Build a posterior using only point evaluations (forces _eval_only=True), then
# wrap two adapters: one in the natural eval-only fast path, the other
# manually flipped to the per-L block path (we monkeypatch _eval_only and
# clear _X_E_total). The two should produce numerically equal Ki0 and Kij.
# -----------------------------------------------------------------------------


def _eval_only_posterior():
    prior = _build_zero_mean_expquad(lengthscale=0.3)
    rng = np.random.default_rng(0)
    X_obs = np.linspace(0.1, 0.9, 5)
    Y_obs = np.sin(2.0 * X_obs) + 0.05 * rng.standard_normal(X_obs.shape)
    sigma = 0.05
    posterior = prior.condition_on_observations(
        Y=Y_obs,
        X=X_obs,
        b=pn.randvars.Normal(np.zeros(X_obs.shape), sigma**2 * np.eye(X_obs.size)),
    )
    return posterior


def test_kernel_adapter_generic_path_matches_fast_path_for_eval_only():
    posterior = _eval_only_posterior()

    # Fast path: untouched adapter.
    fast = LinPDEKernelAdapter(posterior)
    fast._ensure_derivative_setup()
    assert fast._eval_only is True

    # Generic path: same posterior, but force _eval_only=False so the per-L
    # block code runs.
    slow = LinPDEKernelAdapter(posterior)
    slow._ensure_derivative_setup()
    slow._eval_only = False
    slow._X_E_total = None

    X1 = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    X2 = np.linspace(0.05, 0.95, 4).reshape(-1, 1)

    # K is the same path regardless (always uses posterior.cov), so we don't
    # bother re-checking it here.
    Ki0_fast = fast.Ki0(X1, X2, 0)
    Ki0_slow = slow.Ki0(X1, X2, 0)
    np.testing.assert_allclose(Ki0_fast, Ki0_slow, rtol=1e-8, atol=1e-10)

    Kij_fast = fast.Kij(X1, X2, 0, 0)
    Kij_slow = slow.Kij(X1, X2, 0, 0)
    np.testing.assert_allclose(Kij_fast, Kij_slow, rtol=1e-8, atol=1e-10)


def test_eval_kLa_matches_posterior_kLas():
    posterior = _eval_only_posterior()
    adapter = LinPDEKernelAdapter(posterior)
    adapter._ensure_derivative_setup()

    X = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
    kLa_via_adapter = adapter._eval_kLa(X)  # (n, n_E)
    kLa_via_posterior = np.asarray(posterior._kLas(adapter._input_view(X))).reshape(
        X.shape[0], -1
    )
    np.testing.assert_allclose(kLa_via_adapter, kLa_via_posterior, rtol=1e-12)


# -----------------------------------------------------------------------------
# Test 3 — Ki0 FD self-consistency on the weak-form posterior.
#
# Verifies the new ∂_i kLa path is actually computing the derivative of kLa
# correctly by comparing adapter.Ki0 against a central FD of adapter.K.
# -----------------------------------------------------------------------------


def test_weak_form_adapter_Ki0_matches_FD_of_K():
    prior = _build_zero_mean_expquad(lengthscale=0.3)
    grid = np.linspace(0.0, 1.0, 9)
    test_basis = linpde_bases.UnivariateLinearInterpolationBasis(
        grid, zero_boundary=False
    )

    posterior = make_pde_equality_posterior(
        prior,
        test_basis,
        _f_rhs,
        X_bc=np.array([0.0, 1.0]),
        Y_bc=np.array([0.0, 1.0]),
        sigma_bc=1e-6,
    )

    adapter = LinPDEKernelAdapter(posterior)
    adapter._ensure_derivative_setup()
    assert adapter._eval_only is False, (
        "Posterior contains a CompositeLinearFunctional; eval_only flag should "
        "be False, forcing the per-L block path."
    )

    # Pick a few interior test points and one shared X2.
    X1 = np.array([[0.2], [0.4], [0.7]])
    X2 = np.array([[0.3], [0.55], [0.85]])
    h = 1e-4

    K_pp = adapter.K(X1 + np.array([[h]]), X2)
    K_mm = adapter.K(X1 - np.array([[h]]), X2)
    Ki0_fd = (K_pp - K_mm) / (2.0 * h)

    Ki0_an = adapter.Ki0(X1, X2, 0)

    # Hat-basis quad introduces small numerical wobble; allow a generous
    # tolerance. (1e-4 is comfortable; 1e-6 might fail spuriously.)
    np.testing.assert_allclose(Ki0_an, Ki0_fd, atol=1e-4, rtol=5e-3)
