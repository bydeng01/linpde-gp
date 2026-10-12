"""
Integration tests for LinPDEKernelAdapter — Layer B.

REQUIRES: linpde-gp, probnum, jax, numpy (the module is skipped if linpde-gp
or probnum is not importable).

Mathematical claims tested
--------------------------
On a real linpde-gp ConditionalGaussianProcess:

1. FD self-consistency for Ki0:
        adapter.Ki0(X1, X2, i)  ==  d/d(X1)_i  adapter.K(X1, X2)
   verified by central-difference of K with respect to the i-th component of
   the first argument. This is the *core* correctness check: it directly
   corresponds to the math claim "Ki0 is the partial derivative of K w.r.t.
   the first kernel argument's i-th component."

2. FD self-consistency for Kij:
        adapter.Kij(X1, X2, i, j) == d^2/d(X1)_i d(X2)_j  adapter.K(X1, X2)
   verified by 4-point mixed-difference stencil. Corresponds to the math
   claim about mixed second derivatives.

3. Sign convention: for X1_i > X2_i, ``Ki0(X1, X2, i) < 0`` on a strictly
   decreasing kernel. Catches sign-flip bugs.

4. Symmetry: ``K(X1, X2) == K(X2, X1).T`` numerically.

5. PSD: ``K(X, X)`` has nonnegative eigenvalues (modulo tolerance).

6. Comparison against gp_constr's analytic RBF on a *trivially-conditioned*
   posterior (one observation at a far-away point with high noise so its
   influence on the kernel is below tolerance). With the influence neutralized,
   the adapter should reproduce gp_constr's analytic Ki0 / Kij to high
   precision. This catches argument-convention mismatches.

7. Regression guards (set_params / get_params / K_gradients) — same as in
   the unit tests, but also runs in the integration environment.

Run with: ``python -m pytest test_kernel_adapter_integration.py -v``
"""

from __future__ import annotations

import numpy as np

import pytest

# Skip the entire module if linpde-gp is not importable.
linpde_gp = pytest.importorskip("linpde_gp")
probnum = pytest.importorskip("probnum")

from linpde_gp_constrained.kernel_adapter import (
    LinPDEKernelAdapter,
    NoHyperparamOptimization,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_trivial_posterior_1d(
    variance=1.0, lengthscale=0.5, far_point=100.0, far_noise=1e6
):
    """
    Build a 1D ConditionalGaussianProcess whose conditioning is essentially
    no-op: one observation at a far-away point with very high noise. The
    posterior kernel k_E ≈ prior kernel up to extremely tight tolerance,
    which lets us compare against gp_constr's analytic RBF.
    """
    import probnum as pn

    # IMPORTANT: use linpde_gp's ExpQuad, not probnum's. linpde_gp wraps
    # probnum's covariance functions in a JaxCovarianceFunction subclass; only
    # the linpde_gp version supports the .jax(x0, x1) method that the
    # PartialDerivative autodiff fallback path requires. probnum's
    # ScaledCovarianceFunction (returned by `variance * probnum.ExpQuad(...)`)
    # has no .jax attribute and breaks the autodiff chain.
    from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad

    prior = pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(), output_shape=()),
        cov=variance
        * LinpdeExpQuad(
            input_shape=(),
            lengthscales=lengthscale,
        ),
    )
    # Single high-noise observation at a far point.
    Y = np.array([0.0])
    X = np.array([far_point])
    posterior = prior.condition_on_observations(
        Y=Y,
        X=X,
        b=pn.randvars.Normal(np.zeros(1), np.eye(1) * far_noise),
    )
    return posterior


def _gpconstr_analytic_K(X1, X2, variance, lengthscale):
    """Analytic RBF k(x, x') = variance * exp(-0.5 * ((x - x') / ell)^2)."""
    diff = X1[:, None, :] - X2[None, :, :]
    sqd = np.sum((diff / lengthscale) ** 2, axis=-1)
    return variance * np.exp(-0.5 * sqd)


def _gpconstr_analytic_Ki0(X1, X2, variance, lengthscale, i):
    """Analytic d/d(X1)_i k = -k * (X1_i - X2_i) / ell_i^2."""
    K = _gpconstr_analytic_K(X1, X2, variance, lengthscale)
    return -K * (X1[:, i : i + 1] - X2[:, i : i + 1].T) / lengthscale[i] ** 2


def _gpconstr_analytic_Kij(X1, X2, variance, lengthscale, i, j):
    K = _gpconstr_analytic_K(X1, X2, variance, lengthscale)
    delta = 1.0 if i == j else 0.0
    cross = (X1[:, i : i + 1] - X2[:, i : i + 1].T) * (
        X1[:, j : j + 1] - X2[:, j : j + 1].T
    )
    return K * (
        delta / lengthscale[i] ** 2
        - cross / (lengthscale[i] ** 2 * lengthscale[j] ** 2)
    )


def _fd_Ki0(K_fn, X1, X2, i, h=1e-5):
    Xp = X1.copy()
    Xp[:, i] += h
    Xm = X1.copy()
    Xm[:, i] -= h
    return (K_fn(Xp, X2) - K_fn(Xm, X2)) / (2 * h)


def _fd_Kij(K_fn, X1, X2, i, j, h=1e-3):
    def Kpp():
        a = X1.copy()
        a[:, i] += h
        b = X2.copy()
        b[:, j] += h
        return K_fn(a, b)

    def Kpm():
        a = X1.copy()
        a[:, i] += h
        b = X2.copy()
        b[:, j] -= h
        return K_fn(a, b)

    def Kmp():
        a = X1.copy()
        a[:, i] -= h
        b = X2.copy()
        b[:, j] += h
        return K_fn(a, b)

    def Kmm():
        a = X1.copy()
        a[:, i] -= h
        b = X2.copy()
        b[:, j] -= h
        return K_fn(a, b)

    return (Kpp() - Kpm() - Kmp() + Kmm()) / (4 * h * h)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_K_shape_and_symmetry_real():
    rng = np.random.default_rng(0)
    posterior = _make_trivial_posterior_1d()
    adapter = LinPDEKernelAdapter(posterior)
    assert adapter.dim == 1

    X1 = rng.uniform(-1, 1, (5, 1))
    X2 = rng.uniform(-1, 1, (4, 1))
    K12 = adapter.K(X1, X2)
    K21 = adapter.K(X2, X1)

    assert K12.shape == (5, 4)
    assert K21.shape == (4, 5)
    np.testing.assert_allclose(K12, K21.T, atol=1e-12)


def test_K_diagonal_PSD_real():
    rng = np.random.default_rng(0)
    posterior = _make_trivial_posterior_1d()
    adapter = LinPDEKernelAdapter(posterior)

    X = rng.uniform(-1, 1, (6, 1))
    K = adapter.K(X, X)
    assert K.shape == (6, 6)
    K_sym = 0.5 * (K + K.T)
    eigvals = np.linalg.eigvalsh(K_sym)
    assert eigvals.min() > -1e-10


def test_Ki0_vs_finite_difference():
    """Ki0 must equal the finite difference of K w.r.t. the FIRST arg."""
    rng = np.random.default_rng(1)
    posterior = _make_trivial_posterior_1d(lengthscale=0.7)
    adapter = LinPDEKernelAdapter(posterior)

    X1 = rng.uniform(-1, 1, (4, 1))
    X2 = rng.uniform(-1, 1, (3, 1))

    Ki0 = adapter.Ki0(X1, X2, 0)
    Ki0_fd = _fd_Ki0(adapter.K, X1, X2, 0)

    err = np.max(np.abs(Ki0 - Ki0_fd))
    assert err < 1e-6, f"Ki0 vs FD disagreement: {err}"
    assert Ki0.shape == (4, 3)


def test_Kij_vs_finite_difference():
    """Kij must equal the mixed FD: d^2/d(X1)_i d(X2)_j K."""
    rng = np.random.default_rng(2)
    posterior = _make_trivial_posterior_1d(lengthscale=0.7)
    adapter = LinPDEKernelAdapter(posterior)

    X1 = rng.uniform(-1, 1, (4, 1))
    X2 = rng.uniform(-1, 1, (3, 1))

    Kij = adapter.Kij(X1, X2, 0, 0)
    Kij_fd = _fd_Kij(adapter.K, X1, X2, 0, 0)

    err = np.max(np.abs(Kij - Kij_fd))
    assert err < 1e-4, f"Kij vs FD disagreement: {err}"
    assert Kij.shape == (4, 3)


def test_Ki0_sign_convention():
    """Ki0(X1, X2, 0) < 0 when X1_0 > X2_0 for an RBF prior."""
    posterior = _make_trivial_posterior_1d()
    adapter = LinPDEKernelAdapter(posterior)

    X1 = np.array([[0.5]])
    X2 = np.array([[0.0]])
    Ki0 = adapter.Ki0(X1, X2, 0)
    assert Ki0[0, 0] < 0, f"Sign convention violated: Ki0={Ki0}"

    X1 = np.array([[-0.5]])
    Ki0 = adapter.Ki0(X1, X2, 0)
    assert Ki0[0, 0] > 0


def test_adapter_matches_analytic_RBF_on_trivial_posterior():
    """
    Cross-check: with conditioning effectively neutralized (far point + high
    noise), the adapter should reproduce gp_constr's analytic RBF Ki0 / Kij to
    high precision. This catches argument-position bugs that finite-difference
    self-consistency would miss (e.g., differentiating the SECOND arg twice).
    """
    rng = np.random.default_rng(3)
    variance = 1.0
    ell = np.array([0.7])
    posterior = _make_trivial_posterior_1d(variance=variance, lengthscale=ell[0])
    adapter = LinPDEKernelAdapter(posterior)

    X1 = rng.uniform(-1, 1, (4, 1))
    X2 = rng.uniform(-1, 1, (3, 1))

    K_an = _gpconstr_analytic_K(X1, X2, variance, ell)
    Ki0_an = _gpconstr_analytic_Ki0(X1, X2, variance, ell, 0)
    Kij_an = _gpconstr_analytic_Kij(X1, X2, variance, ell, 0, 0)

    np.testing.assert_allclose(adapter.K(X1, X2), K_an, atol=1e-7)
    np.testing.assert_allclose(adapter.Ki0(X1, X2, 0), Ki0_an, atol=1e-7)
    np.testing.assert_allclose(adapter.Kij(X1, X2, 0, 0), Kij_an, atol=1e-6)


def test_Kii_diag_against_diag_of_Kij():
    """Kii_diag(X, i) == diag(Kij(X, X, i, i))."""
    rng = np.random.default_rng(4)
    posterior = _make_trivial_posterior_1d(lengthscale=0.5)
    adapter = LinPDEKernelAdapter(posterior)

    X = rng.uniform(-1, 1, (5, 1))
    diag_via_Kij = np.diag(adapter.Kij(X, X, 0, 0))
    np.testing.assert_allclose(adapter.Kii_diag(X, 0), diag_via_Kij, atol=1e-12)


def test_set_params_raises_real():
    posterior = _make_trivial_posterior_1d()
    adapter = LinPDEKernelAdapter(posterior)
    with pytest.raises(NoHyperparamOptimization):
        adapter.set_params(np.array([1.0, 1.0]))


def test_get_params_raises_real():
    posterior = _make_trivial_posterior_1d()
    adapter = LinPDEKernelAdapter(posterior)
    with pytest.raises(NoHyperparamOptimization):
        adapter.get_params()


def test_K_gradients_raises_real():
    posterior = _make_trivial_posterior_1d()
    adapter = LinPDEKernelAdapter(posterior)
    with pytest.raises(NoHyperparamOptimization):
        adapter.K_gradients(np.zeros((3, 1)), np.zeros((4, 1)))
