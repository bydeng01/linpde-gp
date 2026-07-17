"""Phase 6a regression tests.

Explicit finite-difference verification for the multi-D half-integer Matern
Laplacian closed forms (``HalfIntegerMatern_Identity_WeightedLaplacian`` and
``HalfIntegerMatern_WeightedLaplacian_WeightedLaplacian``).

These complement the parametrized cases in ``cases_matern.py`` by testing two
properties the parametrized framework does not exercise directly:

  1. The closed-form is finite at the diagonal :math:`x_0 = x_1` for the
     multi-D double-Laplacian path. (The generic JAX hessian fallback returns
     NaN here for Matern in :math:`d > 1` because of the ``sqrt(|x|^2)``
     non-smoothness.)
  2. The closed-form matches a 4th-order central-difference reference at
     several non-zero separations.
"""

import numpy as np

import pytest

from linpde_gp.linfuncops import diffops
from linpde_gp.randprocs import covfuncs
from linpde_gp.randprocs.covfuncs.linfuncops import diffops as covfuncs_diffops


def _weighted_laplace_fd(f, x, weights, h):
    """Weighted Laplacian via 2nd-order central differences."""
    out = 0.0
    fx0 = f(x)
    for i in range(x.shape[0]):
        e = np.zeros(x.shape[0])
        e[i] = h
        out += weights[i] * (f(x + e) - 2.0 * fx0 + f(x - e)) / h**2
    return out


@pytest.mark.parametrize("nu", [2.5, 3.5])
def test_identity_weighted_laplacian_fd_matches(nu):
    rng = np.random.default_rng(2025_06_03)
    lengthscales = np.array([1.0, 2.0, 3.0])
    weights = rng.standard_normal(3)

    k = covfuncs.Matern((3,), nu=nu, lengthscales=lengthscales)
    L = diffops.WeightedLaplacian(weights)
    kL = L(k, argnum=1)

    assert isinstance(kL, covfuncs_diffops.HalfIntegerMatern_Identity_WeightedLaplacian)

    # FD at several point pairs with non-trivial separations.
    h = 2e-3  # FD step
    atol = 1e-4
    for trial in range(6):
        x0 = rng.uniform(-1.0, 1.0, size=3)
        x1 = x0 + rng.uniform(-1.5, 1.5, size=3)
        closed = float(kL(x0, x1))
        fd = _weighted_laplace_fd(lambda y: float(k(x0, y)), x1, weights, h)
        assert np.isfinite(closed)
        assert (
            abs(closed - fd) < atol
        ), f"trial {trial}: |closed-fd|={abs(closed-fd):.3e} > {atol:.3e}"


@pytest.mark.parametrize("nu", [2.5, 3.5])
def test_double_weighted_laplacian_fd_matches(nu):
    rng = np.random.default_rng(2025_06_04)
    lengthscales = np.array([1.0, 1.5, 2.0])
    w0 = rng.standard_normal(3)
    w1 = rng.standard_normal(3)

    k = covfuncs.Matern((3,), nu=nu, lengthscales=lengthscales)
    L0 = diffops.WeightedLaplacian(w0)
    L1 = diffops.WeightedLaplacian(w1)
    kL0L1 = L0(L1(k, argnum=1), argnum=0)

    assert isinstance(
        kL0L1, covfuncs_diffops.HalfIntegerMatern_WeightedLaplacian_WeightedLaplacian
    )

    h = 5e-3  # 4th-order FD truncation dominates
    atol = 5e-3
    for trial in range(6):
        x0 = rng.uniform(-1.0, 1.0, size=3)
        x1 = x0 + rng.uniform(-2.0, 2.0, size=3) * 0.5
        closed = float(kL0L1(x0, x1))

        def L1_k_of_xb(xa, xb):
            return _weighted_laplace_fd(lambda y: float(k(xa, y)), xb, w1, h)

        fd = _weighted_laplace_fd(lambda x: L1_k_of_xb(x, x1), x0, w0, h)
        assert np.isfinite(closed)
        assert (
            abs(closed - fd) < atol
        ), f"trial {trial}: |closed-fd|={abs(closed-fd):.3e} > {atol:.3e}"


@pytest.mark.parametrize("nu", [2.5, 3.5])
def test_double_weighted_laplacian_finite_at_diagonal(nu):
    """Phase 6a's headline fix: the JAX hessian fallback NaNs out at r=0
    because of sqrt(|x|^2) non-smoothness for Matern in d > 1. The new
    closed form must give a finite, analytical value at the diagonal."""
    rng = np.random.default_rng(2025_06_05)
    lengthscales = np.array([1.0, 0.5, 2.0])
    w0 = np.array([1.0, 1.0, 1.0])
    w1 = np.array([0.5, 1.5, 0.8])

    k = covfuncs.Matern((3,), nu=nu, lengthscales=lengthscales)
    L0 = diffops.WeightedLaplacian(w0)
    L1 = diffops.WeightedLaplacian(w1)
    kL0L1 = L0(L1(k, argnum=1), argnum=0)

    # Closed-form diagonal value: (2 M + W_0 W_1) * tilde_p_0(0)
    s = k._scale_factors  # = sqrt(2 nu)/lengthscale  (per-dim)
    s2 = s * s
    bar_w0 = w0 * s2
    bar_w1 = w1 * s2
    W0 = float(np.sum(bar_w0))
    W1 = float(np.sum(bar_w1))
    M = float(np.sum(bar_w0 * bar_w1))

    # tilde_p_0(0): 1/3 for Matern 5/2, 1/15 for Matern 7/2
    tp0_at_zero = {2.5: 1.0 / 3.0, 3.5: 1.0 / 15.0}[nu]
    expected = (2.0 * M + W0 * W1) * tp0_at_zero

    for x in (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, -0.3, 0.1]),
        np.array([1.2, 0.7, -0.4]),
    ):
        val = float(kL0L1(x, x))
        assert np.isfinite(val), f"Diagonal value at {x} is not finite: {val}"
        np.testing.assert_allclose(
            val,
            expected,
            rtol=1e-12,
            err_msg=f"Diagonal value at {x} does not match analytical limit",
        )


def test_double_weighted_laplacian_gram_diagonal_no_nan():
    """Pairwise Gram matrix: diagonal entries (where rows/cols coincide)
    must be finite, and the matrix must be symmetric for L0 == L1."""
    rng = np.random.default_rng(2025_06_06)

    k = covfuncs.Matern((3,), nu=2.5, lengthscales=np.array([1.0, 2.0, 1.5]))
    w = np.array([1.0, 0.7, 1.3])
    L = diffops.WeightedLaplacian(w)
    kLL = L(L(k, argnum=1), argnum=0)

    X = rng.standard_normal((10, 3))
    G = np.asarray(kLL(X[:, None], X[None, :]))

    assert np.all(np.isfinite(G)), "Gram matrix has NaN/Inf entries"
    assert np.max(np.abs(G - G.T)) < 1e-12, "Gram matrix is not symmetric"

    # Add a duplicated point to force a diagonal-of-block in the Gram (r=0 row)
    Xd = np.vstack([X, X[0:1]])  # row 0 and row -1 are at the same location
    Gd = np.asarray(kLL(Xd[:, None], Xd[None, :]))
    np.testing.assert_allclose(Gd[0, -1], Gd[0, 0], rtol=1e-12)
    np.testing.assert_allclose(Gd[-1, -1], Gd[0, 0], rtol=1e-12)
