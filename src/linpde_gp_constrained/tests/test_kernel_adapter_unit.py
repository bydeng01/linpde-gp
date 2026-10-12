"""
Unit tests for LinPDEKernelAdapter — Layer A.

These tests do NOT require linpde-gp / probnum / jax to be installed. They
verify only the adapter's plumbing using a hand-rolled mock posterior whose
``cov`` callable directly evaluates an analytic RBF.

Tests covering the math of derivative kernels (Ki0 / Kij correctness) live in
``test_kernel_adapter_integration.py`` and require the full linpde-gp stack.

Mathematical claims tested here
-------------------------------
1. Shape adaptation: gp_constr's (n, d) inputs are routed to linpde-gp's
   expected shape (either (n,) for input_shape == () or (n, d) for (d,)).
2. K(X1, X2) returns shape (n1, n2) and matches the underlying cov callable.
3. K_diag(X) returns shape (n,) matching diag(K(X, X)).
4. dim attribute equals input dimensionality used by gp_constr.
5. Symmetry: K(X1, X2) = K(X2, X1).T (cov is a covariance function).
6. PSD on diagonal: K(X, X) is PSD up to numerical tolerance.
7. Optimization-path methods (set_params / get_params / K_gradients) raise
   NoHyperparamOptimization — regression guard against the two-stage rule.
"""

from linpde_gp_constrained.kernel_adapter import (
    LinPDEKernelAdapter,
    NoHyperparamOptimization,
)
import numpy as np

# -----------------------------------------------------------------------------
# Mock posterior. The kernel API consumed by LinPDEKernelAdapter is just a
# callable .cov(x0, x1) returning an array, plus .cov.input_shape. We mock
# this with an analytic RBF so unit tests can run without linpde-gp.
# -----------------------------------------------------------------------------


class _MockCov:
    """Mock that mimics linpde-gp's CovarianceFunction interface."""

    def __init__(self, input_shape, variance=1.0, lengthscale=1.0):
        self.input_shape = input_shape
        self.variance = float(variance)
        if input_shape == ():
            self.lengthscale = np.atleast_1d(np.asarray(lengthscale))
            assert self.lengthscale.shape == (1,)
        else:
            assert len(input_shape) == 1
            self.lengthscale = np.atleast_1d(np.asarray(lengthscale, dtype=float))
            assert self.lengthscale.shape == input_shape

    def __call__(self, x0, x1):
        """Mock probnum-style cov: broadcast batch dims, strip trailing input_shape.

        For input_shape == ()  : x0 shape (n1, 1), x1 shape (1, n2) -> (n1, n2)
        For input_shape == (d,): x0 shape (n1, 1, d), x1 shape (1, n2, d) -> (n1, n2)

        These match the shapes produced by LinPDEKernelAdapter._format_for_pair
        for gp_constr-style (n, d)-shaped inputs.
        """
        x0 = np.asarray(x0, dtype=float)
        x1 = np.asarray(x1, dtype=float)
        if self.input_shape == ():
            # Trailing input dim is 0 -> all of x0/x1 are batch dims.
            diff = x0 - x1  # broadcast (n1, 1) and (1, n2) -> (n1, n2)
            return self.variance * np.exp(-0.5 * (diff / self.lengthscale[0]) ** 2)
        else:
            # Trailing input dim is d -> strip it from batch shape.
            d = self.input_shape[0]
            assert x0.shape[-1] == d
            assert x1.shape[-1] == d
            diff = x0 - x1  # broadcast (n1, 1, d) and (1, n2, d) -> (n1, n2, d)
            sqd = np.sum((diff / self.lengthscale) ** 2, axis=-1)
            return self.variance * np.exp(-0.5 * sqd)


class _MockGP:
    def __init__(self, cov):
        self.cov = cov


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_dim_property_scalar_input():
    """dim equals 1 for input_shape=() (scalar input prior)."""
    gp = _MockGP(_MockCov(input_shape=(), lengthscale=0.7))
    a = LinPDEKernelAdapter(gp)
    assert a.dim == 1


def test_dim_property_vector_input():
    """dim equals d for input_shape=(d,)."""
    gp = _MockGP(_MockCov(input_shape=(3,), lengthscale=[0.5, 1.0, 2.0]))
    a = LinPDEKernelAdapter(gp)
    assert a.dim == 3


def _analytic_rbf(X1, X2, variance, lengthscale):
    """Independent analytic RBF — k(x, x') = variance * exp(-0.5 * sum((x-x')/ell)^2)."""
    diff = X1[:, None, :] - X2[None, :, :]
    sqd = np.sum((diff / lengthscale) ** 2, axis=-1)
    return variance * np.exp(-0.5 * sqd)


def test_K_shape_and_value_scalar():
    """K returns (n1, n2) and equals analytic RBF for input_shape=()."""
    rng = np.random.default_rng(0)
    variance, ell = 1.7, np.array([0.8])
    gp = _MockGP(_MockCov(input_shape=(), variance=variance, lengthscale=ell[0]))
    a = LinPDEKernelAdapter(gp)

    X1 = rng.uniform(-1, 1, (4, 1))  # gp_constr-style (n, 1)
    X2 = rng.uniform(-1, 1, (3, 1))

    K = a.K(X1, X2)
    assert K.shape == (4, 3)
    np.testing.assert_allclose(K, _analytic_rbf(X1, X2, variance, ell), atol=1e-15)


def test_K_shape_and_value_vector():
    """K returns (n1, n2) and equals analytic RBF for input_shape=(d,)."""
    rng = np.random.default_rng(1)
    variance, ell = 1.0, np.array([0.7, 1.3])
    gp = _MockGP(_MockCov(input_shape=(2,), variance=variance, lengthscale=ell))
    a = LinPDEKernelAdapter(gp)

    X1 = rng.uniform(-1, 1, (5, 2))
    X2 = rng.uniform(-1, 1, (3, 2))

    K = a.K(X1, X2)
    assert K.shape == (5, 3)
    np.testing.assert_allclose(K, _analytic_rbf(X1, X2, variance, ell), atol=1e-15)


def test_K_diag_shape():
    """K_diag returns shape (n,)."""
    rng = np.random.default_rng(2)
    cov = _MockCov(input_shape=(2,), variance=1.0, lengthscale=[1.0, 1.0])
    a = LinPDEKernelAdapter(_MockGP(cov))

    X = rng.uniform(-1, 1, (6, 2))
    d = a.K_diag(X)
    assert d.shape == (6,)
    # For RBF on the diagonal, k(x, x) = variance.
    np.testing.assert_allclose(d, 1.0, atol=1e-15)


def test_K_symmetric():
    """K(X1, X2) == K(X2, X1).T (covariance function symmetry)."""
    rng = np.random.default_rng(3)
    cov = _MockCov(input_shape=(2,), variance=2.5, lengthscale=[0.6, 0.9])
    a = LinPDEKernelAdapter(_MockGP(cov))

    X1 = rng.uniform(-1, 1, (5, 2))
    X2 = rng.uniform(-1, 1, (4, 2))

    np.testing.assert_allclose(a.K(X1, X2), a.K(X2, X1).T, atol=1e-14)


def test_K_diag_psd():
    """K(X, X) is PSD: minimum eigenvalue >= 0 modulo tolerance."""
    rng = np.random.default_rng(4)
    cov = _MockCov(input_shape=(), variance=1.0, lengthscale=0.5)
    a = LinPDEKernelAdapter(_MockGP(cov))

    X = rng.uniform(-1, 1, (8, 1))
    K = a.K(X, X)
    assert K.shape == (8, 8)
    K_sym = 0.5 * (K + K.T)
    eigvals = np.linalg.eigvalsh(K_sym)
    assert eigvals.min() > -1e-10, f"K(X,X) not PSD: min eigval = {eigvals.min()}"


def test_input_shape_validation():
    """Shape mismatch in (n, d) input must raise ValueError."""
    gp = _MockGP(_MockCov(input_shape=(2,), lengthscale=[1.0, 1.0]))
    a = LinPDEKernelAdapter(gp)

    bad = np.zeros((5, 3))  # d should be 2
    raised = False
    try:
        a.K(bad, bad)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for shape mismatch"


def test_set_params_raises():
    """Regression guard: set_params must raise."""
    gp = _MockGP(_MockCov(input_shape=()))
    a = LinPDEKernelAdapter(gp)
    raised = False
    try:
        a.set_params(np.array([1.0, 1.0]))
    except NoHyperparamOptimization:
        raised = True
    assert raised, "set_params must raise NoHyperparamOptimization"


def test_get_params_raises():
    """Regression guard: get_params must raise."""
    gp = _MockGP(_MockCov(input_shape=()))
    a = LinPDEKernelAdapter(gp)
    raised = False
    try:
        a.get_params()
    except NoHyperparamOptimization:
        raised = True
    assert raised


def test_K_gradients_raises():
    """Regression guard: K_gradients must raise."""
    gp = _MockGP(_MockCov(input_shape=()))
    a = LinPDEKernelAdapter(gp)
    raised = False
    try:
        a.K_gradients(np.zeros((3, 1)), np.zeros((4, 1)))
    except NoHyperparamOptimization:
        raised = True
    assert raised


def test_higher_input_shape_unsupported():
    """Multi-axis input shapes (e.g., (2, 3)) must raise NotImplementedError."""

    class _BadCov:
        input_shape = (2, 3)

        def __call__(self, x0, x1):
            raise NotImplementedError

    raised = False
    try:
        LinPDEKernelAdapter(_MockGP(_BadCov()))
    except NotImplementedError:
        raised = True
    assert raised
