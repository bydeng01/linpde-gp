"""Regression tests for Phase 7a/7b.

Cover two pieces of the variable-Helmholtz 2-component pipeline:

* ``IndependentMultiOutputCovarianceFunction`` plugs cleanly into
  ``pn.randprocs.GaussianProcess`` with a 2-vector mean, and ``prior.var``
  returns the diagonal as a ``(N, 2)`` array matching the underlying scalar
  kernel evaluated pointwise.

* Applying ``HelmholtzReal2Operator`` twice (argnum=1 then argnum=0) to a
  2-component prior built from ``IndependentMultiOutputCovarianceFunction``
  produces a fully *structured* covariance: no
  ``JaxLambdaCovarianceFunction`` is ever introduced, the Gram diagonal is
  finite, and the Gram matrix is symmetric under simultaneous swap of
  ``(x, y)`` and the output axes. Without the Phase 7b dispatch additions
  in :mod:`linpde_gp.randprocs.covfuncs.linfuncops` the chain falls back to
  ``JaxLambdaCovarianceFunction`` and the diagonal NaN-s out (the same
  failure mode that motivated the closed-form weighted-Laplacian work in
  Phase 6a).
"""
from __future__ import annotations

import numpy as np
import probnum as pn

import pytest

import linpde_gp
from linpde_gp import functions as lp_functions
from linpde_gp.linfuncops.diffops._helmholtz_operator import HelmholtzReal2Operator
from linpde_gp.randprocs.covfuncs import (
    IndependentMultiOutputCovarianceFunction,
    JaxLambdaCovarianceFunction,
    JaxScaledCovarianceFunction,
    JaxSumCovarianceFunction,
    Matern,
    StackCovarianceFunction,
    Zero,
)


# --------------------------------------------------------------------------- #
# Phase 7a — GP wiring                                                        #
# --------------------------------------------------------------------------- #


def _find_jax_lambdas(k, path=""):
    """Recursively collect every JaxLambdaCovarianceFunction in a structured tree."""
    found = []
    if isinstance(k, JaxLambdaCovarianceFunction):
        found.append(path or "<root>")
    if isinstance(k, StackCovarianceFunction):
        for i, c in enumerate(np.asarray(k.covfuncs).ravel()):
            found.extend(_find_jax_lambdas(c, path + f"/stack[{i}]"))
    if isinstance(k, JaxSumCovarianceFunction):
        for i, s in enumerate(k.summands):
            found.extend(_find_jax_lambdas(s, path + f"/sum[{i}]"))
    if isinstance(k, JaxScaledCovarianceFunction):
        found.extend(_find_jax_lambdas(k.covfunc, path + "/scaled"))
    return found


def test_independent_multi_output_shape_and_off_diagonals():
    """Block-diagonal structure on the output axes for k · I_2."""
    matern = Matern(input_shape=(3,), nu=2.5, lengthscales=0.05)
    mo = IndependentMultiOutputCovarianceFunction(matern, matern)

    assert mo.output_shape_0 == (2,)
    assert mo.output_shape_1 == (2,)

    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(6, 3))

    K = np.asarray(mo(X[:, None, :], X[None, :, :]))
    assert K.shape == (6, 6, 2, 2)

    scalar_K = np.asarray(matern(X[:, None, :], X[None, :, :]))
    np.testing.assert_allclose(K[..., 0, 0], scalar_K)
    np.testing.assert_allclose(K[..., 1, 1], scalar_K)
    np.testing.assert_allclose(K[..., 0, 1], 0.0)
    np.testing.assert_allclose(K[..., 1, 0], 0.0)


def test_independent_multi_output_in_gaussian_process():
    """Prior built from the multi-output covariance produces a (N, 2) variance map."""
    matern = Matern(input_shape=(3,), nu=2.5, lengthscales=0.05)
    mo = IndependentMultiOutputCovarianceFunction(matern, matern)

    prior = pn.randprocs.GaussianProcess(
        mean=lp_functions.Zero(input_shape=(3,), output_shape=(2,)),
        cov=mo,
    )
    assert prior.output_shape == (2,)

    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(7, 3))

    var = prior.var(X)
    assert var.shape == (7, 2)

    # Both output channels share the scalar Matern's pointwise variance.
    scalar_var = np.asarray(matern(X, X))
    np.testing.assert_allclose(var[:, 0], scalar_var)
    np.testing.assert_allclose(var[:, 1], scalar_var)


# --------------------------------------------------------------------------- #
# Phase 7b — structured dispatch on HelmholtzReal2Operator                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real2_setup_3d():
    matern = Matern(input_shape=(3,), nu=2.5, lengthscales=0.05)
    prior_cov = IndependentMultiOutputCovarianceFunction(matern, matern)
    op = HelmholtzReal2Operator(domain_shape=(3,), k_squared=4.0 + 0.5j)
    return matern, prior_cov, op


def test_helmholtz_real2_single_application_is_structured(real2_setup_3d):
    """One application of HelmholtzReal2Operator preserves structure."""
    _, prior_cov, op = real2_setup_3d
    kL = op(prior_cov, argnum=1)
    assert kL.output_shape_0 == (2,)
    assert kL.output_shape_1 == (2,)
    lambdas = _find_jax_lambdas(kL)
    assert lambdas == [], f"Unexpected JaxLambdas after one application: {lambdas}"


def test_helmholtz_real2_double_application_is_structured(real2_setup_3d):
    """Two applications (argnum=1 then argnum=0) remain JaxLambda-free."""
    _, prior_cov, op = real2_setup_3d
    kL = op(prior_cov, argnum=1)
    kL0L1 = op(kL, argnum=0)
    assert kL0L1.output_shape_0 == (2,)
    assert kL0L1.output_shape_1 == (2,)
    lambdas = _find_jax_lambdas(kL0L1)
    assert lambdas == [], (
        "Phase 7b dispatch leaked a JaxLambdaCovarianceFunction: " + ", ".join(lambdas)
    )


def test_helmholtz_real2_gram_diagonal_finite(real2_setup_3d):
    """The full Gram has no NaN/Inf entries."""
    _, prior_cov, op = real2_setup_3d
    kL = op(prior_cov, argnum=1)
    kL0L1 = op(kL, argnum=0)

    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, size=(6, 3))

    diag = np.asarray(kL0L1(X, X))
    assert diag.shape == (6, 2, 2)
    assert np.all(np.isfinite(diag)), "kL0L1 diagonal contains NaN/Inf"

    G = np.asarray(kL0L1(X[:, None, :], X[None, :, :]))
    assert G.shape == (6, 6, 2, 2)
    assert np.all(np.isfinite(G)), "kL0L1 Gram contains NaN/Inf"


def test_helmholtz_real2_gram_symmetric(real2_setup_3d):
    """K(x, y)[i, j, a, b] == K(y, x)[j, i, b, a] (joint (x, y) ↔ (a, b) swap)."""
    _, prior_cov, op = real2_setup_3d
    kL = op(prior_cov, argnum=1)
    kL0L1 = op(kL, argnum=0)

    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, size=(5, 3))

    G = np.asarray(kL0L1(X[:, None, :], X[None, :, :]))
    G_swapped = G.transpose(1, 0, 3, 2)
    abs_max = float(np.max(np.abs(G)))
    abs_err = float(np.max(np.abs(G - G_swapped)))
    # Exact symmetry expected for the closed-form path; allow a tiny rel tolerance
    # for floating-point reductions in the JAX summation order.
    assert abs_err <= 1e-9 * max(abs_max, 1.0), (
        f"Gram symmetry violated: abs_err={abs_err:.4g}, abs_max={abs_max:.4g}"
    )


def test_helmholtz_real2_variable_coefficient_double_application_is_structured():
    """The from_coefficient_field path stays structured too."""
    domain_shape = (3,)
    matern = Matern(input_shape=domain_shape, nu=2.5, lengthscales=0.05)
    prior_cov = IndependentMultiOutputCovarianceFunction(matern, matern)

    def k2(x):
        # Smooth complex k²(x) with non-trivial dissipation. Use jax-compatible ops.
        from jax import numpy as jnp
        r2 = jnp.sum(x ** 2, axis=-1)
        return (4.0 + 12.0 * r2) - 1j * (1.0 + 2.0 * r2)

    k2_field = lp_functions.JaxLambdaFunction(
        k2, input_shape=domain_shape, output_shape=(), vectorize=True
    )
    op = HelmholtzReal2Operator.from_coefficient_field(
        domain_shape=domain_shape, k_squared_field=k2_field
    )

    kL = op(prior_cov, argnum=1)
    kL0L1 = op(kL, argnum=0)

    lambdas = _find_jax_lambdas(kL0L1)
    assert lambdas == [], (
        "Variable-coefficient HelmholtzReal2Operator leaked JaxLambda: " + ", ".join(lambdas)
    )

    rng = np.random.default_rng(4)
    X = rng.uniform(0.0, 1.0, size=(5, 3))
    diag = np.asarray(kL0L1(X, X))
    assert diag.shape == (5, 2, 2)
    assert np.all(np.isfinite(diag)), "Variable-coefficient kL0L1 diagonal has NaN/Inf"
