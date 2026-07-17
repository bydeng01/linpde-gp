"""Regression tests for Phase 8b: structured dispatch on an ICM prior.

Parallels ``test_helmholtz_real2_multi_output_dispatch.py`` but builds the
2-component prior from a :class:`CoregionalizedMultiOutputCovarianceFunction`
with a *non-trivial* off-diagonal ``B`` (so the Re/Im channels are coupled in
the prior, not only through the operator). Verifies that applying
``HelmholtzReal2Operator`` once and twice to the ICM prior stays fully
structured: no ``JaxLambdaCovarianceFunction`` is ever introduced (the
coregionalization matrix ``B`` rides along as scale factors that reach the base
kernel's closed-form handlers), the Gram is finite, and joint-swap symmetry
holds. The variable-coefficient (``CoefficientFieldOperator``) path is checked
too.
"""

from __future__ import annotations

import numpy as np

import pytest

import linpde_gp
from linpde_gp import functions as lp_functions
from linpde_gp.linfuncops import SelectOutput
from linpde_gp.linfuncops.diffops._helmholtz_operator import HelmholtzReal2Operator
from linpde_gp.randprocs.covfuncs import (
    CoregionalizedMultiOutputCovarianceFunction,
    JaxLambdaCovarianceFunction,
    JaxScaledCovarianceFunction,
    JaxSumCovarianceFunction,
    Matern,
    StackCovarianceFunction,
    Zero,
)


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


@pytest.fixture(scope="module")
def icm_setup_3d():
    matern = Matern(input_shape=(3,), nu=2.5, lengthscales=0.05)
    # Non-trivial coregionalization: off-diagonal coupling between Re and Im.
    B = np.array([[1.0, 0.6], [0.6, 1.3]])
    prior_cov = CoregionalizedMultiOutputCovarianceFunction(matern, B)
    op = HelmholtzReal2Operator(domain_shape=(3,), k_squared=4.0 + 0.5j)
    return matern, B, prior_cov, op


# --------------------------------------------------------------------------- #
# SelectOutput structural behaviour                                           #
# --------------------------------------------------------------------------- #


def test_select_output_on_icm_is_scaled_base(icm_setup_3d):
    """SelectOutput on ICM yields a Stack of B[idx, c] * base (no JaxLambda)."""
    matern, B, prior_cov, _ = icm_setup_3d
    for argnum in (0, 1):
        for idx in (0, 1):
            sel = SelectOutput(((3,), (2,)), idx)(prior_cov, argnum=argnum)
            assert isinstance(sel, StackCovarianceFunction)
            assert _find_jax_lambdas(sel) == []

            rng = np.random.default_rng(idx)
            X = rng.uniform(0.0, 1.0, size=(4, 3))
            scalar_K = np.asarray(matern(X[:, None, :], X[None, :, :]))
            sel_K = np.asarray(sel(X[:, None, :], X[None, :, :]))
            # Component c equals B[idx, c] * base.
            for c in range(2):
                comp = sel_K[..., c] if sel.output_idx == 1 else sel_K[..., c]
                np.testing.assert_allclose(comp, B[idx, c] * scalar_K)


# --------------------------------------------------------------------------- #
# HelmholtzReal2Operator on the ICM prior                                     #
# --------------------------------------------------------------------------- #


def test_icm_single_application_is_structured(icm_setup_3d):
    _, _, prior_cov, op = icm_setup_3d
    kL = op(prior_cov, argnum=1)
    assert kL.output_shape_0 == (2,)
    assert kL.output_shape_1 == (2,)
    lambdas = _find_jax_lambdas(kL)
    assert lambdas == [], f"Unexpected JaxLambdas after one application: {lambdas}"


def test_icm_double_application_is_structured(icm_setup_3d):
    _, _, prior_cov, op = icm_setup_3d
    kL = op(prior_cov, argnum=1)
    kL0L1 = op(kL, argnum=0)
    assert kL0L1.output_shape_0 == (2,)
    assert kL0L1.output_shape_1 == (2,)
    lambdas = _find_jax_lambdas(kL0L1)
    assert (
        lambdas == []
    ), "Phase 8b dispatch leaked a JaxLambdaCovarianceFunction: " + ", ".join(lambdas)


def test_icm_gram_diagonal_finite(icm_setup_3d):
    _, _, prior_cov, op = icm_setup_3d
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


def test_icm_gram_symmetric(icm_setup_3d):
    """K(x, y)[i, j, a, b] == K(y, x)[j, i, b, a] (joint (x, y) <-> (a, b) swap)."""
    _, _, prior_cov, op = icm_setup_3d
    kL = op(prior_cov, argnum=1)
    kL0L1 = op(kL, argnum=0)

    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, size=(5, 3))

    G = np.asarray(kL0L1(X[:, None, :], X[None, :, :]))
    G_swapped = G.transpose(1, 0, 3, 2)
    abs_max = float(np.max(np.abs(G)))
    abs_err = float(np.max(np.abs(G - G_swapped)))
    assert abs_err <= 1e-9 * max(
        abs_max, 1.0
    ), f"Gram symmetry violated: abs_err={abs_err:.4g}, abs_max={abs_max:.4g}"


def test_icm_variable_coefficient_double_application_is_structured():
    """The from_coefficient_field path stays structured on the ICM prior too."""
    domain_shape = (3,)
    matern = Matern(input_shape=domain_shape, nu=2.5, lengthscales=0.05)
    B = np.array([[1.0, 0.6], [0.6, 1.3]])
    prior_cov = CoregionalizedMultiOutputCovarianceFunction(matern, B)

    def k2(x):
        from jax import numpy as jnp

        r2 = jnp.sum(x**2, axis=-1)
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
        "Variable-coefficient ICM HelmholtzReal2Operator leaked JaxLambda: "
        + ", ".join(lambdas)
    )

    rng = np.random.default_rng(4)
    X = rng.uniform(0.0, 1.0, size=(5, 3))
    diag = np.asarray(kL0L1(X, X))
    assert diag.shape == (5, 2, 2)
    assert np.all(
        np.isfinite(diag)
    ), "Variable-coefficient ICM kL0L1 diagonal has NaN/Inf"


def test_icm_reduces_to_iid_under_operator():
    """With B = I, the operator-conditioned ICM Gram matches the IID prior's."""
    from linpde_gp.randprocs.covfuncs import IndependentMultiOutputCovarianceFunction

    domain_shape = (3,)
    matern = Matern(input_shape=domain_shape, nu=2.5, lengthscales=0.05)
    op = HelmholtzReal2Operator(domain_shape=domain_shape, k_squared=4.0 + 0.5j)

    icm = CoregionalizedMultiOutputCovarianceFunction(matern, np.eye(2))
    iid = IndependentMultiOutputCovarianceFunction(matern, matern)

    icm_LL = op(op(icm, argnum=1), argnum=0)
    iid_LL = op(op(iid, argnum=1), argnum=0)

    rng = np.random.default_rng(7)
    X = rng.uniform(0.0, 1.0, size=(5, 3))
    G_icm = np.asarray(icm_LL(X[:, None, :], X[None, :, :]))
    G_iid = np.asarray(iid_LL(X[:, None, :], X[None, :, :]))
    np.testing.assert_allclose(G_icm, G_iid, rtol=1e-10, atol=1e-12)
