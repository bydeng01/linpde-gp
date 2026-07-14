"""Acceptance test for Phase 8c: LMC as a JaxSum of ICM terms.

An LMC covariance ``sum_q B_q (x) k_q`` is just a ``JaxSumCovarianceFunction``
of the Phase 8a ICM terms. The Phase 7b sum-dispatch handlers already route
operators across that sum, so 8c adds *no* new dispatch — this test verifies
that the existing machinery suffices:

* a Q=2 LMC is a ``JaxSumCovarianceFunction``;
* the real2 Helmholtz operator can be applied (twice) with no
  ``JaxLambdaCovarianceFunction`` leaking and a finite Gram;
* the operator-conditioned LMC Gram equals the sum of the two operator-
  conditioned ICM Grams (the defining LMC linearity, Acceptance 8c).
"""
from __future__ import annotations

import numpy as np

from linpde_gp.linfuncops.diffops._helmholtz_operator import HelmholtzReal2Operator
from linpde_gp.randprocs.covfuncs import (
    CoregionalizedMultiOutputCovarianceFunction,
    JaxLambdaCovarianceFunction,
    JaxScaledCovarianceFunction,
    JaxSumCovarianceFunction,
    Matern,
    StackCovarianceFunction,
)


def _find_jax_lambdas(k, path=""):
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


def _build_lmc():
    k1 = Matern(input_shape=(3,), nu=2.5, lengthscales=0.05)
    k2 = Matern(input_shape=(3,), nu=2.5, lengthscales=0.20)
    B1 = np.array([[1.0, 0.5], [0.5, 1.2]])
    B2 = np.array([[0.7, -0.2], [-0.2, 0.9]])
    icm1 = CoregionalizedMultiOutputCovarianceFunction(k1, B1)
    icm2 = CoregionalizedMultiOutputCovarianceFunction(k2, B2)
    return icm1, icm2, icm1 + icm2


def test_lmc_is_jaxsum_of_icm():
    icm1, icm2, lmc = _build_lmc()
    assert isinstance(lmc, JaxSumCovarianceFunction)
    assert lmc.output_shape_0 == (2,)
    assert lmc.output_shape_1 == (2,)

    rng = np.random.default_rng(0)
    X0 = rng.uniform(0.0, 1.0, size=(5, 1, 3))
    X1 = rng.uniform(0.0, 1.0, size=(1, 4, 3))
    # Prior-level linearity (sanity).
    np.testing.assert_allclose(
        np.asarray(lmc(X0, X1)),
        np.asarray(icm1(X0, X1)) + np.asarray(icm2(X0, X1)),
        rtol=1e-12, atol=1e-14,
    )


def test_lmc_real2_operator_finite_and_additive():
    """Acceptance 8c: apply the real2 operator; Gram is finite, JaxLambda-free,
    and equals the sum of the two ICM Grams."""
    icm1, icm2, lmc = _build_lmc()
    op = HelmholtzReal2Operator(domain_shape=(3,), k_squared=4.0 + 0.5j)

    lmc_LL = op(op(lmc, argnum=1), argnum=0)
    icm1_LL = op(op(icm1, argnum=1), argnum=0)
    icm2_LL = op(op(icm2, argnum=1), argnum=0)

    assert lmc_LL.output_shape_0 == (2,)
    assert lmc_LL.output_shape_1 == (2,)
    assert _find_jax_lambdas(lmc_LL) == [], "LMC operator path leaked a JaxLambda"

    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(6, 3))

    G_lmc = np.asarray(lmc_LL(X[:, None, :], X[None, :, :]))
    G1 = np.asarray(icm1_LL(X[:, None, :], X[None, :, :]))
    G2 = np.asarray(icm2_LL(X[:, None, :], X[None, :, :]))

    assert G_lmc.shape == (6, 6, 2, 2)
    assert np.all(np.isfinite(G_lmc)), "LMC operator Gram has NaN/Inf"
    np.testing.assert_allclose(G_lmc, G1 + G2, rtol=1e-9, atol=1e-12)
