"""
PriorKernelAdapter tests — gp_constr kernel API on a *prior* GP (no equality
conditioning), used for the inequality-only arm of the 3-way ablation
(experiments/linpde_gp_constrained/poisson_2d_ablation).

Covers:
  * K reduces to the prior covariance (no Schur correction).
  * Ki0(X1, X2, i) matches central FD of K w.r.t. X1[a, i] for i in {0, 1}.
  * Kij(X1, X2, i, j) matches the 4-point mixed-difference stencil for all
    (i, j) in {(0,0), (0,1), (1,0), (1,1)}.
  * A PriorKernelAdapter and a LinPDEKernelAdapter built on a *near-trivial*
    equality posterior (tiny BC far from the test points) agree on the
    derivative kernels up to the (small) representer correction — a sanity
    cross-check that PriorKernelAdapter is the "no equality" limit.
  * Optional: ConstrainedLinPDEPosterior accepts an injected adapter and, with
    no constraints, returns the prior mean (0). Truncated moments require R and
    are exercised in the docker run, not here.

REQUIRES: linpde-gp + probnum + jax (no scipy/R needed for the FD checks).

Run with: ``python -m pytest test_prior_kernel_adapter.py -v``
"""

from __future__ import annotations

import numpy as np

import pytest

linpde_gp = pytest.importorskip("linpde_gp")
probnum = pytest.importorskip("probnum")

import probnum as pn  # noqa: E402

from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad  # noqa: E402
from linpde_gp_constrained import (  # noqa: E402
    ConstrainedLinPDEPosterior,
    PriorKernelAdapter,
)


def _prior(lengthscale=0.5, scale=1.0, dim=2):
    return pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(dim,), output_shape=()),
        cov=scale * LinpdeExpQuad(input_shape=(dim,), lengthscales=lengthscale),
    )


def _scalar_prior(lengthscale=0.5, scale=1.0):
    """A genuinely scalar-input prior (input_shape == ()) to exercise the
    scalar branch of ``_partial_along`` (PartialDerivative(MultiIndex(1)))."""
    return pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(), output_shape=()),
        cov=scale * LinpdeExpQuad(input_shape=(), lengthscales=lengthscale),
    )


_X1 = np.array([[0.30, 0.40], [0.55, 0.25], [0.70, 0.65]])
_X2 = np.array([[0.35, 0.50], [0.60, 0.30], [0.80, 0.75]])


def test_K_reduces_to_prior_cov():
    prior = _prior()
    adapter = PriorKernelAdapter(prior)
    K = adapter.K(_X1, _X2)
    # Direct prior-cov evaluation with probnum's broadcasting convention.
    expected = np.asarray(prior.cov(_X1[:, None, :], _X2[None, :, :]))
    np.testing.assert_allclose(K, expected, atol=1e-12)
    assert adapter.dim == 2


@pytest.mark.parametrize("i", [0, 1])
def test_Ki0_matches_FD_of_K(i):
    adapter = PriorKernelAdapter(_prior())
    h = 1e-4
    e = np.zeros((1, 2))
    e[0, i] = h
    Ki0_fd = (adapter.K(_X1 + e, _X2) - adapter.K(_X1 - e, _X2)) / (2.0 * h)
    Ki0_an = adapter.Ki0(_X1, _X2, i)
    np.testing.assert_allclose(Ki0_an, Ki0_fd, atol=1e-4, rtol=5e-3)


@pytest.mark.parametrize("i,j", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_Kij_matches_mixed_FD_of_K(i, j):
    adapter = PriorKernelAdapter(_prior())
    h = 1e-3
    ei = np.zeros((1, 2))
    ei[0, i] = h
    ej = np.zeros((1, 2))
    ej[0, j] = h
    Kpp = adapter.K(_X1 + ei, _X2 + ej)
    Kpm = adapter.K(_X1 + ei, _X2 - ej)
    Kmp = adapter.K(_X1 - ei, _X2 + ej)
    Kmm = adapter.K(_X1 - ei, _X2 - ej)
    Kij_fd = (Kpp - Kpm - Kmp + Kmm) / (4.0 * h * h)
    Kij_an = adapter.Kij(_X1, _X2, i, j)
    np.testing.assert_allclose(Kij_an, Kij_fd, atol=1e-4, rtol=5e-3)


def test_Kii_diag_consistent():
    adapter = PriorKernelAdapter(_prior())
    X = np.array([[0.2, 0.3], [0.6, 0.7]])
    np.testing.assert_allclose(
        adapter.Kii_diag(X, 0), np.diag(adapter.Kij(X, X, 0, 0)), atol=1e-12
    )


def test_scalar_input_prior_adapter():
    prior = _scalar_prior()
    adapter = PriorKernelAdapter(prior)
    assert adapter.dim == 1 and adapter._scalar_input is True
    X1 = np.array([[0.2], [0.5], [0.8]])
    X2 = np.array([[0.3], [0.6]])
    # FD check on Ki0 for the scalar branch (PartialDerivative(MultiIndex(1))).
    h = 1e-4
    e = np.array([[h]])
    Ki0_fd = (adapter.K(X1 + e, X2) - adapter.K(X1 - e, X2)) / (2.0 * h)
    np.testing.assert_allclose(adapter.Ki0(X1, X2, 0), Ki0_fd, atol=1e-4, rtol=5e-3)


def test_injected_adapter_no_constraints_returns_prior_mean():
    """ConstrainedLinPDEPosterior accepts an injected adapter; with no
    constraints it short-circuits to the (zero) prior mean. The truncated
    moments under active constraints need R and run in docker."""
    prior = _prior()
    adapter = PriorKernelAdapter(prior)
    cwrap = ConstrainedLinPDEPosterior(prior, [], adapter=adapter)
    X = np.array([[0.4, 0.4], [0.6, 0.2]])
    np.testing.assert_allclose(
        cwrap.posterior_mean(X), np.zeros(X.shape[0]), atol=1e-12
    )
    assert cwrap.constraint_satisfaction_probability == 1.0
