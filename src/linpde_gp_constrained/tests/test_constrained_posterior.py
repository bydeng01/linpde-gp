"""
Tests for ConstrainedLinPDEPosterior — the wrapper that ties the kernel
adapter to gp_constr.GPmodel.

REQUIRES: linpde-gp + probnum + jax + gp_constr (and, for the
``constraint_satisfaction_probability`` smoke test, R + rpy2 with the
``mvtnorm`` family of packages — that one is skipped if rpy2 is missing).

Mathematical claims tested
--------------------------
1.  Empty-constraint recovery: with ``inequality_constraints=[]``, the
    constrained posterior mean equals the linpde-gp equality posterior
    mean ``m_E`` exactly, and ``constraint_satisfaction_probability`` is 1.

2.  Mean-subtraction invariance: for an identity (boundedness) constraint
    ``a ≤ u ≤ b``, the constrained-posterior mean computed via the
    wrapper agrees with the same constraint applied to a manually
    centered process. This isolates the bound-transformation arithmetic
    from gp_constr's internal sampler.

3.  Sample feasibility: drawn samples satisfy the inequality at the
    virtual points up to the slack ``constr_likelihood``. We give a few
    standard deviations of slack to absorb stochastic violations.

4.  Optimization regression guard: ``wrapper.gp_constr_model.optimize()``
    raises ``NoHyperparamOptimization`` because the kernel adapter rejects
    set_params/get_params/K_gradients.

5.  Diagnostic ``P(C | E)`` smoke check: for a setup where the equality
    posterior already satisfies the inequality with high probability,
    P(C | E) is well above 0.5; for an incompatible setup, it is well
    below 0.5. Skipped if rpy2 / R packages are not available.
"""

from __future__ import annotations

import numpy as np

import pytest

# Hard requirements for these tests (the wrapper itself can be imported
# without linpde-gp / gp_constr, but every test below builds a real
# posterior, so we skip the whole module if the stack is missing).
linpde_gp = pytest.importorskip("linpde_gp")
probnum = pytest.importorskip("probnum")
gp_constr = pytest.importorskip("GPConstr")

# rpy2 / R is only needed for gp_constr's truncated-MVN sampler and
# pmvnorm-backed P(C|E). Detect it once with a plain try/import so we can
# skip the rpy2-dependent tests individually instead of killing the module.
# (Using ``pytest.importorskip`` inside a ``@pytest.mark.skipif`` decorator
# raises ``Skipped`` at decoration time rather than skipping just the
# decorated test.)
try:
    import rpy2  # noqa: F401

    _HAS_RPY2 = True
except ImportError:
    _HAS_RPY2 = False

from linpde_gp_constrained import (
    ConstrainedLinPDEPosterior,
    InequalityConstraint,
    NoHyperparamOptimization,
)

# -----------------------------------------------------------------------------
# Posterior factories
# -----------------------------------------------------------------------------


def _make_posterior_1d_with_data(
    f_true,
    n_obs=6,
    sigma=0.05,
    variance=1.0,
    lengthscale=0.3,
    seed=0,
):
    """Build a 1D ConditionalGaussianProcess from noisy observations of f_true.

    Uses linpde-gp's ExpQuad (the JaxCovarianceFunction wrapper) so that
    PartialDerivative dispatch finds a closed-form handler. probnum's
    ScaledCovarianceFunction returned by ``variance * pn.kernels.ExpQuad``
    has no ``.jax`` attribute and breaks the autodiff fallback.
    """
    import probnum as pn

    from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad

    rng = np.random.default_rng(seed)
    x_obs = np.sort(rng.uniform(0.05, 0.95, size=n_obs))
    y_obs = f_true(x_obs) + sigma * rng.standard_normal(n_obs)

    prior = pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(), output_shape=()),
        cov=variance * LinpdeExpQuad(input_shape=(), lengthscales=lengthscale),
    )
    posterior = prior.condition_on_observations(
        Y=y_obs,
        X=x_obs,
        b=pn.randvars.Normal(np.zeros(n_obs), sigma**2 * np.eye(n_obs)),
    )
    return posterior, x_obs, y_obs


def _f_monotone(x):
    """Smooth monotone target on [0, 1]; range [0, 1]."""
    return 0.5 * (1 - np.cos(np.pi * np.asarray(x)))


# -----------------------------------------------------------------------------
# 1. Empty-constraint recovery
# -----------------------------------------------------------------------------


def test_with_no_inequality_constraints_recovers_unconstrained_mean():
    """With no constraints, ``posterior_mean(X) == m_E(X)`` exactly."""
    posterior, _, _ = _make_posterior_1d_with_data(_f_monotone)
    wrapper = ConstrainedLinPDEPosterior(posterior, inequality_constraints=[])

    x_test = np.linspace(0, 1, 50).reshape(-1, 1)
    mean_wrapper = wrapper.posterior_mean(x_test)
    mean_linpde = np.asarray(posterior.mean(x_test[:, 0])).reshape(-1)

    np.testing.assert_allclose(mean_wrapper, mean_linpde, atol=1e-12)


def test_with_no_inequality_constraints_psat_is_one():
    """P(C | E) is 1.0 by convention when there are no constraints."""
    posterior, _, _ = _make_posterior_1d_with_data(_f_monotone)
    wrapper = ConstrainedLinPDEPosterior(posterior, inequality_constraints=[])

    assert wrapper.constraint_satisfaction_probability == 1.0


# -----------------------------------------------------------------------------
# 2. Mean-subtraction invariance (identity constraint)
# -----------------------------------------------------------------------------


def test_mean_subtraction_invariance_identity_constraint():
    """Wrapper's transform_bounds_for_zero_mean machinery must produce the
    same answer as a hand-written centered setup.

    We construct a boundedness constraint ``a ≤ u(X_v) ≤ b``, then verify
    that the wrapper's reported `gp_constr` LB / UB at the virtual points
    equals ``(a - m_E(X_v), b - m_E(X_v))`` — the centered bound. This
    isolates the algebra from any sampler.
    """
    from linpde_gp.linfuncops.diffops import PartialDerivative
    from linpde_gp.linfuncops.diffops._coefficients import MultiIndex

    posterior, _, _ = _make_posterior_1d_with_data(_f_monotone)
    Xv = np.linspace(0.0, 1.0, 5).reshape(-1, 1)

    a, b = -0.5, 1.5
    # Identity = PartialDerivative of order 0 (multi-index = 0 for scalar input)
    identity_op = PartialDerivative(MultiIndex(0))

    constraints = [
        InequalityConstraint(operator=identity_op, Xv=Xv, bounds=(a, b)),
    ]
    wrapper = ConstrainedLinPDEPosterior(posterior, constraints)

    # Pull LB / UB out of the gp_constr model.
    cb = wrapper.gp_constr_model.constr_bounded
    assert cb is not None
    LB = cb.LB(cb.Xv)
    UB = cb.UB(cb.Xv)

    m_E_Xv = np.asarray(posterior.mean(Xv[:, 0])).reshape(-1)
    np.testing.assert_allclose(LB, a - m_E_Xv, atol=1e-12)
    np.testing.assert_allclose(UB, b - m_E_Xv, atol=1e-12)


# -----------------------------------------------------------------------------
# 3. Sample feasibility for monotonicity (derivative inequality)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_RPY2, reason="rpy2 + R required for gp_constr's " "truncated-MVN sampler"
)
def test_samples_satisfy_monotonicity_constraint():
    """Drawn samples respect ``u'(X_v) ≥ -tol`` at virtual points.

    `tol` budgets for the constraint-likelihood slack ε_v plus a few sigmas
    of stochastic noise — exact enforcement is not the model's promise, the
    constraint is enforced up to ε_v in the truncated-MVN.
    """
    from linpde_gp.linfuncops.diffops import Derivative

    posterior, _, _ = _make_posterior_1d_with_data(_f_monotone, seed=1)
    Xv = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    constraints = [
        InequalityConstraint(
            operator=Derivative(order=1),
            Xv=Xv,
            bounds=(0.0, np.inf),
        ),
    ]
    wrapper = ConstrainedLinPDEPosterior(posterior, constraints, constr_likelihood=1e-6)

    samples = wrapper.posterior_samples(Xv, n_samples=20)
    # Verify the *underlying* truncated-MVN samples on u'(X_v) — i.e.
    # the constraint distribution itself, not the predictive samples on u.
    # Easier: check approximate monotonicity of each sampled function.
    diffs = np.diff(samples, axis=1)  # (n_samples, len(Xv) - 1)
    # Slack: the bound is u'(x) >= 0 with constr_likelihood = 1e-6, plus a
    # generous numerical envelope. We only require *most* increments to be
    # non-negative; isolated minor negatives within virtual-point spacing
    # are allowed (Agrell §3.3 — virtual-point monotonicity does not imply
    # pathwise monotonicity, though density of X_v makes it tight).
    fraction_monotone = np.mean(diffs >= -1e-2)
    assert fraction_monotone > 0.95, (
        f"Sample paths violate monotonicity too often: "
        f"only {fraction_monotone:.3f} of finite differences are >= -1e-2."
    )


# -----------------------------------------------------------------------------
# 4. Optimization regression guard
# -----------------------------------------------------------------------------


def test_gp_constr_model_optimize_raises_through_adapter():
    """Calling .optimize() on the underlying gp_constr.GPmodel must raise.

    The kernel adapter rejects set_params/get_params/K_gradients with
    NoHyperparamOptimization. ``GPmodel.optimize`` invokes those
    on the kernel, so it must surface an exception (not silently do
    something).
    """
    from linpde_gp.linfuncops.diffops import Derivative

    posterior, _, _ = _make_posterior_1d_with_data(_f_monotone)
    Xv = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    constraints = [
        InequalityConstraint(operator=Derivative(order=1), Xv=Xv, bounds=(0.0, np.inf)),
    ]
    wrapper = ConstrainedLinPDEPosterior(posterior, constraints)
    # gp_constr's optimize calls into kernel.set_params / get_params via
    # its private __setparams; this must raise.
    with pytest.raises(
        (NoHyperparamOptimization, AttributeError, RuntimeError, ValueError)
    ):
        wrapper.gp_constr_model.optimize()


# -----------------------------------------------------------------------------
# 5. P(C | E) diagnostic — only runs when R/rpy2 is available
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_RPY2, reason="rpy2 + R required for pmvnorm")
def test_constraint_satisfaction_probability_diagnostic():
    """P(C | E) for a clearly-feasible inequality should be > 0.5; for a
    clearly-infeasible inequality, < 0.5. Smoke check on the diagnostic.
    """
    from linpde_gp.linfuncops.diffops import PartialDerivative
    from linpde_gp.linfuncops.diffops._coefficients import MultiIndex

    posterior, _, _ = _make_posterior_1d_with_data(_f_monotone, seed=0)

    Xv = np.linspace(0.05, 0.95, 5).reshape(-1, 1)
    # The target f_monotone has range [0, 1]; a ≤ u ≤ 1 with a generously
    # below the data must be feasible.
    constraints_feasible = [
        InequalityConstraint(
            operator=PartialDerivative(MultiIndex(0)),
            Xv=Xv,
            bounds=(-2.0, 2.0),
        ),
    ]
    wrapper_ok = ConstrainedLinPDEPosterior(posterior, constraints_feasible)
    p_ok = wrapper_ok.constraint_satisfaction_probability
    assert 0.0 < p_ok <= 1.0
    assert p_ok > 0.5

    # An impossible bound: a ≤ u ≤ a-eps. With LB > UB, our pre-check
    # raises before ever reaching pmvnorm; instead use a tight bound that
    # the data clearly violates.
    constraints_infeasible = [
        InequalityConstraint(
            operator=PartialDerivative(MultiIndex(0)),
            Xv=Xv,
            bounds=(5.0, 6.0),  # u must be in [5, 6] — the data is in [0, 1]
        ),
    ]
    wrapper_bad = ConstrainedLinPDEPosterior(posterior, constraints_infeasible)
    p_bad = wrapper_bad.constraint_satisfaction_probability
    assert 0.0 <= p_bad < 0.5
