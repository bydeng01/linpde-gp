"""
Tests for the noisy/sparse data-assimilation experiment
(experiments/linpde_gp_constrained/poisson_2d_data_assimilation).

The experiment replaces the exact partial Dirichlet data of the 2D showcase
with NOISY scattered observations of u (sigma_obs > 0). The point of these
tests is to lock the two claims that make the experiment meaningful:

  * (R-free) A noisy point observation of u needs NO wrapper change — it
    is the same _EvaluationFunctional + observation-noise object the collocation
    posterior already accepts, and conditioning on it degrades the equality
    posterior gracefully (RMSE rises with sigma_obs, falls with n_obs).
  * (R-free) The monotonicity inequality stays informative (MC P(C|E)
    bounded away from 0 and 1) in the information-poor regime.
  * (needs R) The inequality reduces the posterior-mean RMSE, and the
    gain is LARGER for noisier/sparser data than for cleaner/denser data —
    the experiment's headline.

The R-gated test skips cleanly where rpy2 / R is unavailable; the rest need
only linpde-gp + probnum + jax + scipy.
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

_DENOM = float((np.exp(1.0) - 1.0) ** 2)
_LS = 0.5
_NC = 5  # 3×3 interior collocation points
_NV = 4  # 4×4 virtual points/axis -> joint constraint dim 32
_AXES = (0, 1)
_LO, _HI = 0.2, 0.8


def _u_true(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x, y = X[:, 0], X[:, 1]
    return (np.exp(x) - 1.0) * (np.exp(y) - 1.0) / _DENOM


def _f_rhs(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x, y = X[:, 0], X[:, 1]
    return -(np.exp(x) * (np.exp(y) - 1.0) + np.exp(y) * (np.exp(x) - 1.0)) / _DENOM


def _prior():
    return pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(2,), output_shape=()),
        cov=1.0 * LinpdeExpQuad(input_shape=(2,), lengthscales=_LS),
    )


def _interior_grid(n):
    a = np.linspace(0.0, 1.0, n)[1:-1]
    XX, YY = np.meshgrid(a, a, indexing="ij")
    return np.column_stack([XX.ravel(), YY.ravel()])


def _scatter(n_obs, rng, lo=1e-3, hi=1.0 - 1e-3):
    """Deterministic space-filling (Halton) locations, mirroring
    run._obs_points (default): fixed, nested across n_obs, so seeds vary only
    the noise. ``rng`` is accepted for signature parity but unused here."""
    from scipy.stats import qmc

    pts = qmc.Halton(d=2, scramble=False).random(n_obs + 1)[1:]
    return lo + (hi - lo) * pts


def _axis_op(axis):
    mi = np.zeros(2, dtype=int)
    mi[axis] = 1
    return PartialDerivative(MultiIndex(mi))


def _build(n_obs, sigma_obs, obs_seed, obs_hi=1.0 - 1e-3):
    """Equality (PDE + noisy data) posterior, constrained wrapper, virtual grid.

    Mirrors run.solve_assimilation's construction; kept local so the test does
    not import the experiment module. ``obs_hi < 1`` starves the (1,1) corner."""
    rng = np.random.default_rng(1000 + obs_seed)
    prior = _prior()
    X_pde = _interior_grid(_NC)
    X_obs = _scatter(n_obs, rng, hi=obs_hi)
    Y_obs = _u_true(X_obs) + sigma_obs * rng.standard_normal(n_obs)
    posterior = make_pde_collocation_posterior(
        prior,
        X_pde,
        _f_rhs,
        sigma_pde=0.0,
        X_bc=X_obs,
        Y_bc=Y_obs,
        sigma_bc=sigma_obs,
        pde_jitter=1e-8,
        bc_jitter=1e-12,
    )
    a_v = np.linspace(_LO, _HI, _NV)
    VX, VY = np.meshgrid(a_v, a_v, indexing="ij")
    Xv = np.column_stack([VX.ravel(), VY.ravel()])
    cwrap = ConstrainedLinPDEPosterior(
        posterior,
        [
            InequalityConstraint(operator=_axis_op(ax), Xv=Xv, bounds=(0.0, np.inf))
            for ax in _AXES
        ],
        constr_likelihood=1e-4,
    )
    return posterior, cwrap, Xv


def _eval_grid(n=30):
    a = np.linspace(0.0, 1.0, n)
    XX, YY = np.meshgrid(a, a, indexing="ij")
    return np.column_stack([XX.ravel(), YY.ravel()]), n


def _rmse(mean, truth):
    return float(np.sqrt(np.mean((mean - truth) ** 2)))


def _mc_p_ce(posterior, Xv, rng, axes=_AXES, n_mc=20000, jitter=1e-9):
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


# -----------------------------------------------------------------------------
# R-free — noisy data needs no wrapper change; PSD gram; graceful decay.
# -----------------------------------------------------------------------------
def test_noisy_observation_block_constructs_and_is_psd():
    """A noisy scattered-u observation block builds a valid equality posterior
    (the gram Cholesky-factorises) for sigma_obs > 0 — i.e. the noise is
    absorbed into k_E exactly through the existing X_bc/Y_bc/sigma_bc slot, no
    upstream change."""
    posterior, _, _ = _build(n_obs=16, sigma_obs=0.1, obs_seed=0)
    G = np.asarray(posterior.gram.todense())
    G = 0.5 * (G + G.T)
    # Symmetric positive-definite: Cholesky succeeds.
    np.linalg.cholesky(G)
    # The posterior mean evaluates on the numpy path.
    X_eval, _ = _eval_grid(8)
    m = np.asarray(posterior.mean(X_eval)).reshape(-1)
    assert m.shape == (64,)
    assert np.all(np.isfinite(m))


def test_equality_rmse_grows_with_noise_and_shrinks_with_count():
    """The equality posterior degrades gracefully: averaged over a few seeds,
    RMSE(u|E) increases with sigma_obs and decreases with n_obs."""
    X_eval, _ = _eval_grid(30)
    truth = _u_true(X_eval)

    def avg_rmse(n_obs, sigma_obs):
        vals = []
        for s in (0, 1, 2):
            post, _, _ = _build(n_obs, sigma_obs, s)
            vals.append(_rmse(np.asarray(post.mean(X_eval)).reshape(-1), truth))
        return float(np.mean(vals))

    # Noise monotonicity (n_obs fixed): more noise -> larger RMSE.
    assert avg_rmse(16, 0.20) > avg_rmse(16, 0.02)
    # Count monotonicity (sigma fixed): fewer obs -> larger RMSE.
    assert avg_rmse(6, 0.10) > avg_rmse(40, 0.10)


def test_inequality_is_informative_in_poor_regime_R_free():
    """In an information-poor regime the joint MC P(C|E) over both axes bites —
    neither decorative (~1) nor almost-surely violated (~0). R-free."""
    posterior, _, Xv = _build(n_obs=8, sigma_obs=0.15, obs_seed=0)
    p = _mc_p_ce(posterior, Xv, np.random.default_rng(0))
    assert 0.02 < p < 0.98, p


# -----------------------------------------------------------------------------
# Needs R — the inequality's RMSE gain is larger for poorer data.
# -----------------------------------------------------------------------------
def _gain(n_obs, sigma_obs, obs_seed, obs_hi=1.0 - 1e-3):
    """Relative RMSE gain 1 - RMSE(con)/RMSE(unc) at one operating point."""
    posterior, cwrap, _ = _build(n_obs, sigma_obs, obs_seed, obs_hi=obs_hi)
    X_eval, _ = _eval_grid(30)
    truth = _u_true(X_eval)
    unc = np.asarray(posterior.mean(X_eval)).reshape(-1)
    con = np.asarray(cwrap.posterior_mean(X_eval)).reshape(-1)
    r_unc = _rmse(unc, truth)
    return 1.0 - _rmse(con, truth) / r_unc, r_unc


def test_inequality_gain_grows_as_data_degrade():
    """Headline (needs R): the relative RMSE gain from the monotonicity
    inequality is larger when the data are noisier/sparser than when they are
    clean/dense.

    We use the data-STARVED corner regime (obs_hi < 1, the partial-BC showcase
    under noise), because the gain on the MEAN appears only where the data leave
    a non-monotone, under-determined region — with full-domain uniform data the
    smooth PDE + data already determine u and the scale-free inequality only
    trims the band. Averaged over a couple of seeds to damp realisation noise.
    """
    try:
        # Information-poor: few, noisy observations + a data-free corner.
        poor = np.mean([_gain(10, 0.10, s, obs_hi=0.85)[0] for s in (0, 1)])
        # Information-rich: many, clean observations everywhere.
        rich = np.mean([_gain(40, 0.01, s, obs_hi=1.0 - 1e-3)[0] for s in (0, 1)])
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"gp_constr R backend unavailable: {exc}")

    # The constraint helps the mean materially when data are poor ...
    assert poor > 0.05, f"expected a sizeable gain in the poor regime; got {poor:.3f}"
    # ... and more than it does when data are rich (gain grows as data degrade).
    assert (
        poor > rich
    ), f"gain did not grow as data degraded: poor={poor:.3f} rich={rich:.3f}"
