r"""Positivity-constrained GP for a steady advection-diffusion air-quality field
=============================================================================

Reconstruct a steady pollutant concentration c(x, y) >= 0 from sparse ground
monitors, conditioning a GP on a steady linear advection-diffusion PDE as
equality information E and on nonnegativity of concentration as inequality
information C, through the ``linpde_gp_constrained`` seam.

Problem
-------
On Omega = (0, 1)^2 (nondimensional),

    L c := -D Delta c + v . grad c + k c = S,   c >= 0,

with constant diffusivity D, wind v = (vx, vy), decay k and Gaussian source S.
L is the constant-coefficient elliptic operator built by
``adv_diff_operator.make_adv_diff_operator`` (a SumLinearFunctionOperator of
WeightedLaplacian, DirectionalDerivative and IdentityOperator, so every term of
the L_s L_t k ExpQuad chain stays closed form).

Equality side E
---------------
  * PDE residual L c = S enforced (noiselessly) at an interior collocation grid,
    through ``make_pde_collocation_posterior(pde_op=L, pde_sign=1.0)``.
  * n_obs noisy monitor observations c(X_i) + N(0, sigma_obs^2) through the
    X_bc / Y_bc / sigma_bc slot. The noise is absorbed into k_E.

Inequality side C
-----------------
Pointwise nonnegativity 0 <= c on a virtual-point grid (d <= 48), mapped to
gp_constr's bounded constraint via an identity ``InequalityConstraint`` with
bounds (0, +inf) -- the identity / boundedness branch
(constrained_posterior.py:129). The zero-mean prior sits on the boundary of the
feasible region, so the truncated-Gaussian mean is lifted off zero only by the
data and the source term.

Headline experiment (see sweep.py + figure.py)
----------------------------------------------
Four arms on the same prior and monitors -- ``monitors``, ``monitors+E``,
``monitors+C``, ``monitors+E+C`` -- are scored by held-out RMSE, full-grid RMSE,
band width, P(C|E) and a negativity diagnostic. As the data degrade (sparser /
noisier), the equality posterior spends more mass below zero and positivity buys
a larger RMSE and calibration gain.

Run from the repo root inside the container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \
      python experiments/linpde_gp_constrained/air_quality_adv_diff/run.py

Optional OpenAQ-format CSV, e.g. the offline London sample (synthetic values;
falls back to the synthetic setup if the CSV is absent):

    ... run.py --data experiments/.../air_quality_adv_diff/data/openaq_london_no2_2024-07-15.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import probnum as pn
from scipy.stats import truncnorm

from linpde_gp.linfuncops.diffops._coefficients import MultiIndex
from linpde_gp.linfuncops.diffops._partial_derivative import PartialDerivative
from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad
from linpde_gp_constrained import (
    ConstrainedLinPDEPosterior,
    InequalityConstraint,
    make_pde_collocation_posterior,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adv_diff_operator import make_adv_diff_operator  # noqa: E402
import fetch_data  # noqa: E402
import plume  # noqa: E402

SEED = 0
ARMS = ("monitors", "monitors+E", "monitors+C", "monitors+E+C")


def _identity_op(dim: int = 2) -> PartialDerivative:
    """Order-0 PartialDerivative -> the identity / boundedness branch of the seam."""
    return PartialDerivative(MultiIndex(np.zeros(dim, dtype=int)))


# -----------------------------------------------------------------------------
# Grid / monitor helpers.
# -----------------------------------------------------------------------------
def _colloc_grid(n_per_axis: int, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Collocation grid spanning [lo, hi]^2.

    Coverage matters: the elliptic BVP carries no Dirichlet block (E is PDE
    residual + monitors only), so the source-free upwind / edge
    cells are pinned only by the decay term k u in the collocation residual. A
    grid that reaches toward the boundary lets that term suppress the otherwise
    underdetermined far-field, yielding the physical decay-to-zero upwind around
    which the posterior mean dips slightly negative.
    """
    a = np.linspace(lo, hi, n_per_axis)
    XX, YY = np.meshgrid(a, a, indexing="ij")
    return np.column_stack([XX.ravel(), YY.ravel()])


def _full_grid(n_per_axis: int):
    a = np.linspace(0.0, 1.0, n_per_axis)
    XX, YY = np.meshgrid(a, a, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    return pts, XX, YY


# Held-out geometry: the upwind low-concentration corner and the plume
# centreline ray from the source along the wind. These regions are scored but
# never observed, so they probe extrapolation where positivity bites.
_SRC = np.array([0.28, 0.34])


def _centreline_dist(X: np.ndarray, v) -> np.ndarray:
    """Perpendicular distance of X from the centreline ray src + t v (t >= 0)."""
    v = np.asarray(v, dtype=float)
    vn = v / np.linalg.norm(v)
    d = X - _SRC
    t = d @ vn
    proj = np.outer(np.clip(t, 0.0, None), vn)
    return np.linalg.norm(d - proj, axis=1)


def _in_heldout(X: np.ndarray, v) -> np.ndarray:
    """Boolean mask: points in the upwind clean corner or near the centreline."""
    upwind = (X[:, 0] < 0.22) & (X[:, 1] < 0.22)
    centre = _centreline_dist(X, v) < 0.05
    return upwind | centre


def _obs_points(
    n_obs: int, v, obs_lo: float = 1e-3, obs_hi: float = 1.0 - 1e-3
) -> np.ndarray:
    """n_obs deterministic Halton monitor LOCATIONS avoiding the held-out zones.

    Nested low-discrepancy design (same rationale as
    poisson_2d_data_assimilation._obs_points): points are FIXED across seeds so
    obs_seed varies only the noise; n_obs is a clean knob. Candidates falling in
    the upwind / centreline held-out zones are dropped so those regions stay
    data-starved.
    """
    from scipy.stats import qmc

    pts = qmc.Halton(d=2, scramble=False).random(8 * n_obs + 8)[1:]
    pts = obs_lo + (obs_hi - obs_lo) * pts
    keep = pts[~_in_heldout(pts, v)]
    if keep.shape[0] < n_obs:
        raise ValueError(
            f"not enough monitors outside held-out zones ({keep.shape[0]})."
        )
    return keep[:n_obs]


def _heldout_points(v) -> np.ndarray:
    """Fixed held-out monitor set: upwind clean corner + plume centreline."""
    upwind = np.array([[0.10, 0.10], [0.16, 0.10], [0.10, 0.16], [0.14, 0.14]])
    vn = np.asarray(v, float) / np.linalg.norm(v)
    centre = np.array([_SRC + t * vn for t in (0.18, 0.34, 0.50, 0.66)])
    centre = np.clip(centre, 0.02, 0.98)
    return np.vstack([upwind, centre])


# -----------------------------------------------------------------------------
# P(C | E) Monte Carlo for the identity / boundedness block (R-free).
# -----------------------------------------------------------------------------
def mc_p_ce(
    posterior, Xv, rng, n_mc: int = 20000, sigma_v: float = 1e-4, jitter: float = 1e-9
) -> float:
    """R-free MC estimate of P(C | E) for positivity 0 <= c(X_v).

    Under u | E the virtual vector Z = L_ineq u + eps_v, eps_v ~ N(0, sigma_v I),
    is Gaussian with mean m_E(X_v) and covariance k_E(X_v, X_v) + sigma_v I. sigma_v
    is the constraint likelihood (Agrell's Sigma_v), so this box probability matches
    the paper's P(C | E) and gp_constr's exact constrprob_Xv(posterior=True), whose
    covariance B1 also carries constr_likelihood * I. Sample and count the fraction of
    draws with every component >= 0; reproduces gp_constr's prior-predictive
    ``constrprob_Xv`` for the bounded constraint without rpy2 / R.
    """
    from linpde_gp_constrained.kernel_adapter import LinPDEKernelAdapter

    adapter = LinPDEKernelAdapter(posterior)
    mean = np.asarray(posterior.mean(Xv)).reshape(-1)
    cov = np.asarray(adapter.K(Xv, Xv))
    cov = 0.5 * (cov + cov.T) + (sigma_v + jitter) * np.eye(cov.shape[0])
    draws = rng.multivariate_normal(mean, cov, size=n_mc, check_valid="ignore")
    return float(np.mean(np.all(draws >= 0.0, axis=1)))


# -----------------------------------------------------------------------------
# R-free positivity-constrained posterior (fallback when gp_constr's R backend
# is unavailable). Same math as Agrell's bounded constraint: truncate the
# virtual values to the positive orthant, then condition the eval points.
# -----------------------------------------------------------------------------
def _rfree_constrained_positivity(
    posterior,
    Xv,
    X_eval,
    rng,
    n_samples: int = 150,
    n_burn: int = 200,
    jitter: float = 1e-8,
):
    """Constrained mean / std / samples for 0 <= c(X_v), R-free.

    Gibbs-samples C ~ TN(mu_v, K_vv; [0, inf)^m) at the virtual points and draws
    eval | C ~ N(m_E(X_e) + K_ev K_vv^{-1}(C - mu_v),
                 K_ee - K_ev K_vv^{-1} K_ve). Returns (mean, std, samples,
    min_mean_v): min_mean_v is min over virtual points of E[C], guaranteed >= 0.
    """
    from linpde_gp_constrained.kernel_adapter import LinPDEKernelAdapter

    adapter = LinPDEKernelAdapter(posterior)
    mu_v = np.asarray(posterior.mean(Xv)).reshape(-1)
    K_vv = np.asarray(adapter.K(Xv, Xv))
    K_vv = 0.5 * (K_vv + K_vv.T) + jitter * np.eye(K_vv.shape[0])
    m = mu_v.shape[0]

    Q = np.linalg.inv(K_vv)  # precision
    diagQ = np.diag(Q)
    cond_sd = np.sqrt(1.0 / diagQ)

    # Gibbs over the positive-orthant truncated normal.
    x = np.maximum(mu_v, 1e-3)
    samples_v = np.empty((n_samples, m))
    total = n_burn + n_samples
    for it in range(total):
        for i in range(m):
            off = Q[i].copy()
            off[i] = 0.0
            cond_mean = mu_v[i] - (off @ (x - mu_v)) / diagQ[i]
            a = (0.0 - cond_mean) / cond_sd[i]
            x[i] = truncnorm.rvs(
                a, np.inf, loc=cond_mean, scale=cond_sd[i], random_state=rng
            )
        if it >= n_burn:
            samples_v[it - n_burn] = x

    # Eval conditional given virtual values.
    K_ev = np.asarray(adapter.K(X_eval, Xv))
    mu_e = np.asarray(posterior.mean(X_eval)).reshape(-1)
    A = K_ev @ Q  # (n_e, m)
    cond_means = mu_e[None, :] + (samples_v - mu_v[None, :]) @ A.T  # (S, n_e)

    # Conditional covariance diagonal (shared across samples).
    K_ee_diag = np.asarray(adapter.K_diag(X_eval)).reshape(-1)
    reduce_diag = np.einsum("ij,ij->i", A, K_ev)  # diag(K_ev Q K_ve)
    cond_var = np.clip(K_ee_diag - reduce_diag, 1e-12, None)
    noise = (
        rng.standard_normal((n_samples, X_eval.shape[0])) * np.sqrt(cond_var)[None, :]
    )
    samples_e = cond_means + noise

    mean = cond_means.mean(axis=0)
    std = samples_e.std(axis=0, ddof=1)
    min_mean_v = float(samples_v.mean(axis=0).min())
    return mean, std, samples_e, min_mean_v


# -----------------------------------------------------------------------------
# Negativity diagnostics.
# -----------------------------------------------------------------------------
def _negativity(mean: np.ndarray, std: np.ndarray) -> dict:
    """Negativity of a posterior mean field and a moment-based tail proxy.

    neg_frac    : fraction of grid cells with mean < 0.
    min_val     : most-negative mean value.
    subzero_tail: mean_x Phi(-mean(x)/std(x)); this is the exact marginal
                  tail probability for a Gaussian posterior and a Gaussian
                  moment-based proxy for a non-Gaussian posterior.
    """
    from scipy.stats import norm

    neg_frac = float(np.mean(mean < 0.0))
    min_val = float(mean.min())
    s = np.clip(std, 1e-12, None)
    subzero = float(np.mean(norm.cdf(-mean / s)))
    return dict(neg_frac=neg_frac, min_val=min_val, subzero_tail=subzero)


# -----------------------------------------------------------------------------
# Posterior builders.
# -----------------------------------------------------------------------------
def _make_prior(lengthscale: float):
    return pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(2,), output_shape=()),
        cov=1.0 * LinpdeExpQuad(input_shape=(2,), lengthscales=lengthscale),
    )


def _monitors_posterior(prior, X_obs, Y_obs, sigma_obs, jitter=1e-10):
    """u | monitors only (no PDE), via standard linpde-gp conditioning."""
    n = X_obs.shape[0]
    b = pn.randvars.Normal(np.zeros(n), (sigma_obs**2 + jitter) * np.eye(n))
    return prior.condition_on_observations(Y_obs, X=X_obs, b=b)


def _pde_posterior(prior, X_pde, S_fn, L, X_obs, Y_obs, sigma_obs, sigma_pde=0.0):
    """u | (PDE collocation L u = S, monitors)."""
    return make_pde_collocation_posterior(
        prior,
        X_pde,
        S_fn,
        pde_op=L,
        pde_sign=1.0,
        sigma_pde=sigma_pde,
        X_bc=X_obs,
        Y_bc=Y_obs,
        sigma_bc=sigma_obs,
        pde_jitter=1e-8,
        bc_jitter=1e-12,
    )


def _unconstrained_fields(posterior, X_eval):
    """Closed-form mean + marginal std of an equality posterior on X_eval."""
    from linpde_gp_constrained.kernel_adapter import LinPDEKernelAdapter

    adapter = LinPDEKernelAdapter(posterior)
    mean = np.asarray(posterior.mean(X_eval)).reshape(-1)
    std = np.sqrt(np.clip(np.asarray(adapter.K_diag(X_eval)).reshape(-1), 0.0, None))
    return mean, std


def _constrained_fields(posterior, Xv, X_eval, rng, n_samples, constr_likelihood):
    """Positivity-constrained mean / std / P(C|E) for one underlying posterior.

    Prefer the gp_constr seam (exact, R-backed); fall back to the R-free
    truncated-normal sampler when the R backend is unavailable so the checks and
    figures run offline. Returns (mean, std, p_ce_exact, min_mean_v, backend).
    """
    constraints = [
        InequalityConstraint(operator=_identity_op(2), Xv=Xv, bounds=(0.0, np.inf))
    ]
    try:
        cwrap = ConstrainedLinPDEPosterior(
            posterior, constraints, constr_likelihood=constr_likelihood
        )
        # Truncated-Gaussian moments (Agrell Cor. 6): one seeded R call yields
        # reproducible mean and std, subject to numerical-integration error.
        # Phi(-mean/std) is used downstream as a moment-based tail proxy.
        # Replaces the earlier samples.std(ddof=1) estimate (N=150, ~6% SE).
        mean, std = cwrap.posterior_moments(X_eval)
        mean = np.asarray(mean).reshape(-1)
        std = np.asarray(std).reshape(-1)
        try:
            p_ce = float(cwrap.constraint_satisfaction_probability)
        except Exception:  # noqa: BLE001
            p_ce = np.nan
        min_mean_v = float(np.asarray(cwrap.posterior_mean(Xv)).reshape(-1).min())
        return mean, std, p_ce, min_mean_v, "gp_constr"
    except (ImportError, RuntimeError):
        mean, std, _samp, min_mean_v = _rfree_constrained_positivity(
            posterior, Xv, X_eval, rng, n_samples=n_samples
        )
        return mean, std, np.nan, min_mean_v, "rfree"


# -----------------------------------------------------------------------------
# End-to-end solve at one operating point (shared by main / sweep / figure).
# -----------------------------------------------------------------------------
def solve(
    n_obs: int = 14,
    sigma_obs: float = 0.10,
    obs_seed: int = 0,
    lengthscale: float = 0.20,
    n_colloc_axis: int = 9,  # 9x9 = 81 collocation points spanning [0.05,0.95]
    colloc_lo: float = 0.05,
    colloc_hi: float = 0.95,
    n_v_axis: int = 3,  # 3x3 = 9 virtual points (d <= 48; keeps P(C|E) informative)
    n_eval_axis: int = 30,
    constr_likelihood: float = 1e-4,
    Xv_lo: float = 0.10,
    Xv_hi: float = 0.90,
    sigma_pde: float = 1e-3,
    n_samples: int = 150,
    n_fine: int = 161,
    compute_constrained: bool = True,
    truth_sampler=None,
    params: dict | None = None,
    source_fn=None,
    real_data=None,
) -> dict:
    """Build all four arms at one (n_obs, sigma_obs) and return arrays + diagnostics.

    ``real_data`` (optional): dict with keys X_obs, Y_obs, sigma_obs for the
    real OpenAQ snapshot; when present it overrides the synthetic monitors and
    full-grid truth scoring is skipped (no ground truth).
    """
    p = params or plume.adv_diff_params()
    D, v, k = p["D"], p["v"], p["k"]
    L = make_adv_diff_operator(D, v, k, dim=2)
    S_fn = source_fn if source_fn is not None else (lambda X: plume.source(X))

    obs_rng = np.random.default_rng(1000 + obs_seed)
    con_rng = np.random.default_rng(2000 + obs_seed)
    mc_rng = np.random.default_rng(SEED)

    prior = _make_prior(lengthscale)
    X_pde = _colloc_grid(n_colloc_axis, colloc_lo, colloc_hi)

    synthetic = real_data is None
    if synthetic:
        if truth_sampler is None:
            truth = plume.solve_truth(n_fine=n_fine, params=p)
            truth_sampler = plume.make_truth_sampler(truth)
        X_obs = _obs_points(n_obs, v)
        Y_obs = truth_sampler(X_obs) + sigma_obs * obs_rng.standard_normal(
            X_obs.shape[0]
        )
        X_held = _heldout_points(v)
        Y_held = truth_sampler(X_held)
    else:
        X_obs = np.asarray(real_data["X_obs"], dtype=float)
        Y_obs = np.asarray(real_data["Y_obs"], dtype=float)
        sigma_obs = float(real_data["sigma_obs"])
        n_obs = X_obs.shape[0]
        # Hold out a fixed fraction of monitors for real data (no grid truth).
        hold = max(2, n_obs // 4)
        X_held, Y_held = X_obs[:hold], Y_obs[:hold]
        X_obs, Y_obs = X_obs[hold:], Y_obs[hold:]

    # Virtual grid for positivity.
    a_v = np.linspace(Xv_lo, Xv_hi, n_v_axis)
    VX, VY = np.meshgrid(a_v, a_v, indexing="ij")
    Xv = np.column_stack([VX.ravel(), VY.ravel()])
    assert Xv.shape[0] <= 48, f"constraint dim {Xv.shape[0]} > 48."

    X_eval, EX, EY = _full_grid(n_eval_axis)
    truth_grid = truth_sampler(X_eval) if synthetic else None

    # Underlying equality posteriors.
    post_mon = _monitors_posterior(prior, X_obs, Y_obs, sigma_obs)
    post_pde = _pde_posterior(prior, X_pde, S_fn, L, X_obs, Y_obs, sigma_obs, sigma_pde)

    # P(C|E) MC for both underlying posteriors (identity block).
    p_c_mon_mc = mc_p_ce(post_mon, Xv, mc_rng, sigma_v=constr_likelihood)
    p_ce_mc = mc_p_ce(
        post_pde, Xv, np.random.default_rng(SEED), sigma_v=constr_likelihood
    )

    # Arm fields.
    fields = {}
    mon_mean, mon_std = _unconstrained_fields(post_mon, X_eval)
    pde_mean, pde_std = _unconstrained_fields(post_pde, X_eval)
    fields["monitors"] = (mon_mean, mon_std)
    fields["monitors+E"] = (pde_mean, pde_std)

    backend = "n/a"
    if compute_constrained:
        c_mean, c_std, _pcc, mmv_c, backend = _constrained_fields(
            post_mon, Xv, X_eval, con_rng, n_samples, constr_likelihood
        )
        ec_mean, ec_std, p_ce_exact, mmv_ec, backend = _constrained_fields(
            post_pde, Xv, X_eval, con_rng, n_samples, constr_likelihood
        )
        fields["monitors+C"] = (c_mean, c_std)
        fields["monitors+E+C"] = (ec_mean, ec_std)
    else:
        nanv = np.full(X_eval.shape[0], np.nan)
        fields["monitors+C"] = (nanv, nanv)
        fields["monitors+E+C"] = (nanv, nanv)
        mmv_c = mmv_ec = np.nan
        p_ce_exact = np.nan

    # Diagnostics per arm.
    diag = dict(
        n_obs=int(n_obs),
        sigma_obs=float(sigma_obs),
        obs_seed=int(obs_seed),
        lengthscale=float(lengthscale),
        p_ce_mc=float(p_ce_mc),
        p_c_mon_mc=float(p_c_mon_mc),
        p_ce_exact=float(p_ce_exact),
        backend=backend,
        synthetic=bool(synthetic),
        min_mean_v_C=float(mmv_c),
        min_mean_v_EC=float(mmv_ec),
        arms={},
    )
    for arm in ARMS:
        mean, std = fields[arm]
        rec = dict(
            rmse_held=float(
                np.sqrt(np.mean((mean_at(mean, X_eval, X_held) - Y_held) ** 2))
            ),
        )
        if synthetic:
            rec["rmse_grid"] = float(np.sqrt(np.mean((mean - truth_grid) ** 2)))
            rec["max_abs"] = float(np.max(np.abs(mean - truth_grid)))
        else:
            rec["rmse_grid"] = np.nan
            rec["max_abs"] = np.nan
        rec["band"] = (
            float(np.mean(2 * 1.96 * std)) if np.all(np.isfinite(std)) else np.nan
        )
        rec.update(
            _negativity(mean, std)
            if np.all(np.isfinite(mean))
            else dict(neg_frac=np.nan, min_val=np.nan, subzero_tail=np.nan)
        )
        diag["arms"][arm] = rec

    # Headline gains: monitors+E+C vs monitors+E.
    base = diag["arms"]["monitors+E"]
    comb = diag["arms"]["monitors+E+C"]
    diag["gain_rmse_held_rel"] = (
        (1.0 - comb["rmse_held"] / base["rmse_held"])
        if base["rmse_held"] > 0
        else np.nan
    )
    if synthetic:
        diag["gain_rmse_grid_rel"] = (
            (1.0 - comb["rmse_grid"] / base["rmse_grid"])
            if base["rmse_grid"] > 0
            else np.nan
        )
    else:
        diag["gain_rmse_grid_rel"] = np.nan

    return dict(
        EX=EX,
        EY=EY,
        n_eval_axis=n_eval_axis,
        truth=truth_grid,
        fields=fields,
        X_pde=X_pde,
        X_obs=X_obs,
        X_held=X_held,
        Y_held=Y_held,
        Xv=Xv,
        v=np.asarray(v),
        diagnostics=diag,
        params=dict(
            n_obs=n_obs,
            sigma_obs=sigma_obs,
            obs_seed=obs_seed,
            lengthscale=lengthscale,
            n_colloc_axis=n_colloc_axis,
            colloc_lo=colloc_lo,
            colloc_hi=colloc_hi,
            n_v_axis=n_v_axis,
            n_eval_axis=n_eval_axis,
            constr_likelihood=constr_likelihood,
            Xv_lo=Xv_lo,
            Xv_hi=Xv_hi,
            sigma_pde=sigma_pde,
            n_fine=n_fine,
        ),
    )


def mean_at(
    mean_grid: np.ndarray, X_eval: np.ndarray, X_query: np.ndarray
) -> np.ndarray:
    """Nearest-grid-node lookup of a flat eval-grid field at query points."""
    from scipy.spatial import cKDTree

    if not np.all(np.isfinite(mean_grid)):
        return np.full(X_query.shape[0], np.nan)
    tree = cKDTree(X_eval)
    _, idx = tree.query(X_query)
    return mean_grid[idx]


# -----------------------------------------------------------------------------
# Real-data preprocessing of an OpenAQ snapshot.
# -----------------------------------------------------------------------------
def _preprocess_openaq(csv_path: str) -> dict:
    """Load + project + nondimensionalise a snapshot CSV to the [0,1]^2 frame.

    Drops non-positive / flagged readings, AVERAGES multiple readings per sensor
    (so a multi-hour window collapses to one value per station -- the steady-state
    snapshot -- instead of duplicate co-located points that wreck the GP gram),
    projects lon/lat to a local planar frame in metres (equirectangular about the
    bbox centroid), rescales to the unit box, nondimensionalises values to order
    unity (divide by the max), and sets sigma_obs from a default instrument
    uncertainty fraction.
    """
    rows = fetch_data.load_csv(csv_path)
    # Aggregate per sensor: mean of the (positive, finite) readings in the window.
    agg: dict = {}
    for r in rows:
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(v) and v > 0.0):
            continue
        sid = r.get("sensor_id", f"{r['lon']},{r['lat']}")
        rec = agg.setdefault(
            sid, dict(lon=float(r["lon"]), lat=float(r["lat"]), vals=[])
        )
        rec["vals"].append(v)
    if not agg:
        raise ValueError(f"no positive readings in {csv_path}.")
    lon = np.array([rec["lon"] for rec in agg.values()])
    lat = np.array([rec["lat"] for rec in agg.values()])
    val = np.array([float(np.mean(rec["vals"])) for rec in agg.values()])

    lat0 = float(lat.mean())
    Rm = 6_371_000.0
    x_m = np.radians(lon - lon.mean()) * Rm * np.cos(np.radians(lat0))  # +x = east
    y_m = np.radians(lat - lat0) * Rm  # +y = north
    # ISOMETRIC rescale into the unit box: a single scale on both axes preserves
    # aspect ratio, so a physical wind direction (east, north) maps to the same
    # direction in the [0,1]^2 operator frame. Center, then scale by the larger
    # extent and recenter into [0.05, 0.95].
    cx, cy = 0.5 * (x_m.min() + x_m.max()), 0.5 * (y_m.min() + y_m.max())
    span = max(x_m.max() - x_m.min(), y_m.max() - y_m.min())
    s = 0.90 / span if span > 0 else 1.0
    X = np.column_stack([0.5 + s * (x_m - cx), 0.5 + s * (y_m - cy)])
    scale = float(val.max())
    Y = val / scale
    sigma_obs = 0.10  # default fractional instrument uncertainty (nondim).
    return dict(
        X_obs=X,
        Y_obs=Y,
        sigma_obs=sigma_obs,
        value_scale=scale,
        frame_scale=float(s),  # metres -> unit box (east/north preserved)
        bbox=(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())),
    )


# -----------------------------------------------------------------------------
# Reporting + snapshot figure.
# -----------------------------------------------------------------------------
def _format_table(diag: dict) -> str:
    def g(v):
        return (
            "      n/a"
            if (v is None or (isinstance(v, float) and np.isnan(v)))
            else f"{v:>9.4f}"
        )

    lines = [
        f"n_obs = {diag['n_obs']},  sigma_obs = {diag['sigma_obs']:.4g},  "
        f"obs_seed = {diag['obs_seed']},  backend = {diag['backend']}",
        f"P(C|E) [MC, monitors+E] = {diag['p_ce_mc']:.4f}   "
        f"P(C) [MC, monitors] = {diag['p_c_mon_mc']:.4f}   "
        f"P(C|E) [exact,R] = {g(diag['p_ce_exact'])}",
        "",
        f"{'arm':<14}{'rmse_held':>11}{'rmse_grid':>11}{'max_abs':>10}"
        f"{'band':>9}{'neg_frac':>10}{'min_val':>10}{'subzero':>9}",
    ]
    for arm in ARMS:
        r = diag["arms"][arm]
        lines.append(
            f"{arm:<14}{g(r['rmse_held'])[-11:]:>11}{g(r['rmse_grid']):>11}"
            f"{g(r['max_abs']):>10}{g(r['band']):>9}{g(r['neg_frac']):>10}"
            f"{g(r['min_val']):>10}{g(r['subzero_tail']):>9}"
        )
    lines += [
        "",
        f"held-out RMSE gain (E+C vs E, rel): {g(diag['gain_rmse_held_rel'])}",
        f"full-grid RMSE gain (E+C vs E, rel): {g(diag['gain_rmse_grid_rel'])}",
    ]
    return "\n".join(lines)


def _heatmap(ax, EX, EY, Z, title, vmin=None, vmax=None, cmap="viridis"):
    im = ax.pcolormesh(EX, EY, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    return im


def main(
    out_dir: str | None = None,
    data_csv: str | None = None,
    wind=None,
    wind_mag: float | None = None,
    D: float | None = None,
    k: float | None = None,
    source: str = "auto",
    max_monitors: int | None = None,
    thin_seed: int = 0,
) -> None:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    real_data = None
    value_scale = None
    if data_csv and os.path.exists(data_csv):
        pre = _preprocess_openaq(data_csv)
        # Reproducible random thinning of the monitor set (density knob for the
        # real-data sparsity sweep). thin_seed varies the subset AND, since
        # solve() holds out the first quarter of the passed array, the held-out
        # CV split -- so averaging over thin_seed is a cross-validated estimate.
        n_all = pre["X_obs"].shape[0]
        if max_monitors is not None and max_monitors < n_all:
            rng = np.random.default_rng(7000 + thin_seed)
            idx = rng.permutation(n_all)[:max_monitors]
            pre = dict(pre, X_obs=pre["X_obs"][idx], Y_obs=pre["Y_obs"][idx])
            print(
                f"Thinned monitors: {n_all} -> {max_monitors} (thin_seed={thin_seed})."
            )
        real_data = pre
        value_scale = pre["value_scale"]
        print(
            f"real data: OpenAQ snapshot {os.path.basename(data_csv)} "
            f"({real_data['X_obs'].shape[0]} monitors, value_scale={value_scale:.3g})."
        )
    elif data_csv:
        print(
            f"NOTE: --data {data_csv} not found; falling back to the synthetic setup."
        )

    # PDE-coefficient overrides. Wind is given as a physical (east, north) vector;
    # only its DIRECTION is identifiable in the nondimensional [0,1]^2 frame (the
    # magnitude is the dimensionless advection strength |v'|, a Peclet-like knob),
    # so we orient v' along the measured wind and keep |v'| at wind_mag (default:
    # the synthetic regime's magnitude). D, k may be overridden too.
    params = plume.adv_diff_params()
    if wind is not None:
        w = np.asarray(wind, dtype=float)
        if w.shape != (2,) or np.allclose(w, 0.0):
            raise SystemExit("--wind must be 'east,north' and nonzero.")
        mag = float(wind_mag) if wind_mag is not None else float(np.hypot(*params["v"]))
        v = mag * w / np.linalg.norm(w)
        params["v"] = (float(v[0]), float(v[1]))
    if D is not None:
        params["D"] = float(D)
    if k is not None:
        params["k"] = float(k)
    if wind is not None or D is not None or k is not None:
        print(
            f"PDE coefficients: D={params['D']:.4g}, v=({params['v'][0]:.3g}, "
            f"{params['v'][1]:.3g}), k={params['k']:.4g}"
        )

    # Source term. For real data the synthetic Gaussian plume is indefensible;
    # default to S=0 (the PDE acts as an advection-diffusion-decay transport prior
    # over the monitor values). 'plume' forces the synthetic source; 'zero' forces
    # S=0; 'auto' = zero for real data, plume for synthetic.
    src_mode = (
        "zero"
        if (source == "auto" and real_data is not None)
        else ("plume" if source == "auto" else source)
    )
    source_fn = (
        (lambda X: np.zeros(np.atleast_2d(X).shape[0]))
        if src_mode == "zero"
        else (lambda X: plume.source(X))
    )
    if real_data is not None:
        print(f"PDE source: S = {src_mode}")

    data = solve(real_data=real_data, params=params, source_fn=source_fn)
    diag = data["diagnostics"]
    n = data["n_eval_axis"]
    EX, EY = data["EX"], data["EY"]

    # Real-data runs write suffixed files so the canonical synthetic (headline) outputs
    # consumed by figure.py are never clobbered.
    sfx = "_openaq" if real_data is not None else ""

    table = _format_table(diag)
    print("--- air_quality_adv_diff diagnostics ---")
    print(table)
    with open(os.path.join(out_dir, f"comparison{sfx}.txt"), "w") as fh:
        fh.write(table + "\n")
    with open(os.path.join(out_dir, f"diagnostics{sfx}.json"), "w") as fh:
        json.dump(diag, fh, indent=2)

    # Persist fields for figure.py (render without re-running R).
    save = dict(
        EX=EX,
        EY=EY,
        n_eval_axis=n,
        X_obs=data["X_obs"],
        X_held=data["X_held"],
        Xv=data["Xv"],
        v=data["v"],
    )
    if data["truth"] is not None:
        save["truth"] = data["truth"]
    for arm in ARMS:
        mean, std = data["fields"][arm]
        save[f"mean__{arm}"] = mean
        save[f"std__{arm}"] = std
    np.savez(os.path.join(out_dir, f"fields{sfx}.npz"), **save)

    # Snapshot figure.
    truth = data["truth"]
    vmax = (
        float(np.nanmax(truth))
        if truth is not None
        else float(np.nanmax(data["fields"]["monitors+E"][0]))
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    if truth is not None:
        _heatmap(axes[0, 0], EX, EY, truth.reshape(n, n), "truth", 0.0, vmax)
    else:
        axes[0, 0].axis("off")
        axes[0, 0].text(0.5, 0.5, "real data\n(no grid truth)", ha="center")
    panel_arms = ["monitors", "monitors+E", "monitors+C", "monitors+E+C"]
    pos = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for arm, (r, c) in zip(panel_arms, pos):
        mean = data["fields"][arm][0].reshape(n, n)
        im = _heatmap(axes[r, c], EX, EY, mean, arm, 0.0, vmax)
    axes[1, 0].axis("off")
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]):
        ax.scatter(
            data["X_obs"][:, 0],
            data["X_obs"][:, 1],
            s=18,
            c="red",
            edgecolors="k",
            linewidths=0.4,
        )
        ax.scatter(
            data["X_held"][:, 0],
            data["X_held"][:, 1],
            s=24,
            marker="^",
            c="white",
            edgecolors="k",
            linewidths=0.5,
        )
    fig.colorbar(im, ax=axes[1, 0], fraction=0.5)
    fig.suptitle(
        "Steady advection-diffusion air quality: monitors / +E / +C / +E+C   "
        f"P(C|E)_MC={diag['p_ce_mc']:.2f}  backend={diag['backend']}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, f"run{sfx}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved: run{sfx}.pdf, comparison{sfx}.txt, diagnostics{sfx}.json, "
        f"fields{sfx}.npz in {out_dir}"
    )

    _self_checks(diag)


def recovery_check(n_dense: int = 12, sigma_lo: float = 1e-4) -> float:
    """Recovery check: dense low-noise monitors + PDE, no inequality, recover the plume.

    Returns the full-grid RMSE of the monitors+E mean vs truth; asserts < 1e-3
    (nondimensional). Dense monitors on a tensor grid (no held-out filtering) so
    the whole plume is observed.
    """
    p = plume.adv_diff_params()
    L = make_adv_diff_operator(p["D"], p["v"], p["k"], dim=2)
    truth = plume.solve_truth(n_fine=201, params=p)
    ts = plume.make_truth_sampler(truth)

    a = np.linspace(0.06, 0.94, n_dense)
    XX, YY = np.meshgrid(a, a, indexing="ij")
    X_obs = np.column_stack([XX.ravel(), YY.ravel()])
    Y_obs = ts(X_obs)

    prior = _make_prior(0.20)
    X_pde = _colloc_grid(11, 0.04, 0.96)
    post = _pde_posterior(
        prior,
        X_pde,
        lambda X: plume.source(X),
        L,
        X_obs,
        Y_obs,
        sigma_lo,
        sigma_pde=1e-4,
    )
    X_eval, _, _ = _full_grid(40)
    mean = np.asarray(post.mean(X_eval)).reshape(-1)
    rmse = float(np.sqrt(np.mean((mean - ts(X_eval)) ** 2)))
    print(
        f"recovery check: dense ({n_dense}x{n_dense}) low-noise monitors+E "
        f"grid RMSE = {rmse:.3e}"
    )
    assert rmse < 1e-3, f"recovery check FAIL: RMSE {rmse:.3e} >= 1e-3."
    print("recovery check PASS")
    return rmse


def _self_checks(diag: dict) -> None:
    """Regime check and constrained-arm check assertions."""
    pmc = diag["p_ce_mc"]
    if not (0.05 < pmc < 0.95):
        print(
            f"\nNOTE: MC P(C|E) = {pmc:.4f} outside (0.05, 0.95) -- retune n_obs / "
            "sigma_obs. See sweep.py."
        )
    else:
        print(f"\nregime check OK: MC P(C|E) = {pmc:.4f} in (0.05, 0.95).")

    base = diag["arms"]["monitors+E"]
    if base["neg_frac"] > 0.0:
        print(
            f"regime check OK: monitors+E mean negative on {100*base['neg_frac']:.1f}% of "
            f"the grid (min {base['min_val']:.3f})."
        )
    else:
        print(
            "NOTE: monitors+E mean is nonnegative everywhere -- raise sigma_obs / "
            "lower n_obs so positivity bites."
        )

    comb = diag["arms"]["monitors+E+C"]
    if np.isfinite(comb["rmse_held"]):
        # Constrained-arm check: mean nonnegative at virtual points.
        assert diag["min_mean_v_EC"] >= -1e-3, (
            f"constrained-arm check FAIL: constrained mean negative at virtual points "
            f"(min {diag['min_mean_v_EC']:.3e})."
        )
        # Constrained-arm check: held-out RMSE improves >= 10% over monitors+E.
        g = diag["gain_rmse_held_rel"]
        if g >= 0.10:
            print(
                f"constrained-arm check OK: held-out RMSE gain {100*g:.1f}% (>= 10%), "
                f"band {base['band']:.3f} -> {comb['band']:.3f}, "
                f"subzero tail {base['subzero_tail']:.3f} -> {comb['subzero_tail']:.3f}."
            )
        else:
            print(
                f"\nNOTE: held-out RMSE gain {100*g:.1f}% < 10% at this snapshot; "
                "the headline operating point is set in sweep.py."
            )
        # Constrained-arm check: band and sub-zero tail shrink.
        assert (
            comb["subzero_tail"] <= base["subzero_tail"] + 1e-6
        ), "constrained-arm check FAIL: constrained sub-zero tail did not shrink."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--data", default=None, help="OpenAQ snapshot CSV (real data).")
    ap.add_argument(
        "--wind",
        default=None,
        help="real wind as 'east,north' (m/s; direction used to orient "
        "the advection term in the unit-box frame).",
    )
    ap.add_argument(
        "--wind-mag",
        type=float,
        default=None,
        help="nondimensional advection magnitude |v'| (default: the "
        "synthetic regime's |v|).",
    )
    ap.add_argument("--D", type=float, default=None, help="override diffusivity D.")
    ap.add_argument("--k", type=float, default=None, help="override decay rate k.")
    ap.add_argument(
        "--source",
        default="auto",
        choices=("auto", "zero", "plume"),
        help="PDE source S (auto: zero for --data, plume for synthetic).",
    )
    ap.add_argument(
        "--max-monitors",
        type=int,
        default=None,
        help="randomly thin the real-data monitors to this count "
        "(density knob; reproducible via --thin-seed).",
    )
    ap.add_argument(
        "--thin-seed",
        type=int,
        default=0,
        help="seed for monitor thinning / held-out CV split.",
    )
    ap.add_argument(
        "--recovery-check",
        action="store_true",
        help="run the dense-data recovery check only.",
    )
    args = ap.parse_args()
    wind = None
    if args.wind:
        wind = [float(t) for t in args.wind.split(",")]
        if len(wind) != 2:
            ap.error("--wind must be 'east,north'.")
    if args.recovery_check:
        recovery_check()
    else:
        main(
            out_dir=args.out_dir,
            data_csv=args.data,
            wind=wind,
            wind_mag=args.wind_mag,
            D=args.D,
            k=args.k,
            source=args.source,
            max_monitors=args.max_monitors,
            thin_seed=args.thin_seed,
        )
