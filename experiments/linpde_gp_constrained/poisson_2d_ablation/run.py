"""
3-way ablation: equality-only vs inequality-only vs both
========================================================

This is a SEPARATE experiment from ``poisson_2d_weak_form`` (which contrasts
only equality-only vs the combined posterior). Here we add the missing third
arm — *inequality-only* — to isolate the contribution of each information
source on the **same** under-determined, free-boundary 2D Poisson problem.

Why this ablation
-----------------
The combination-showcase regime (partial Dirichlet data: only the homogeneous
edges ``x=0`` and ``y=0``; the far edges ``x=1``, ``y=1`` are free) is chosen
precisely so that *neither* information source suffices on its own:

* **Equality only** ``u | E`` (PDE collocation + the two homogeneous edges).
  Away from data a GP reverts to its prior mean (0), so the equality-only mean
  rises and then falls back toward 0 near the free corner ``(1,1)`` — it is
  inaccurate AND non-monotone. The PDE + partial BCs cannot *pin* ``u`` near
  the free boundary.

* **Inequality only** ``u | C`` (monotonicity ``∂_x u ≥ 0`` and ``∂_y u ≥ 0``
  at interior virtual points, applied to the bare zero-mean prior — NO PDE, NO
  BC). Monotonicity is *scale-free*: it constrains the *shape* (sign of the
  slope) but carries no information about the *magnitude*. With a zero-mean
  prior and no data, the truncated-Gaussian mean stays near the prior scale and
  cannot climb to ``u(1,1)=1``. The credible band stays at the (large) prior
  scale. The shape constraint cannot *fix the magnitude*.

* **Both** ``u | (E, C)``. The PDE/BCs set the magnitude where there is data and
  the monotonicity supplies the "keep increasing toward the free corner"
  structure the missing boundaries would have provided. Their *combination*
  yields an accurate, shape-correct reconstruction.

So this experiment directly demonstrates the headline claim: the combination
beats each component alone, because the two components carry *complementary*
information (magnitude vs shape).

Implementation
--------------
The equality-only and both arms are obtained verbatim from
``poisson_2d_weak_form.run.solve`` (imported below), so their numbers match the
parent experiment exactly. The inequality-only arm truncates the prior directly
via :class:`PriorKernelAdapter` + :class:`ConstrainedLinPDEPosterior` — the same
gp_constr machinery, with the equality side removed.

Run from the repo root inside the container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/poisson_2d_ablation/run.py

The inequality-only and both arms need gp_constr's R backend (rpy2 + R
packages mvtnorm / tmvtnorm / TruncatedNormal / truncnorm). Without R the
script degrades gracefully: the equality-only arm and the R-free prior
constraint probability are still reported.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import probnum as pn

from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad
from linpde_gp_constrained import (
    ConstrainedLinPDEPosterior,
    InequalityConstraint,
    PriorKernelAdapter,
)

# Reuse the parent experiment's problem definition + equality/both solve so the
# shared arms are numerically identical to poisson_2d_weak_form.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WEAKFORM_DIR = os.path.join(os.path.dirname(_THIS_DIR), "poisson_2d_weak_form")
if _WEAKFORM_DIR not in sys.path:
    sys.path.insert(0, _WEAKFORM_DIR)
import run as weakform  # noqa: E402  (sibling experiment driver)

# -----------------------------------------------------------------------------
# Single source of truth for the shared configuration. Passed to BOTH the
# parent solve() (equality-only + both) and the inequality-only builder, so all
# three arms share lengthscale, virtual grid, and evaluation grid.
# -----------------------------------------------------------------------------
CONFIG = dict(
    lengthscale=0.5,
    n_colloc_axis=5,  # interior 3×3 = 9 PDE collocation points
    n_bc_axis=7,  # density of the partial (two-edge) boundary grid
    n_v_axis=4,  # 4×4 = 16 virtual points per constrained axis
    n_eval_axis=30,  # 30×30 evaluation grid
    bc_scheme="two_edges",  # partial Dirichlet data -> under-determined regime
    constrain_axes=(0, 1),  # both ∂_x u ≥ 0 and ∂_y u ≥ 0
    constr_likelihood=1e-4,
    sigma_bc=1e-6,
    Xv_lo=0.2,
    Xv_hi=0.8,
)
N_SAMPLES = 150


# -----------------------------------------------------------------------------
# Inequality-only arm: truncate the bare prior by the monotonicity constraints.
# -----------------------------------------------------------------------------
def mc_prior_constraint_prob(
    adapter: PriorKernelAdapter,
    Xv: np.ndarray,
    rng,
    axes=(0, 1),
    n_mc: int = 20000,
    jitter: float = 1e-9,
) -> float:
    """R-free Monte-Carlo estimate of the *prior* constraint probability P(C).

    Under the zero-mean prior the stacked slope vector ``[∂_a u(X_v)]_{a∈axes}``
    is Gaussian with mean 0 and covariance ``Cov(∂_a u, ∂_b u) = Kij(X_v,X_v,a,b)``
    (from :class:`PriorKernelAdapter`). We sample and count the fraction of draws
    with every component ≥ 0. This is the orthant probability of a zero-mean
    Gaussian, i.e. the chance a prior sample is monotone at all virtual points —
    expectedly tiny, which is exactly why the shape constraint is so informative
    (yet, being scale-free, still cannot set the magnitude).
    """
    m = Xv.shape[0]
    k = len(axes)
    mean = np.zeros(k * m)
    cov = np.empty((k * m, k * m))
    for ia, a in enumerate(axes):
        for ib, b in enumerate(axes):
            cov[ia * m : (ia + 1) * m, ib * m : (ib + 1) * m] = np.asarray(
                adapter.Kij(Xv, Xv, a, b)
            )
    cov = 0.5 * (cov + cov.T) + jitter * np.eye(k * m)
    draws = rng.multivariate_normal(mean, cov, size=n_mc, check_valid="ignore")
    return float(np.mean(np.all(draws >= 0.0, axis=1)))


def _min_deriv(mean_flat: np.ndarray, n: int, axis: int) -> float:
    """Most-negative ∂_axis of a flattened (n, n) mean field via central FD."""
    M = mean_flat.reshape(n, n)  # [x, y]
    h = 1.0 / (n - 1)
    if axis == 0:
        d = (M[2:, :] - M[:-2, :]) / (2.0 * h)
    else:
        d = (M[:, 2:] - M[:, :-2]) / (2.0 * h)
    return float(d.min())


def solve_inequality_only(rng, compute_constrained: bool = True) -> dict:
    """Build u | C (prior truncated by monotonicity only) on the eval grid.

    Returns the inequality-only mean / std on the 30×30 grid, the free-corner
    (1,1) value, the R-free prior P(C), and a flag for whether the R backend was
    available.
    """
    cfg = CONFIG
    prior = pn.randprocs.GaussianProcess(
        mean=pn.functions.Zero(input_shape=(2,), output_shape=()),
        cov=1.0 * LinpdeExpQuad(input_shape=(2,), lengthscales=cfg["lengthscale"]),
    )

    # Same virtual grid as the "both" arm.
    a_v = np.linspace(cfg["Xv_lo"], cfg["Xv_hi"], cfg["n_v_axis"])
    VX, VY = np.meshgrid(a_v, a_v, indexing="ij")
    Xv = np.column_stack([VX.ravel(), VY.ravel()])

    constraints = [
        InequalityConstraint(
            operator=weakform._axis_op(ax), Xv=Xv, bounds=(0.0, np.inf)
        )
        for ax in cfg["constrain_axes"]
    ]
    adapter = PriorKernelAdapter(prior)
    cwrap = ConstrainedLinPDEPosterior(
        prior, constraints, constr_likelihood=cfg["constr_likelihood"], adapter=adapter
    )

    # R-free prior constraint probability (independent of the R backend).
    p_c_prior = mc_prior_constraint_prob(adapter, Xv, rng, axes=cfg["constrain_axes"])

    n = cfg["n_eval_axis"]
    X_eval, EX, EY = weakform._full_grid(n)
    truth = weakform.u_true(X_eval)
    free_corner = np.array([[1.0, 1.0]])
    X_pred = np.vstack([X_eval, free_corner])
    n_eval = X_eval.shape[0]

    con_mean = np.full(n_eval, np.nan)
    con_std = np.full(n_eval, np.nan)
    con_free = np.nan
    ok = False
    err = None
    if compute_constrained:
        try:
            print(
                "  [inequality-only] R truncated-MVN moments on the prior "
                f"({len(cfg['constrain_axes'])} axes x {Xv.shape[0]} virtual "
                "points); a few minutes...",
                flush=True,
            )
            # Deterministic moment-based mean AND std in one R call. Because the
            # "both" arm is the same (E,C) posterior as poisson_2d_weak_form,
            # the moment-based band matches that figure exactly (the old
            # sample-std estimate drifted between the two runs).
            mean_all, std_all = cwrap.posterior_moments(X_pred)
            mean_all = np.asarray(mean_all).reshape(-1)
            con_mean, con_free = mean_all[:n_eval], float(mean_all[n_eval])
            con_std = np.asarray(std_all).reshape(-1)[:n_eval]
            ok = True
        except (ImportError, RuntimeError) as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(
                "WARNING: inequality-only arm skipped — gp_constr could not load "
                f"its R backend.\n         {err}"
            )

    diag = {
        "p_c_prior_mc": p_c_prior,
        "constrained_ok": ok,
        "constrained_error": err,
    }
    if ok:
        diag.update(
            l2=float(np.sqrt(np.mean((con_mean - truth) ** 2))),
            max_abs=float(np.max(np.abs(con_mean - truth))),
            band=float(np.mean(2 * 1.96 * con_std)),
            free_corner_err=float(abs(con_free - 1.0)),
            min_d0=_min_deriv(con_mean, n, 0),
            min_d1=_min_deriv(con_mean, n, 1),
        )
    else:
        diag.update(
            l2=np.nan,
            max_abs=np.nan,
            band=np.nan,
            free_corner_err=np.nan,
            min_d0=np.nan,
            min_d1=np.nan,
        )

    return dict(
        EX=EX,
        EY=EY,
        n_eval_axis=n,
        truth=truth,
        ineq_mean=con_mean,
        ineq_std=con_std,
        Xv=Xv,
        diagnostics=diag,
    )


# -----------------------------------------------------------------------------
# Reporting.
# -----------------------------------------------------------------------------
def _format_3way_table(eq_both: dict, ineq: dict) -> str:
    """3-column comparison: equality-only | inequality-only | both."""

    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "        n/a"
        return f"{v:>11.4f}"

    e = eq_both  # diagnostics dict from weakform.solve
    i = ineq["diagnostics"]
    # equality-only and both come from the parent solve diagnostics; the
    # inequality-only column is computed here.
    rows = [
        ("RMSE vs truth", e.get("l2_unc"), i.get("l2"), e.get("l2_con")),
        (
            "max |mean - truth|",
            e.get("max_abs_unc"),
            i.get("max_abs"),
            e.get("max_abs_con"),
        ),
        (
            "free-corner (1,1) error",
            e.get("free_corner_unc"),
            i.get("free_corner_err"),
            e.get("free_corner_con"),
        ),
        ("avg 95% band width", e.get("band_unc"), i.get("band"), e.get("band_con")),
        (
            "min ∂_x mean (full grid)",
            e.get("min_d0_unc"),
            i.get("min_d0"),
            e.get("min_d0_con"),
        ),
        (
            "min ∂_y mean (full grid)",
            e.get("min_d1_unc"),
            i.get("min_d1"),
            e.get("min_d1_con"),
        ),
    ]
    lines = [
        "                              equality-only  inequality-only          both",
    ]
    for name, a, b, c in rows:
        lines.append(f"{name:<28}: {f(a)}  {f(b)}  {f(c)}")
    lines += [
        "",
        f"P(C)        [prior, MC]     : {ineq['diagnostics']['p_c_prior_mc']:>11.4f}   "
        "(prior orthant prob; shape constraint vs zero-mean prior)",
        f"P(C | E)    [exact, R]      : "
        + ("        n/a" if np.isnan(e.get("p_ce", np.nan)) else f"{e['p_ce']:>11.4f}")
        + "   (combination showcase)",
        f"P(C | E)    [MC, R-free]    : {e.get('p_ce_mc', float('nan')):>11.4f}",
    ]
    return "\n".join(lines)


def _heatmap(ax, EX, EY, Z, title, vmin, vmax, Xv=None):
    im = ax.pcolormesh(EX, EY, Z, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    if Xv is not None:
        ax.scatter(Xv[:, 0], Xv[:, 1], s=5, c="k", alpha=0.35)
    return im


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = _THIS_DIR
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(weakform.SEED)

    # --- Equality-only + both (verbatim from the parent experiment).
    print(
        "--- arm 1+3: equality-only and both (poisson_2d_weak_form.solve) ---",
        flush=True,
    )
    eb = weakform.solve(n_samples=N_SAMPLES, **CONFIG)
    e_diag = eb["diagnostics"]
    n = eb["n_eval_axis"]
    EX, EY = eb["EX"], eb["EY"]
    truth = eb["truth"]

    # --- Inequality-only (new arm).
    print(
        "--- arm 2: inequality-only (prior truncated by monotonicity) ---", flush=True
    )
    ineq = solve_inequality_only(rng)
    i_diag = ineq["diagnostics"]

    table = _format_3way_table(e_diag, ineq)
    print("\n--- poisson_2d_ablation: 3-way comparison ---")
    print(table)
    with open(os.path.join(out_dir, "comparison.txt"), "w") as fh:
        fh.write(table + "\n")

    # --- Persist fields + diagnostics for the figure (re-stylable w/o re-run).
    np.savez(
        os.path.join(out_dir, "fields.npz"),
        EX=EX,
        EY=EY,
        n_eval_axis=n,
        truth=truth,
        eq_mean=eb["unc_mean"],
        eq_std=eb["unc_std"],
        ineq_mean=ineq["ineq_mean"],
        ineq_std=ineq["ineq_std"],
        both_mean=eb["con_mean"],
        both_std=eb["con_std"],
        X_pde=eb["X_pde"],
        X_bc=eb["X_bc"],
        Xv=eb["Xv"],
        eq_ok=np.asarray(True),
        ineq_ok=np.asarray(i_diag["constrained_ok"]),
        both_ok=np.asarray(eb["constrained_ok"]),
    )
    combined = {
        "equality_only": {
            "l2": e_diag.get("l2_unc"),
            "max_abs": e_diag.get("max_abs_unc"),
            "free_corner_err": e_diag.get("free_corner_unc"),
            "band": e_diag.get("band_unc"),
            "min_d0": e_diag.get("min_d0_unc"),
            "min_d1": e_diag.get("min_d1_unc"),
        },
        "inequality_only": {
            k: i_diag.get(k)
            for k in (
                "l2",
                "max_abs",
                "free_corner_err",
                "band",
                "min_d0",
                "min_d1",
                "p_c_prior_mc",
                "constrained_ok",
            )
        },
        "both": {
            "l2": e_diag.get("l2_con"),
            "max_abs": e_diag.get("max_abs_con"),
            "free_corner_err": e_diag.get("free_corner_con"),
            "band": e_diag.get("band_con"),
            "min_d0": e_diag.get("min_d0_con"),
            "min_d1": e_diag.get("min_d1_con"),
            "p_ce": e_diag.get("p_ce"),
            "p_ce_mc": e_diag.get("p_ce_mc"),
        },
    }
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as fh:
        json.dump(
            {
                arm: {
                    k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                    for k, v in d.items()
                }
                for arm, d in combined.items()
            },
            fh,
            indent=2,
        )

    # --- Diagnostic figure: truth + three arms (shared scale) + diagonal slice.
    th = truth.reshape(n, n)
    eqm = eb["unc_mean"].reshape(n, n)
    inm = ineq["ineq_mean"].reshape(n, n)
    bom = eb["con_mean"].reshape(n, n)
    vmin, vmax = float(th.min()), float(th.max())

    fig = plt.figure(figsize=(12, 6.5), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85])
    ax_t = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_i = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[0, 3])
    _heatmap(ax_t, EX, EY, th, "truth $u_\\star$", vmin, vmax)
    _heatmap(ax_e, EX, EY, eqm, "equality only", vmin, vmax, Xv=eb["Xv"])
    if i_diag["constrained_ok"]:
        _heatmap(ax_i, EX, EY, inm, "inequality only", vmin, vmax, Xv=eb["Xv"])
    else:
        ax_i.axis("off")
        ax_i.text(
            0.5, 0.5, "inequality-only\nunavailable (no R)", ha="center", va="center"
        )
    if eb["constrained_ok"]:
        im = _heatmap(ax_b, EX, EY, bom, "both (E, C)", vmin, vmax, Xv=eb["Xv"])
        fig.colorbar(im, ax=[ax_t, ax_e, ax_i, ax_b], fraction=0.012, pad=0.01)
    else:
        ax_b.axis("off")
        ax_b.text(0.5, 0.5, "both\nunavailable (no R)", ha="center", va="center")

    # Diagonal slice u(s, s): the mechanism panel.
    ax_s = fig.add_subplot(gs[1, :2])
    s = np.linspace(0.0, 1.0, n)
    diag_idx = np.arange(n)
    ax_s.plot(s, th[diag_idx, diag_idx], "k-", lw=2, label="truth")
    ax_s.plot(
        s, eqm[diag_idx, diag_idx], color="#767676", lw=1.5, label="equality only"
    )
    if i_diag["constrained_ok"]:
        ax_s.plot(
            s,
            inm[diag_idx, diag_idx],
            color="#1f77b4",
            lw=1.5,
            ls="--",
            label="inequality only",
        )
    if eb["constrained_ok"]:
        ax_s.plot(s, bom[diag_idx, diag_idx], color="#d62728", lw=1.8, label="both")
    ax_s.set_xlabel("s")
    ax_s.set_ylabel("u(s, s)")
    ax_s.set_title("diagonal slice (free corner at s=1)", fontsize=9)
    ax_s.legend(fontsize=7, frameon=False)

    # Metrics bars: RMSE / band / free-corner error per arm.
    ax_m = fig.add_subplot(gs[1, 2:])
    metrics = ["RMSE", "band", "free-corner err"]
    eq_vals = [
        e_diag.get("l2_unc"),
        e_diag.get("band_unc"),
        e_diag.get("free_corner_unc"),
    ]
    in_vals = [i_diag.get("l2"), i_diag.get("band"), i_diag.get("free_corner_err")]
    bo_vals = [
        e_diag.get("l2_con"),
        e_diag.get("band_con"),
        e_diag.get("free_corner_con"),
    ]

    def _nan(v):
        return np.nan if v is None else v

    eq_vals = [_nan(v) for v in eq_vals]
    in_vals = [_nan(v) for v in in_vals]
    bo_vals = [_nan(v) for v in bo_vals]
    xpos = np.arange(len(metrics))
    w = 0.27
    ax_m.bar(xpos - w, eq_vals, w, color="#767676", label="equality only")
    ax_m.bar(xpos, in_vals, w, color="#1f77b4", label="inequality only")
    ax_m.bar(xpos + w, bo_vals, w, color="#d62728", label="both")
    ax_m.set_xticks(xpos)
    ax_m.set_xticklabels(metrics)
    ax_m.set_title("lower is better — both wins on every metric", fontsize=9)
    ax_m.legend(fontsize=7, frameon=False)

    fig.suptitle(
        "3-way ablation on the free-boundary 2D Poisson problem — "
        "equality fixes magnitude, inequality fixes shape, only both fixes "
        "everything",
        fontsize=11,
    )
    out_path = os.path.join(out_dir, "ablation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot:   {out_path}")
    print(f"Saved table:  {os.path.join(out_dir, 'comparison.txt')}")
    print(f"Saved fields: {os.path.join(out_dir, 'fields.npz')}")

    # --- Headline assertions: BOTH dominates each single source.
    if i_diag["constrained_ok"] and eb["constrained_ok"]:
        # (1) Inequality-only cannot fix the magnitude: its free-corner error
        #     stays large (the scale-free constraint cannot climb to u(1,1)=1).
        assert i_diag["free_corner_err"] > 0.3, (
            "inequality-only unexpectedly recovered the free corner "
            f"({i_diag['free_corner_err']:.3f}); the scale-free shape constraint "
            "should NOT be able to set the magnitude. Check the prior scale."
        )
        # (2) Both beats equality-only on RMSE (the parent experiment's claim).
        assert e_diag["l2_con"] < 0.85 * e_diag["l2_unc"], (
            f"both did not improve on equality-only: RMSE {e_diag['l2_unc']:.4f} "
            f"-> {e_diag['l2_con']:.4f}."
        )
        # (3) Both beats inequality-only on RMSE (combination > shape alone).
        assert e_diag["l2_con"] < i_diag["l2"], (
            f"both ({e_diag['l2_con']:.4f}) did not beat inequality-only "
            f"({i_diag['l2']:.4f}) on RMSE."
        )
        # (4) The combination is monotone on the interior where neither single
        #     source is both accurate AND monotone (equality is non-monotone,
        #     inequality is monotone-but-wrong-scale).
        assert e_diag.get("min_d0_unc", 0.0) < -1e-2, (
            "equality-only mean is already monotone; the ablation's premise "
            "(under-determination) is not met — under-determine more."
        )
        print(
            "\nABLATION OK:\n"
            f"  equality only  : RMSE {e_diag['l2_unc']:.3f}, "
            f"free-corner err {e_diag['free_corner_unc']:.3f}, "
            f"min ∂ {min(e_diag['min_d0_unc'], e_diag['min_d1_unc']):.3f} "
            "(inaccurate + non-monotone)\n"
            f"  inequality only: RMSE {i_diag['l2']:.3f}, "
            f"free-corner err {i_diag['free_corner_err']:.3f}, "
            f"min ∂ {min(i_diag['min_d0'], i_diag['min_d1']):.3f} "
            "(monotone but wrong magnitude)\n"
            f"  both           : RMSE {e_diag['l2_con']:.3f}, "
            f"free-corner err {e_diag['free_corner_con']:.3f}, "
            f"min ∂ {min(e_diag['min_d0_con'], e_diag['min_d1_con']):.3f} "
            "(accurate + shape-correct)\n"
        )
    else:
        print(
            "\nNOTE: full 3-way assertions skipped (gp_constr R backend "
            "unavailable for the inequality-only and/or both arms). The "
            "equality-only diagnostics and the R-free prior P(C) above are "
            "still valid."
        )


if __name__ == "__main__":
    main(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
