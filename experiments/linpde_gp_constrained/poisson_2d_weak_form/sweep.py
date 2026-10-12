"""
R-free tuning sweep for the 2D *combination showcase* (partial boundary data).

Goal: find a config in which equality information ALONE is insufficient, so
that adding the monotonicity inequality measurably helps. The signals to read
(all R-free — equality-only posterior + the MC P(C|E)):

  * ``maxabs_unc``  — max |m_E - u_true| for the equality-only posterior. We
    WANT this appreciable (the equality side is under-determined), say
    >~ 0.05; it is NOT a pass/fail check here.
  * ``min dx/dy``   — most-negative ∂_x / ∂_y of the equality-only mean. We
    want these clearly NEGATIVE: a non-monotone equality mean is exactly what
    the inequality can correct (monotonicity is scale-free, so it only helps
    when the mean actually dips).
  * ``P(C|E)_mc``   — joint MC P(C|E) over the constrained axes; want in
    (0.05, 0.95).

A good showcase row has appreciable ``maxabs_unc``, clearly negative ``min
dx``/``min dy``, and ``P(C|E)_mc`` in range. Set it as run.solve()'s defaults,
then ONE R run confirms RMSE_con < RMSE_unc and the combined recovery.

Run:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/poisson_2d_weak_form/sweep.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as run_module  # noqa: E402

# Showcase is fixed (two_edges, 3×3 collocation, ℓ=0.5, both axes); we scan the
# levers that raise P(C|E) for the paper — the number of virtual points per axis
# (fewer => fewer joint constraints => higher P(C|E)) and the virtual-grid
# extent (pulling in off the low-slope edges raises P(C|E)). The MC P(C|E) here
# is an accurate R-free proxy for the exact value (they matched 0.052 vs 0.053).
# (n_v_axis, Xv_lo, Xv_hi)
CONFIGS = [
    (4, 0.10, 0.90),  # current default -> P(C|E)_mc ~ 0.05 (borderline)
    (3, 0.10, 0.90),  # fewer constraints (9/axis, joint dim 18)
    (3, 0.15, 0.85),
    (3, 0.20, 0.80),
    (4, 0.15, 0.85),
    (4, 0.20, 0.80),
    (5, 0.15, 0.85),  # denser, for reference
]
_SCHEME, _NC, _NBC, _ELL, _AXES = "two_edges", 5, 7, 0.50, (0, 1)


def main() -> None:
    print(f"  fixed: scheme={_SCHEME}, colloc={(_NC-2)**2}, ell={_ELL}, axes={_AXES}")
    print(
        f"{'n_v':>4} {'Xv_lo':>6} {'Xv_hi':>6} {'jointdim':>8} | "
        f"{'maxabs_unc':>10} {'min dx':>9} {'min dy':>9} "
        f"{'P(C|E)mc':>9}  note"
    )
    print("-" * 80)
    for nv, lo, hi in CONFIGS:
        data = run_module.solve(
            lengthscale=_ELL,
            n_colloc_axis=_NC,
            n_bc_axis=_NBC,
            bc_scheme=_SCHEME,
            constrain_axes=_AXES,
            n_v_axis=nv,
            Xv_lo=lo,
            Xv_hi=hi,
            n_eval_axis=20,
            compute_constrained=False,
        )
        d = data["diagnostics"]
        maxabs = d["max_abs_unc"]
        mdx = d.get("min_d0_unc", float("nan"))
        mdy = d.get("min_d1_unc", float("nan"))
        pmc = d["p_ce_mc"]
        jointdim = len(_AXES) * nv * nv
        # "Comfortable" for the paper: still strongly under-determined (maxabs
        # large) and non-monotone, with P(C|E) clear of the 0.05 floor.
        good = (maxabs > 0.15) and (min(mdx, mdy) < -0.05) and (0.12 < pmc < 0.6)
        note = "<-- comfortable" if good else ""
        print(
            f"{nv:>4} {lo:>6.2f} {hi:>6.2f} {jointdim:>8} | "
            f"{maxabs:>10.4f} {mdx:>9.4f} {mdy:>9.4f} {pmc:>9.4f}  {note}"
        )


if __name__ == "__main__":
    main()
