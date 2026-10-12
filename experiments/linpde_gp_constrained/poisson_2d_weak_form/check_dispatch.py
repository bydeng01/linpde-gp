"""Dispatch check: confirm the 2D double-Laplacian ExpQuad dispatch is closed form.

Run inside the docker container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \
      python experiments/linpde_gp_constrained/poisson_2d_weak_form/check_dispatch.py

Expected output (closed form — NOT a JaxLambdaCovarianceFunction):

    inner: <type> -> core ExpQuad_Identity_WeightedLaplacian
    outer: <type> -> core ExpQuad_WeightedLaplacian_WeightedLaplacian
    dispatch check PASS

Note: with a ``1.0 * ExpQuad`` amplitude (as in the demo) the result is a
``JaxScaledCovarianceFunction`` *wrapping* the closed form — the Laplacian
dispatch pulls the scalar out and applies to the inner ExpQuad. That is still a
closed form (its ``_evaluate`` is numpy, no JAX tracing); we unwrap the scalar
wrapper(s) and check the *core* type. The genuine failure mode is a
``JaxLambdaCovarianceFunction`` core (the autodiff fallback), which would force a
switch to a half-integer Matern prior.
"""

import numpy as np
import probnum as pn

from linpde_gp.linfuncops.diffops import WeightedLaplacian
from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad


def _core(k):
    """Unwrap scalar/sum wrappers to reach the underlying covariance function."""
    seen = 0
    while hasattr(k, "covfunc") and seen < 16:
        k = k.covfunc
        seen += 1
    return k


def main() -> None:
    prior_cov = 1.0 * LinpdeExpQuad(input_shape=(2,), lengthscales=0.25)

    inner = WeightedLaplacian(np.ones(2))(prior_cov, argnum=1)
    print("inner:", type(inner).__name__, "-> core", type(_core(inner)).__name__)

    k_DD = WeightedLaplacian(np.ones(2))(inner, argnum=0)
    print("outer:", type(k_DD).__name__, "-> core", type(_core(k_DD)).__name__)

    core_name = type(_core(k_DD)).__name__
    assert core_name != "JaxLambdaCovarianceFunction", (
        "dispatch check FAILED: chain fell back to the JAX autodiff path "
        "(JaxLambdaCovarianceFunction). Switch to a half-integer Matern prior "
        "with complete 2D dispatch."
    )
    assert (
        core_name == "ExpQuad_WeightedLaplacian_WeightedLaplacian"
    ), f"dispatch check FAILED: expected closed-form core, got {core_name}."

    # Spot-check the numpy _evaluate path actually returns finite numbers (no
    # JAX tracing) on a small 2D grid.
    X = np.array([[0.2, 0.3], [0.7, 0.1], [0.5, 0.9]])
    G = np.asarray(k_DD(X[:, None, :], X[None, :, :]))
    assert G.shape == (3, 3) and np.all(np.isfinite(G)), G
    print("dispatch check PASS")


if __name__ == "__main__":
    main()
