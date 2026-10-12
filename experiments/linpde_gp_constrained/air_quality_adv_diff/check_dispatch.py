r"""Dispatch check: ExpQuad x advection-diffusion cross-cov is closed form.

L = -D Delta + v . grad + k I. The collocation posterior needs the closed-form
chains

    L_t  k(x, t)              (kLa block),
    L_s L_t k(s, t)           (gram block),

for the ``ExpQuad`` prior. This probe applies ``L`` (built by
``adv_diff_operator.make_adv_diff_operator``) to ``ExpQuad`` on argnum=1 then
argnum=0 and asserts the core covariance is NOT the JAX autodiff fallback
(``JaxLambdaCovarianceFunction``), then cross-checks the numeric value against a
direct ``jax.hessian`` / ``jax.jvp`` evaluation of L_s L_t on the scalar kernel.

Run inside the container (no R backend needed):

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \
      python experiments/linpde_gp_constrained/air_quality_adv_diff/check_dispatch.py
"""

from __future__ import annotations

import os
import sys

import jax
from jax import numpy as jnp
import numpy as np

from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adv_diff_operator import make_adv_diff_operator  # noqa: E402

D, V, K = 0.05, (0.6, 0.2), 0.3
LS = 0.25


def _has_jax_fallback(k, seen=None) -> bool:
    """True if any node of the covariance tree is the JAX autodiff fallback.

    L is a ``SumLinearFunctionOperator``, so ``L_s L_t k`` is a sum of
    scaled closed-form ExpQuad-diffop covariances. We walk ``covfunc`` and
    ``summands`` and check no leaf is a ``JaxLambdaCovarianceFunction``.
    """
    if seen is None:
        seen = set()
    if id(k) in seen:
        return False
    seen.add(id(k))
    if type(k).__name__ == "JaxLambdaCovarianceFunction":
        return True
    for attr in ("covfunc",):
        if hasattr(k, attr):
            if _has_jax_fallback(getattr(k, attr), seen):
                return True
    for attr in ("summands", "covfuncs"):
        if hasattr(k, attr):
            for sub in getattr(k, attr):
                if _has_jax_fallback(sub, seen):
                    return True
    return False


def _expquad_jax(s, t, ls):
    """Scalar ExpQuad k(s, t) = exp(-||s - t||^2 / (2 ls^2)) on R^2 (jax)."""
    d = s - t
    return jnp.exp(-jnp.sum(d * d) / (2.0 * ls * ls))


def _L_apply(fn, arg_idx, s, t, D, v, k):
    """Apply L = -D Delta + v.grad + k I to ``fn`` in the ``arg_idx`` argument."""
    v = jnp.asarray(v)

    def scalar(x):
        return fn(x, t) if arg_idx == 0 else fn(s, x)

    x0 = s if arg_idx == 0 else t
    val = scalar(x0)
    hess = jax.hessian(scalar)(x0)
    lap = jnp.trace(hess)
    grad = jax.grad(scalar)(x0)
    return -D * lap + jnp.dot(v, grad) + k * val


def _LsLt_reference(s, t, D, v, k, ls):
    """L_s L_t k(s, t) by nested autodiff — ground truth for the closed form."""

    def Lt_only(s_, t_):
        return _L_apply(lambda a, b: _expquad_jax(a, b, ls), 1, s_, t_, D, v, k)

    return _L_apply(Lt_only, 0, s, t, D, v, k)


def main() -> None:
    prior_cov = 1.0 * LinpdeExpQuad(input_shape=(2,), lengthscales=LS)
    L = make_adv_diff_operator(D, V, K, dim=2)

    inner = L(prior_cov, argnum=1)
    print("inner:", type(inner).__name__)

    k_LsLt = L(inner, argnum=0)
    print("outer:", type(k_LsLt).__name__)

    assert not _has_jax_fallback(k_LsLt), (
        "dispatch check FAILED: L_s L_t ExpQuad contains a JAX autodiff leaf "
        "(JaxLambdaCovarianceFunction). Implement the closed-form bivariate "
        "operator locally (do not patch linpde-gp)."
    )
    print("no JaxLambdaCovarianceFunction leaf (closed-form dispatch).")

    # Numeric cross-check against nested-autodiff reference on a small grid.
    X = np.array([[0.2, 0.3], [0.7, 0.1], [0.5, 0.9], [0.35, 0.65]])
    G = np.asarray(k_LsLt(X[:, None, :], X[None, :, :]))
    assert G.shape == (4, 4) and np.all(np.isfinite(G)), G

    ref = np.array(
        [
            [
                float(_LsLt_reference(jnp.asarray(a), jnp.asarray(b), D, V, K, LS))
                for b in X
            ]
            for a in X
        ]
    )
    err = float(np.max(np.abs(G - ref)))
    print(f"max |closed-form - autodiff reference| = {err:.3e}")
    assert (
        err < 1e-6
    ), f"dispatch check FAILED: closed form disagrees with autodiff ({err:.3e})."

    # Also exercise the kLa block (L_t only) used for the cross-covariance.
    kLa = np.asarray(inner(X[:, None, :], X[None, :, :]))
    assert kLa.shape == (4, 4) and np.all(np.isfinite(kLa)), kLa

    print("dispatch check PASS")


if __name__ == "__main__":
    main()
