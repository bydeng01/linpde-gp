"""Dispatch check: closed-form ExpQuad dispatch for the composite Dupire operator.

The collocation posterior needs the chain L k L' for

    L = d_T - (sigma^2/2)(d_xx - d_x) + r d_x
      = DirectionalDerivative([sigma^2/2 + r, 1]) + WeightedLaplacian([-sigma^2/2, 0]),

i.e. (i) L(k, argnum=1), (ii) L(L(k, argnum=1), argnum=0), and -- for the
kernel adapter's Ki0/Kij and the L-on-differentiated-prior path of
_eval_dki_kLa -- (iii) L(D_i(k, argnum=0), argnum=1). SumLinearFunctionOperator
distributes over its summands and the LDO registry distributes over
JaxSum/JaxScaled wrappers, so every leaf must land on one of the closed-form
cores ExpQuad_{Identity,DirectionalDerivative,WeightedLaplacian}_* --- never on
JaxLambdaCovarianceFunction (the autodiff fallback). The probe asserts that and
cross-checks the numerics against central finite differences of the anisotropic
ExpQuad on a small grid.

Run from the repo root inside the container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/black_scholes_surface/check_dispatch.py
"""

import numpy as np
from run import R_RATE, SIGMA, dupire_operator

from linpde_gp.linfuncops.diffops import DirectionalDerivative
from linpde_gp.randprocs.covfuncs import ExpQuad as LinpdeExpQuad


def _leaf_cores(k, out=None):
    """Collect the leaf covariance cores under scalar/sum wrappers."""
    if out is None:
        out = []
    if hasattr(k, "summands"):
        for s in k.summands:
            _leaf_cores(s, out)
        return out
    seen = 0
    while hasattr(k, "covfunc") and seen < 16:
        k = k.covfunc
        seen += 1
    if hasattr(k, "summands"):
        return _leaf_cores(k, out)
    out.append(k)
    return out


def _assert_closed_form(k, label):
    cores = _leaf_cores(k)
    names = sorted({type(c).__name__ for c in cores})
    print(f"{label}: {type(k).__name__} -> leaf cores {names}")
    assert "JaxLambdaCovarianceFunction" not in names, (
        f"dispatch check FAILED at {label}: chain fell back to the JAX autodiff path. "
        "Implement local closed-form bivariate handlers in this directory "
        "(do not patch linpde-gp)."
    )


def _fd_check(k_op, base_eval, op_args, X, h=1e-5, tol=1e-5):
    """Compare the dispatched operator kernel against FD on the base kernel.

    op_args = (argnum, coeff_dx, coeff_dT, coeff_dxx): the FD model applies
    coeff_dx d_x + coeff_dT d_T + coeff_dxx d_xx to the requested argument of
    the bivariate base kernel.
    """
    argnum, c_dx, c_dT, c_dxx = op_args

    def shifted(dx0, dT0):
        Xs = np.array(X, copy=True)
        Xs[:, 0] += dx0
        Xs[:, 1] += dT0
        return base_eval(Xs, X) if argnum == 0 else base_eval(X, Xs)

    f0 = base_eval(X, X)
    d_x = (shifted(h, 0) - shifted(-h, 0)) / (2 * h)
    d_T = (shifted(0, h) - shifted(0, -h)) / (2 * h)
    d_xx = (shifted(h, 0) - 2 * f0 + shifted(-h, 0)) / h**2
    want = c_dx * d_x + c_dT * d_T + c_dxx * d_xx
    got = np.asarray(k_op(X[:, None, :], X[None, :, :]))
    err = np.max(np.abs(got - want))
    assert err < tol, f"FD mismatch: {err:.3e} >= {tol}"
    return err


def main() -> None:
    prior_cov = 1.0 * LinpdeExpQuad(input_shape=(2,), lengthscales=np.array([0.3, 0.5]))
    L = dupire_operator(SIGMA, R_RATE)
    c1 = 0.5 * SIGMA**2 + R_RATE  # d_x coefficient
    c2 = -0.5 * SIGMA**2  # d_xx coefficient

    # (i) inner: L applied to argument 1.
    kL1 = L(prior_cov, argnum=1)
    _assert_closed_form(kL1, "inner  L k (argnum=1)")

    # (ii) outer: L applied to argument 0 of the inner result (the PDE gram).
    kL0L1 = L(kL1, argnum=0)
    _assert_closed_form(kL0L1, "outer  L k L' (argnum=0)")

    # (iii) adapter path: D_i k (argnum=0), then L on argnum=1 (Ki0/Kij and
    # _eval_dki_kLa use exactly this ordering).
    for i, e in enumerate(np.eye(2)):
        Di = DirectionalDerivative(e)
        kDi = Di(prior_cov, argnum=0)
        _assert_closed_form(L(kDi, argnum=1), f"mixed  L (D_{'xT'[i]} k)")

    # Numeric spot checks vs FD on a small grid.
    X = np.array([[-0.2, 0.3], [0.0, 0.6], [0.15, 0.85], [-0.05, 0.2]])

    def base(Xa, Xb):
        return np.asarray(prior_cov(Xa[:, None, :], Xb[None, :, :]))

    err1 = _fd_check(kL1, base, (1, c1, 1.0, c2), X)
    print(f"numeric: |L k (argnum=1) - FD| = {err1:.3e}")

    def baseL1(Xa, Xb):
        return np.asarray(kL1(Xa[:, None, :], Xb[None, :, :]))

    err0 = _fd_check(kL0L1, baseL1, (0, c1, 1.0, c2), X)
    print(f"numeric: |L k L' (argnum=0) - FD| = {err0:.3e}")

    G = np.asarray(kL0L1(X[:, None, :], X[None, :, :]))
    w = np.linalg.eigvalsh(0.5 * (G + G.T))
    assert np.all(np.isfinite(G)) and w.min() > -1e-10, (G, w)
    print(f"gram spot-check: finite, min eig {w.min():.3e} >= 0")
    print("dispatch check PASS")


if __name__ == "__main__":
    main()
