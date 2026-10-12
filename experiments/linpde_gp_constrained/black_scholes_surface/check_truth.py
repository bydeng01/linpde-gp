"""Truth check: Black-Scholes truth + Dupire residual, no GP.

Three checks on the synthetic ground truth used by run.py (sigma = 0.2,
r = 0.02, S0 = 1, zero dividends):

  1. put-call parity  c - p = 1 - e^{x - rT}  to 1e-12 on a dense grid;
  2. known reference values (ATM 1y and two wing points) against an
     independent Gauss-Hermite quadrature of the risk-neutral expectation
     E[(e^{x_T} - e^x)^+] to 1e-9;
  3. central-FD Dupire residual of the truth,
         L c = d_T c - (sigma^2/2)(d_xx c - d_x c) + r d_x c,
     below 1e-6 on the probe interior x in [ln 0.75, ln 1.25],
     T in [0.1, 0.95] (away from the small-T corner where the FD truncation
     term h^2 * d4c/dx4 / 12 ~ h^2 / (sigma sqrt(T))^3 blows up; the
     analytic residual is identically zero everywhere, which check 3 also
     verifies to 1e-12 via the closed-form Greeks).

Run from the repo root inside the container:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \\
      python experiments/linpde_gp_constrained/black_scholes_surface/check_truth.py
"""

import numpy as np
from run import (
    R_RATE,
    SIGMA,
    T_HI,
    T_LO,
    X_HI,
    X_LO,
    bs_call,
    bs_call_dT,
    bs_call_dx,
    bs_call_dxx,
    bs_put,
    dupire_residual_fd,
)


def _quadrature_call(x: float, T: float, sigma: float, r: float) -> float:
    """Risk-neutral e^{-rT} E[(S_T - K)^+] / S0 by adaptive quadrature.

    S_T / S0 = exp((r - sigma^2/2) T + sigma sqrt(T) Z), Z ~ N(0,1). The
    payoff kink is removed analytically: integrate the smooth integrand from
    the exercise boundary z* (where S_T = K) upward. Independent of the
    closed-form Phi-based pricer.
    """
    from scipy.integrate import quad

    sT = sigma * np.sqrt(T)
    mu = (r - 0.5 * sigma**2) * T
    z_star = (x - mu) / sT  # S_T(z*) = e^x

    def integrand(z):
        return (
            (np.exp(mu + sT * z) - np.exp(x))
            * np.exp(-0.5 * z**2)
            / np.sqrt(2.0 * np.pi)
        )

    val, _ = quad(integrand, z_star, 12.0, epsabs=1e-13, epsrel=1e-13)
    return float(np.exp(-r * T) * val)


def main() -> None:
    rng_grid_x = np.linspace(X_LO, X_HI, 41)
    rng_grid_T = np.linspace(T_LO, T_HI, 31)
    XX, TT = np.meshgrid(rng_grid_x, rng_grid_T, indexing="ij")
    X = np.column_stack([XX.ravel(), TT.ravel()])

    # --- 1. Put-call parity.
    parity_err = np.max(
        np.abs(bs_call(X) - bs_put(X) - (1.0 - np.exp(X[:, 0] - R_RATE * X[:, 1])))
    )
    assert parity_err < 1e-12, f"put-call parity violated: {parity_err:.3e}"
    print(f"parity      : max |c - p - (1 - e^(x-rT))| = {parity_err:.3e}")

    # --- 2. Known values vs independent quadrature (ATM 1y + both wings).
    probes = [(0.0, 1.0), (np.log(0.75), 0.5), (np.log(1.25), 0.5)]
    for x, T in probes:
        c_cf = float(bs_call(np.array([[x, T]]))[0])
        c_q = _quadrature_call(x, T, SIGMA, R_RATE)
        assert abs(c_cf - c_q) < 1e-9, (
            f"closed form vs quadrature at (x={x:.3f}, T={T}): "
            f"{c_cf:.12f} vs {c_q:.12f}"
        )
        print(
            f"reference   : c({x:+.4f}, {T:.2f}) = {c_cf:.9f}  "
            f"(quadrature {c_q:.9f})"
        )

    # --- 3a. Analytic Dupire residual (closed-form Greeks) — exact zero.
    x, T = X[:, 0], X[:, 1]
    res_an = np.max(
        np.abs(
            bs_call_dT(X)
            - 0.5 * SIGMA**2 * (bs_call_dxx(X) - bs_call_dx(X))
            + R_RATE * bs_call_dx(X)
        )
    )
    assert res_an < 1e-12, f"analytic Dupire residual: {res_an:.3e}"
    print(f"residual    : analytic max |L c| = {res_an:.3e}")

    # --- 3b. FD Dupire residual on the probe interior.
    a_x = np.linspace(np.log(0.75), np.log(1.25), 25)
    a_T = np.linspace(0.10, 0.95, 25)
    GX, GT = np.meshgrid(a_x, a_T, indexing="ij")
    Xi = np.column_stack([GX.ravel(), GT.ravel()])
    res_fd = np.max(np.abs(dupire_residual_fd(bs_call, Xi)))
    assert res_fd < 1e-6, f"FD Dupire residual {res_fd:.3e} >= 1e-6"
    print(f"residual    : FD max |L c| = {res_fd:.3e} (interior, < 1e-6)")

    print("truth check PASS")


if __name__ == "__main__":
    main()
