r"""Synthetic steady advection-diffusion plume (ground truth).

Operator
--------
    L c = -D Delta c + v . grad c + k c = S   on Omega = (0, 1)^2,

with homogeneous Dirichlet BC c = 0 on dOmega and a strictly positive Gaussian
emission source S >= 0. For an elliptic L with k >= 0, S >= 0 and c|_dOmega = 0
the maximum principle gives c > 0 in the interior, so the truth is strictly
positive.

Discretisation
--------------
Second-order central differences on a uniform N x N grid: 5-point Laplacian for
-D Delta, central first differences for v . grad, pointwise k for the decay.
Central advection is monotone (positivity-preserving) only at cell Peclet number
Pe = |v| h / (2 D) <= 1; the default (D, v, h) keep Pe well below 1, so the
solve is oscillation-free and the truth stays positive while remaining a
second-order-accurate solution of the *continuous* L the GP conditions on.

The truth check includes a manufactured-solution convergence test: for a known
smooth c*, solving L c = (L c*) with Dirichlet data c*|_dOmega recovers c* at
the expected O(h^2) rate, confirming the FD operator reproduces L to
discretisation order.

Run as a script for the truth check:

    PYTHONPATH=src:gp_constr:src/linpde_gp_constrained \
      python experiments/linpde_gp_constrained/air_quality_adv_diff/plume.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -----------------------------------------------------------------------------
# Physical parameters of the steady advection-diffusion problem.
# -----------------------------------------------------------------------------
def adv_diff_params() -> dict:
    """Constant coefficients (D, v, k) shared by truth solve and the GP operator.

    Chosen so the cell Peclet number stays well below 1 on the fine truth grid
    (central advection positive) and the plume is clearly advected downwind:
    v points right-and-up, the source sits near the lower-left (upwind) so the
    lower-left stays clean (the regime where positivity bites).
    """
    return dict(D=0.05, v=(0.6, 0.2), k=0.30)


def source(X: np.ndarray, sources=None) -> np.ndarray:
    """Gaussian emission source S(x, y) >= 0 at points X of shape (n, 2) -> (n,).

    Default: a single Gaussian at the upwind lower-left. ``sources`` overrides
    with a list of (x0, y0, amp, width) tuples.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if sources is None:
        # Amplitude nondimensionalises the plume to max ~ 1 (order unity).
        sources = [(0.28, 0.34, 6.5, 0.09)]
    out = np.zeros(X.shape[0])
    for x0, y0, amp, w in sources:
        r2 = (X[:, 0] - x0) ** 2 + (X[:, 1] - y0) ** 2
        out = out + amp * np.exp(-r2 / (2.0 * w * w))
    return out


# -----------------------------------------------------------------------------
# Finite-difference assembly of L on a uniform interior grid (Dirichlet BC).
# -----------------------------------------------------------------------------
def _assemble_L(n: int, D: float, v, k: float) -> tuple[sp.csr_matrix, np.ndarray]:
    """Assemble the (n-2)^2 interior FD matrix for L with homogeneous Dirichlet.

    Returns (A, Xint) where A is the sparse operator on interior unknowns and
    Xint is the (m, 2) array of interior node coordinates (row-major in [i, j]).
    """
    h = 1.0 / (n - 1)
    coord = np.linspace(0.0, 1.0, n)
    vx, vy = float(v[0]), float(v[1])

    idx = -np.ones((n, n), dtype=int)
    ii = np.arange(1, n - 1)
    II, JJ = np.meshgrid(ii, ii, indexing="ij")
    interior = np.column_stack([II.ravel(), JJ.ravel()])
    for m, (i, j) in enumerate(interior):
        idx[i, j] = m
    m_int = interior.shape[0]

    Xint = np.column_stack([coord[interior[:, 0]], coord[interior[:, 1]]])

    rows, cols, vals = [], [], []

    def add(r, c, val):
        rows.append(r)
        cols.append(c)
        vals.append(val)

    inv_h2 = 1.0 / (h * h)
    inv_2h = 1.0 / (2.0 * h)
    for m, (i, j) in enumerate(interior):
        # -D Delta : 5-point stencil.
        add(m, m, 4.0 * D * inv_h2 + k)
        for di, dj, adv in ((-1, 0, -vx), (1, 0, vx), (0, -1, -vy), (0, 1, vy)):
            ni, nj = i + di, j + dj
            # -D term off-diagonal.
            lap_off = -D * inv_h2
            # central advection: +adv * inv_2h on the (+) neighbour.
            adv_off = adv * inv_2h
            if 0 < ni < n - 1 and 0 < nj < n - 1:
                add(m, idx[ni, nj], lap_off + adv_off)
            # Dirichlet neighbour (boundary) contributes 0 to RHS (c=0 there).
    A = sp.csr_matrix((vals, (rows, cols)), shape=(m_int, m_int))
    return A, Xint


def solve_truth(n_fine: int = 81, sources=None, params: dict | None = None) -> dict:
    """Solve L c = S on the fine grid with homogeneous Dirichlet BC.

    Returns a dict with the full (n_fine, n_fine) field ``C`` (with zero
    boundary), axis ``coord``, the interior residual infinity norm, and the
    interior minimum (strict positivity check).
    """
    p = params or adv_diff_params()
    D, v, k = p["D"], p["v"], p["k"]
    A, Xint = _assemble_L(n_fine, D, v, k)
    S = source(Xint, sources=sources)
    c_int = spla.spsolve(A.tocsc(), S)

    C = np.zeros((n_fine, n_fine))
    n = n_fine
    ii = np.arange(1, n - 1)
    II, JJ = np.meshgrid(ii, ii, indexing="ij")
    C[II.ravel(), JJ.ravel()] = c_int

    resid = float(np.max(np.abs(A @ c_int - S)))
    return dict(
        C=C,
        coord=np.linspace(0.0, 1.0, n_fine),
        n_fine=n_fine,
        c_min_interior=float(c_int.min()),
        c_max=float(c_int.max()),
        residual_inf=resid,
        params=p,
    )


# -----------------------------------------------------------------------------
# Bilinear sampler of the fine-grid truth at arbitrary points.
# -----------------------------------------------------------------------------
def make_truth_sampler(truth: dict):
    """Return ``f(X) -> c_true(X)`` bilinearly interpolating the fine truth.

    X has shape (n, 2) in [0, 1]^2. Used to place monitors and to score the GP
    reconstruction on the evaluation grid.
    """
    C = truth["C"]
    coord = truth["coord"]
    n = truth["n_fine"]
    h = 1.0 / (n - 1)

    def f(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        xs = np.clip(X[:, 0], 0.0, 1.0)
        ys = np.clip(X[:, 1], 0.0, 1.0)
        gx = np.clip((xs / h).astype(int), 0, n - 2)
        gy = np.clip((ys / h).astype(int), 0, n - 2)
        tx = (xs - coord[gx]) / h
        ty = (ys - coord[gy]) / h
        c00 = C[gx, gy]
        c10 = C[gx + 1, gy]
        c01 = C[gx, gy + 1]
        c11 = C[gx + 1, gy + 1]
        return (
            c00 * (1 - tx) * (1 - ty)
            + c10 * tx * (1 - ty)
            + c01 * (1 - tx) * ty
            + c11 * tx * ty
        )

    return f


# -----------------------------------------------------------------------------
# Manufactured-solution convergence test for the FD operator.
# -----------------------------------------------------------------------------
def _manufactured_convergence(params: dict, grids=(21, 41, 81)) -> list[tuple]:
    """Solve L c = L c* with Dirichlet c*|_bdry for a known smooth c*.

    c*(x, y) = sin(pi x) sin(pi y) (vanishes on dOmega, so the homogeneous-BC
    solver applies directly). Returns [(n, h, max_err)]; the error must fall at
    ~O(h^2).
    """
    D, v, k = params["D"], params["v"], params["k"]
    vx, vy = v

    def cstar(X):
        x, y = X[:, 0], X[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def Lcstar(X):
        x, y = X[:, 0], X[:, 1]
        sx, sy = np.sin(np.pi * x), np.sin(np.pi * y)
        cx, cy = np.cos(np.pi * x), np.cos(np.pi * y)
        lap = -2.0 * (np.pi**2) * sx * sy
        gradx = np.pi * cx * sy
        grady = np.pi * sx * cy
        return -D * lap + vx * gradx + vy * grady + k * sx * sy

    out = []
    for n in grids:
        A, Xint = _assemble_L(n, D, v, k)
        rhs = Lcstar(Xint)
        c_int = spla.spsolve(A.tocsc(), rhs)
        err = float(np.max(np.abs(c_int - cstar(Xint))))
        out.append((n, 1.0 / (n - 1), err))
    return out


def main() -> None:
    p = adv_diff_params()
    print("--- air_quality_adv_diff truth check (truth + FD operator) ---")
    print(f"params: D={p['D']}, v={p['v']}, k={p['k']}")
    h_fine = 1.0 / (81 - 1)
    pe = float(np.hypot(*p["v"]) * h_fine / (2.0 * p["D"]))
    print(f"cell Peclet number on n=81 grid: {pe:.4f}  (need <= 1 for positivity)")
    assert pe <= 1.0, f"cell Peclet {pe:.3f} > 1; reduce v or h, or raise D."

    conv = _manufactured_convergence(p)
    print("manufactured-solution convergence (c* = sin pi x sin pi y):")
    prev = None
    rates = []
    for n, h, err in conv:
        rate = (
            ""
            if prev is None
            else f"  rate ~ {np.log(prev[1] / err) / np.log(prev[0] / h):.2f}"
        )
        print(f"  n={n:>3}  h={h:.4f}  max|c_FD - c*|={err:.3e}{rate}")
        if prev is not None:
            rates.append(np.log(prev[1] / err) / np.log(prev[0] / h))
        prev = (h, err)
    assert (
        min(rates) > 1.7
    ), f"FD convergence rate {min(rates):.2f} < 1.7 (expected ~2)."

    truth = solve_truth(n_fine=81)
    print(
        f"truth plume: min(interior)={truth['c_min_interior']:.4e}, "
        f"max={truth['c_max']:.4f}, FD residual_inf={truth['residual_inf']:.2e}"
    )
    assert truth["c_min_interior"] > 0.0, "truth not strictly positive in interior."
    print("truth check PASS")


if __name__ == "__main__":
    main()
