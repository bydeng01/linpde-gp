"""
Empirically verify gp_constr's Ki0/Kij sign and argument conventions on a
hand-rolled RBF, by finite-difference check against numerical derivatives of K.

Ground-truth math:
  k(x, x') = sigma^2 * exp(-0.5 * sum_d (x_d - x'_d)^2 / ell_d^2)
  dk/dx_i        = -k * (x_i - x'_i) / ell_i^2
  d^2k/dx_i dx'_j = k * [delta_{ij}/ell_i^2 - (x_i-x'_i)(x_j-x'_j)/(ell_i^2 ell_j^2)]

gp_constr/GPConstr/kern.py defines (lines 207-234 for the RBF subclass):
  Ki0(X1, X2, i) = -0.5 * K(X1, X2) * Ri(X1, X2, i)
  Ri(X1, X2, i)  = (2/ell_i^2) * (X1[:,i] - X2[:,i].T)
  Kij(X1, X2, i, j) = K * ((1/4) * Ri(X1,X2,i) * Ri(X2,X1,j).T + F)
                       where F = 1/ell_i^2 if i==j else 0

We reimplement these without sklearn and finite-difference them.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Replicated gp_constr RBF (lines 11-234 of gp_constr/GPConstr/kern.py),
# stripped of sklearn dep (we use np.subtract.outer for distances).
# -----------------------------------------------------------------------------


class kernel_RBF_replicated:
    def __init__(self, variance, lengthscale):
        self.lengthscale = np.atleast_1d(np.asarray(lengthscale, dtype=float))
        self.variance = float(variance)
        self.dim = len(self.lengthscale)

    def _euclidean_dist_squared(self, X1, X2):
        # ||X1 - X2||^2, shape (n1, n2)
        diff = X1[:, None, :] - X2[None, :, :]
        return np.sum(diff * diff, axis=-1)

    def R(self, X1, X2):
        return self._euclidean_dist_squared(
            X1 / self.lengthscale, X2 / self.lengthscale
        )

    def r(self, X1, X2):
        return np.sqrt(self.R(X1, X2))

    def K(self, X1, X2):
        return self.variance * np.exp(-0.5 * self.R(X1, X2))

    def Ri(self, X1, X2, i):
        # (2/ell_i^2) * (X1[:,i] - X2[:,i].T)
        return (2 / self.lengthscale[i] ** 2) * (
            X1[:, i].reshape(-1, 1) - X2[:, i].reshape(-1, 1).T
        )

    def Ki0(self, X1, X2, i):
        return -0.5 * self.K(X1, X2) * self.Ri(X1, X2, i)

    def Kij(self, X1, X2, i, j):
        F = 1 / self.lengthscale[i] ** 2 if i == j else 0
        K = self.K(X1, X2)
        return K * ((1 / 4) * self.Ri(X1, X2, i) * self.Ri(X2, X1, j).T + F)


# -----------------------------------------------------------------------------
# Independent ground-truth RBF using closed-form derivatives only (no
# reuse of gp_constr's Ri trick, so any convention bug shows up).
# -----------------------------------------------------------------------------


def rbf_K(X1, X2, variance, lengthscale):
    diff = X1[:, None, :] - X2[None, :, :]
    sqd = np.sum((diff / lengthscale) ** 2, axis=-1)
    return variance * np.exp(-0.5 * sqd)


def rbf_dKdx_i(X1, X2, variance, lengthscale, i):
    """Analytic dK/d(X1)_i = -K * (X1_i - X2_i) / ell_i^2."""
    K = rbf_K(X1, X2, variance, lengthscale)
    return -K * (X1[:, i : i + 1] - X2[:, i : i + 1].T) / lengthscale[i] ** 2


def rbf_d2Kdxidxj(X1, X2, variance, lengthscale, i, j):
    """Analytic d^2 K / d(X1)_i d(X2)_j."""
    K = rbf_K(X1, X2, variance, lengthscale)
    delta = 1.0 if i == j else 0.0
    cross = (X1[:, i : i + 1] - X2[:, i : i + 1].T) * (
        X1[:, j : j + 1] - X2[:, j : j + 1].T
    )
    return K * (
        delta / lengthscale[i] ** 2
        - cross / (lengthscale[i] ** 2 * lengthscale[j] ** 2)
    )


# -----------------------------------------------------------------------------
# Finite-difference helpers
# -----------------------------------------------------------------------------


def fd_Ki0(K_fn, X1, X2, i, h=1e-5):
    """Central difference for d K / d (X1)_i."""
    Xp = X1.copy()
    Xp[:, i] += h
    Xm = X1.copy()
    Xm[:, i] -= h
    return (K_fn(Xp, X2) - K_fn(Xm, X2)) / (2 * h)


def fd_Kij(K_fn, X1, X2, i, j, h=1e-3):
    """Central difference for d^2 K / d(X1)_i d(X2)_j (4-point stencil)."""

    def Kpp():
        a = X1.copy()
        a[:, i] += h
        b = X2.copy()
        b[:, j] += h
        return K_fn(a, b)

    def Kpm():
        a = X1.copy()
        a[:, i] += h
        b = X2.copy()
        b[:, j] -= h
        return K_fn(a, b)

    def Kmp():
        a = X1.copy()
        a[:, i] -= h
        b = X2.copy()
        b[:, j] += h
        return K_fn(a, b)

    def Kmm():
        a = X1.copy()
        a[:, i] -= h
        b = X2.copy()
        b[:, j] -= h
        return K_fn(a, b)

    return (Kpp() - Kpm() - Kmp() + Kmm()) / (4 * h * h)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def main():
    rng = np.random.default_rng(0)
    print("=" * 70)
    print("gp_constr CONVENTION VERIFICATION (numpy-only)")
    print("=" * 70)

    # Test 1: 1D
    print("\n[Test 1] 1D, n1=4, n2=3")
    variance, ell = 1.7, np.array([0.8])
    X1 = rng.uniform(-1, 1, (4, 1))
    X2 = rng.uniform(-1, 1, (3, 1))

    k = kernel_RBF_replicated(variance, ell)

    # K matches independent ground truth
    err_K = np.max(np.abs(k.K(X1, X2) - rbf_K(X1, X2, variance, ell)))
    print(f"  ||K_gpconstr - K_groundtruth||_inf = {err_K:.2e}")
    assert err_K < 1e-12

    # Ki0 matches finite difference of K
    Ki0_gp = k.Ki0(X1, X2, 0)
    Ki0_fd = fd_Ki0(k.K, X1, X2, 0)
    Ki0_an = rbf_dKdx_i(X1, X2, variance, ell, 0)
    print(
        f"  ||Ki0 - FD(K)||_inf      = {np.max(np.abs(Ki0_gp - Ki0_fd)):.2e}  (gp_constr vs FD)"
    )
    print(
        f"  ||Ki0 - analytic||_inf   = {np.max(np.abs(Ki0_gp - Ki0_an)):.2e}  (gp_constr vs analytic)"
    )
    assert np.max(np.abs(Ki0_gp - Ki0_fd)) < 1e-7
    assert np.max(np.abs(Ki0_gp - Ki0_an)) < 1e-12

    # Kij matches FD of K
    Kij_gp = k.Kij(X1, X2, 0, 0)
    Kij_fd = fd_Kij(k.K, X1, X2, 0, 0)
    Kij_an = rbf_d2Kdxidxj(X1, X2, variance, ell, 0, 0)
    print(f"  ||Kij(0,0) - FD(K)||_inf    = {np.max(np.abs(Kij_gp - Kij_fd)):.2e}")
    print(f"  ||Kij(0,0) - analytic||_inf = {np.max(np.abs(Kij_gp - Kij_an)):.2e}")
    assert np.max(np.abs(Kij_gp - Kij_fd)) < 1e-5
    assert np.max(np.abs(Kij_gp - Kij_an)) < 1e-12

    # Test 2: 2D, mixed Kij(0, 1)
    print("\n[Test 2] 2D, n1=5, n2=4, mixed Kij(0,1)")
    variance, ell = 2.3, np.array([0.7, 1.1])
    X1 = rng.uniform(-1, 1, (5, 2))
    X2 = rng.uniform(-1, 1, (4, 2))
    k = kernel_RBF_replicated(variance, ell)

    Kij_gp = k.Kij(X1, X2, 0, 1)
    Kij_fd = fd_Kij(k.K, X1, X2, 0, 1)
    Kij_an = rbf_d2Kdxidxj(X1, X2, variance, ell, 0, 1)
    print(f"  ||Kij(0,1) - FD(K)||_inf    = {np.max(np.abs(Kij_gp - Kij_fd)):.2e}")
    print(f"  ||Kij(0,1) - analytic||_inf = {np.max(np.abs(Kij_gp - Kij_an)):.2e}")
    assert np.max(np.abs(Kij_gp - Kij_fd)) < 1e-5
    assert np.max(np.abs(Kij_gp - Kij_an)) < 1e-12

    # Test 3: Sign verification: dK/dx_i is NEGATIVE when x_i > x'_i
    # (kernel is decreasing in distance; moving x further away decreases k)
    print("\n[Test 3] Sign of Ki0: should be negative when X1_i > X2_i")
    X1 = np.array([[1.0]])
    X2 = np.array([[0.0]])
    k = kernel_RBF_replicated(1.0, np.array([1.0]))
    print(f"  X1=[1.0], X2=[0.0]: Ki0(0) = {k.Ki0(X1, X2, 0)[0,0]:+.4f}")
    assert k.Ki0(X1, X2, 0)[0, 0] < 0, "Sign convention bug!"

    # And POSITIVE when X1_i < X2_i
    X1 = np.array([[-1.0]])
    X2 = np.array([[0.0]])
    print(f"  X1=[-1.0], X2=[0.0]: Ki0(0) = {k.Ki0(X1, X2, 0)[0,0]:+.4f}")
    assert k.Ki0(X1, X2, 0)[0, 0] > 0, "Sign convention bug!"

    # Test 4: Kii_diag(X, i) = diag of Kij(X, X, i, i) and equals variance/ell_i^2
    print("\n[Test 4] Diagonal: Kii_diag(X, i) at coincident points")
    variance, ell = 1.5, np.array([0.9, 1.3])
    X = rng.uniform(-1, 1, (6, 2))
    k = kernel_RBF_replicated(variance, ell)

    diag_via_Kij = np.diag(k.Kij(X, X, 0, 0))
    expected = variance / ell[0] ** 2  # closed form for RBF
    print(f"  diag(Kij(X,X,0,0)) = {diag_via_Kij}")
    print(f"  expected (constant) = {expected:.6f}")
    assert np.allclose(
        diag_via_Kij, expected, atol=1e-12
    ), f"Kii_diag mismatch: {diag_via_Kij} vs {expected}"

    print("\n" + "=" * 70)
    print("ALL CONVENTION CHECKS PASSED")
    print("Confirmed:")
    print("  Ki0 = dK/d(X1)_i  (FIRST argument differentiated)")
    print("  Kij = d^2K/d(X1)_i d(X2)_j  (FIRST arg by i, SECOND arg by j)")
    print("  Sign: Ki0 < 0 when X1_i > X2_i (kernel decreases with separation)")
    print("=" * 70)


if __name__ == "__main__":
    main()
