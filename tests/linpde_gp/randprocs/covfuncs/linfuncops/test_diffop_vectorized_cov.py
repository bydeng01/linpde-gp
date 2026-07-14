"""End-to-end tests for the vectorized (multi-output) diffop-on-covariance
path that powers ``HelmholtzReal2Operator.to_linfunctl(X)(gp)``.

Two tests:
  1. Shape smoke test — the original 2026-04 test, with the API call
     corrected (``cov`` is already a ``pn.linops.LinearOperator``, not a
     ``Covariance`` wrapper, on a non-conditional ``GaussianProcess``).
  2. Analytical block-structure test — verifies that the (8, 8) covariance
     equals what the math predicts for L K L^T with
     K = (1/2) k_scalar(x, x') · I_2 and
     L = [[d²+α, -β], [β, d²+α]].
"""

import jax
import jax.numpy as jnp
import numpy as np
import probnum as pn

import linpde_gp


def _build_setup():
    """Shared setup for both tests."""
    ell = 0.8
    alpha = 1.0
    beta = 0.2
    gp = pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(), output_shape=(2,)),
        cov=linpde_gp.randprocs.covfuncs.Real2FromScalarKernel(
            linpde_gp.randprocs.covfuncs.ExpQuad(input_shape=(), lengthscales=ell)
        ),
    )
    X = np.linspace(0.0, 1.0, 4)
    L = linpde_gp.linfuncops.diffops.HelmholtzReal2Operator(
        domain_shape=(), k_squared=alpha + 1j * beta
    ).to_linfunctl(X)
    return gp, X, L, ell, alpha, beta


def test_helmholtz_real2_covariance_matrix_shape():
    gp, X, L, _, _, _ = _build_setup()

    # ``L(gp)`` is a ``pn.randvars.Normal``. Its ``.cov`` is the linear
    # operator directly (a ``BlockMatrix`` here from the vector-output
    # operator). The earlier version of this test asked for ``.linop`` —
    # that attribute lives on the linpde_gp ``Covariance`` wrapper, which
    # the prior-GP path unwraps before storing on ``pn.randvars.Normal``
    # (see linpde_gp/randprocs/_gaussian_process/_lintransforms.py:15).
    cov_linop = L(gp).cov
    dense = cov_linop.todense()

    # 4 evaluation points × 2 output components → 8 × 8 covariance.
    assert dense.shape == (8, 8)


def test_helmholtz_real2_covariance_matches_analytical_block_structure():
    """Mathematical-soundness check on the (8, 8) covariance.

    For ``Real2FromScalarKernel`` the prior covariance is
    ``K(x, x') = (1/2) k(x, x') I_2`` with k = ExpQuad. The
    ``HelmholtzReal2Operator`` with ``k_squared = α + iβ`` is the matrix
    operator ``L = [[Δ + α, −β], [β, Δ + α]]``. Then::

        (L K Lᵀ)(x, x') = (1/2) [Σ_c L_{a,c}(x) L_{b,c}(x')] k(x, x'),

    and a short calculation (using the symmetry
    ``∂²_x k(x, x') = ∂²_{x'} k(x, x')`` for stationary k) gives:

      * (0, 0) and (1, 1) blocks: ``(Δ_x + α)(Δ_{x'} + α) k + β² k``
        scaled by 1/2.
      * (0, 1) and (1, 0) blocks: ``(β / 2)(Δ_x − Δ_{x'}) k = 0``.

    This test asserts both, against an independent ``jax.grad`` computation
    of the radial derivatives.
    """
    gp, X, L, ell, alpha, beta = _build_setup()
    cov_dense = L(gp).cov.todense()

    assert cov_dense.shape == (8, 8)

    # --- Independent analytical reference ---
    def k_scalar(x0, x1):
        return jnp.exp(-((x0 - x1) ** 2) / (2.0 * ell ** 2))

    d2_x = jax.grad(jax.grad(k_scalar, argnums=0), argnums=0)
    d2_y = jax.grad(jax.grad(k_scalar, argnums=1), argnums=1)
    d2_d2 = jax.grad(
        jax.grad(jax.grad(jax.grad(k_scalar, argnums=0), argnums=0), argnums=1),
        argnums=1,
    )

    K_diag = np.zeros((4, 4))
    X_jax = jnp.asarray(X)
    for i in range(4):
        for j in range(4):
            K_diag[i, j] = float(
                d2_d2(X_jax[i], X_jax[j])
                + alpha * d2_x(X_jax[i], X_jax[j])
                + alpha * d2_y(X_jax[i], X_jax[j])
                + (alpha ** 2 + beta ** 2) * k_scalar(X_jax[i], X_jax[j])
            )
    K_diag *= 0.5

    # JAX defaults to float32 — the round-off floor is around 1e-6.
    atol = 5e-6
    np.testing.assert_allclose(cov_dense[:4, :4], K_diag, atol=atol)
    np.testing.assert_allclose(cov_dense[4:, 4:], K_diag, atol=atol)
    # Off-diagonal blocks vanish by stationarity of k.
    np.testing.assert_allclose(cov_dense[:4, 4:], 0.0, atol=atol)
    np.testing.assert_allclose(cov_dense[4:, :4], 0.0, atol=atol)
    # Symmetry of the full covariance.
    np.testing.assert_allclose(cov_dense, cov_dense.T, atol=atol)
