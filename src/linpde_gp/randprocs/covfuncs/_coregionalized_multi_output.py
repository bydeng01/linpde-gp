"""Intrinsic Coregionalization Model (ICM) covariance function.

This is the Phase 8a building block for coregionalized (ICM/LMC) priors. It is
*purely additive*: it does not modify, subclass, or otherwise touch
:class:`IndependentMultiOutputCovarianceFunction` or any existing covariance
function. With ``B = np.eye(P)`` and a single shared base kernel it reduces, to
machine precision, to ``IndependentMultiOutputCovarianceFunction(base, ..., base)``
(see the reduction-identity unit test), which is the backward-compatibility
anchor for the whole phase.
"""

from typing import Optional

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike

from ._jax import JaxCovarianceFunction


class CoregionalizedMultiOutputCovarianceFunction(JaxCovarianceFunction):
    r"""Intrinsic Coregionalization Model (ICM) covariance function.

    Couples ``P`` scalar output channels through a single shared base
    covariance function ``base`` and a constant ``P x P`` symmetric
    positive-semidefinite coregionalization matrix ``B``:

    .. math::
        \mathrm{Cov}[u_c(x), u_{c'}(x')] = B[c, c'] \cdot \mathrm{base}(x, x').

    Equivalently, the Gram (cross-covariance) operator is the Kronecker
    product :math:`B \otimes K` where :math:`K` is the base kernel's Gram
    operator. Because ``B`` is constant in ``x``, any spatial linear operator
    ``L`` factors straight through: :math:`L_x (B \otimes k) = B \otimes (L_x k)`.
    This is what makes the Phase 8b dispatch handlers a thin wrapper around the
    base kernel's specialized operator handlers.

    Parameters
    ----------
    base
        Scalar-valued base covariance function
        (``base.output_shape_0 == base.output_shape_1 == ()``).
    B
        ``(P, P)`` symmetric positive-semidefinite coregionalization matrix.
    """

    def __init__(self, base: JaxCovarianceFunction, B: ArrayLike):
        if base.output_shape_0 != () or base.output_shape_1 != ():
            raise ValueError(
                "The base covariance function must be scalar-valued "
                f"(got output_shape_0={base.output_shape_0}, "
                f"output_shape_1={base.output_shape_1})."
            )

        B = np.asarray(B, dtype=np.double)
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise ValueError(f"B must be a square (P, P) matrix, got shape {B.shape}.")
        if not np.allclose(B, B.T):
            raise ValueError("B must be symmetric.")

        eigvals = np.linalg.eigvalsh(B)
        tol = 1e-8 * max(1.0, float(np.abs(eigvals).max()))
        if eigvals.min() < -tol:
            raise ValueError(
                "B must be positive semidefinite "
                f"(smallest eigenvalue {eigvals.min():.3e} < -{tol:.3e})."
            )

        self._base = base
        self._B = B
        self._B_cholesky = self._compute_cholesky(B, eigvals)

        super().__init__(
            input_shape=base.input_shape,
            output_shape_0=(B.shape[0],),
            output_shape_1=(B.shape[0],),
        )

    @staticmethod
    def _compute_cholesky(B: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
        """Lower-triangular Cholesky factor ``L`` with ``B = L L^T``.

        Falls back to a minimally jittered factorization if ``B`` is PSD but
        not strictly PD (e.g. a rank-deficient coregionalization matrix), so
        the factor stays available for any fitted/searched ``B``.
        """
        try:
            return np.linalg.cholesky(B)
        except np.linalg.LinAlgError:
            jitter = 1e-12 * max(1.0, float(np.abs(eigvals).max()))
            return np.linalg.cholesky(B + jitter * np.eye(B.shape[0]))

    @property
    def base(self) -> JaxCovarianceFunction:
        """The shared scalar base covariance function (read-only)."""
        return self._base

    @property
    def B(self) -> np.ndarray:
        """The coregionalization matrix ``B`` (read-only copy)."""
        return self._B.copy()

    @property
    def B_cholesky(self) -> np.ndarray:
        """Lower Cholesky factor ``L`` of ``B`` with ``B = L L^T`` (read-only copy)."""
        return self._B_cholesky.copy()

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        # Shape checking (mirrors IndependentMultiOutputCovarianceFunction).
        self._check_shapes(x0.shape, x1.shape if x1 is not None else None)

        base_vals = self._base(x0, x1)
        # base_vals has shape broadcast_batch_shape; broadcast against B (P, P)
        # to write *every* (c, c') entry: result[..., c, c'] = B[c, c'] * base.
        return base_vals[..., None, None] * self._B

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        # Shape checking (mirrors IndependentMultiOutputCovarianceFunction).
        self._check_shapes(x0.shape, x1.shape if x1 is not None else None)

        base_vals = self._base.jax(x0, x1)
        return base_vals[..., None, None] * jnp.asarray(self._B)

    def linop(
        self, x0: ArrayLike, x1: Optional[ArrayLike] = None
    ) -> pn.linops.LinearOperator:
        # B is constant in x, so the cross-covariance operator is the Kronecker
        # product B (x) K. With B = I_P this is exactly the block-diagonal
        # operator produced by IndependentMultiOutputCovarianceFunction with a
        # shared base kernel, which is the reduction-identity (V5) anchor.
        K = self._base.linop(x0, x1)
        return pn.linops.Kronecker(pn.linops.aslinop(self._B), K)
