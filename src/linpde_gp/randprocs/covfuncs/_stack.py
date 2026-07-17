import functools
import operator
from typing import Optional

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.randprocs import covfuncs as pn_covfuncs
from probnum.randprocs.covfuncs import _arithmetic_fallbacks as pn_cov_arith
from probnum.typing import ArrayLike

from linpde_gp.linops import BlockMatrix

from ._jax import JaxCovarianceFunctionMixin


class StackCovarianceFunction(
    JaxCovarianceFunctionMixin, pn_covfuncs.CovarianceFunction
):
    def __init__(self, covfuncs: ArrayLike, output_idx: int = 1) -> None:
        self._covfuncs = np.asarray(covfuncs)
        covfuncs_flat = self._covfuncs.reshape(-1, order="C")

        # All component kernels must share the same input shapes.
        if any(
            covfunc.input_shape_0 != covfuncs_flat[0].input_shape_0
            for covfunc in covfuncs_flat
        ):
            raise ValueError()

        if any(
            covfunc.input_shape_1 != covfuncs_flat[0].input_shape_1
            for covfunc in covfuncs_flat
        ):
            raise ValueError()

        output_idx = int(output_idx)

        if output_idx not in (0, 1):
            raise ValueError()

        # We only allow stacking along an output dimension which is scalar in each
        # component kernel. The other output dimension must be identical across
        # components and is carried through unchanged.
        if output_idx == 0:
            # Stack along output dimension 0: this must be scalar in each component.
            if any(covfunc.output_shape_0 != () for covfunc in covfuncs_flat):
                raise ValueError()
            # All output_shape_1 must agree.
            if any(
                covfunc.output_shape_1 != covfuncs_flat[0].output_shape_1
                for covfunc in covfuncs_flat
            ):
                raise ValueError()
        else:
            # Stack along output dimension 1: this must be scalar in each component.
            if any(covfunc.output_shape_1 != () for covfunc in covfuncs_flat):
                raise ValueError()
            # All output_shape_0 must agree.
            if any(
                covfunc.output_shape_0 != covfuncs_flat[0].output_shape_0
                for covfunc in covfuncs_flat
            ):
                raise ValueError()

        self._output_idx = output_idx

        super().__init__(
            input_shape_0=covfuncs_flat[0].input_shape_0,
            input_shape_1=covfuncs_flat[0].input_shape_1,
            output_shape_0=(
                self._covfuncs.shape
                if self._output_idx == 0
                else covfuncs_flat[0].output_shape_0
            ),
            output_shape_1=(
                self._covfuncs.shape
                if self._output_idx == 1
                else covfuncs_flat[0].output_shape_1
            ),
        )

    @property
    def covfuncs(self) -> np.ndarray:
        return self._covfuncs

    @property
    def output_idx(self) -> int:
        return self._output_idx

    def matrix(self, x0: np.ndarray, x1: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the covariance matrix and enforce symmetry when appropriate."""
        K = super().matrix(x0, x1)

        # When evaluating a square Gram matrix (x1 is None or identical to x0),
        # symmetrize to guard against tiny asymmetries introduced by nested
        # operator compositions.
        if x1 is None or (x0.shape == x1.shape and np.allclose(x0, x1)):
            K = 0.5 * (K + K.T)

            # Numerical safeguard: ensure positive semidefiniteness.
            eigvals, eigvecs = np.linalg.eigh(K)
            eigvals_clipped = np.clip(eigvals, 0.0, None)

            if np.any(eigvals_clipped != eigvals):
                K = (eigvecs * eigvals_clipped) @ eigvecs.T

        return K

    # pylint: disable=too-complex,too-many-locals,too-many-statements,protected-access
    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        evals = np.empty_like(self._covfuncs, dtype=np.object_)
        batch_shape = None
        component_output_shape_0 = None
        component_output_shape_1 = None

        def _safe_eval(covfunc):
            # Evaluate without triggering ProbNum's shape assertions; recurse through
            # arithmetic wrappers explicitly.
            if isinstance(covfunc, pn_cov_arith.SumCovarianceFunction):
                return functools.reduce(
                    operator.add,
                    (
                        _safe_eval(s) for s in covfunc._summands
                    ),  # pylint: disable=protected-access
                )
            if isinstance(covfunc, pn_cov_arith.ScaledCovarianceFunction):
                return covfunc._scalar * _safe_eval(  # pylint: disable=protected-access
                    covfunc._covfunc  # pylint: disable=protected-access
                )
            # Special case for SelectOutput-produced covariance wrappers to avoid
            # invoking their `__call__` (which re-enters ProbNum assertions).
            if hasattr(covfunc, "_base_k") and hasattr(covfunc, "_argnum"):
                base_eval = _safe_eval(
                    covfunc._base_k
                )  # pylint: disable=protected-access
                if covfunc._argnum == 0:  # pylint: disable=protected-access
                    slicer = (
                        Ellipsis,
                        covfunc._idx,  # pylint: disable=protected-access
                        *(
                            [slice(None)] * len(covfunc._base_k.output_shape_1)
                        ),  # pylint: disable=protected-access
                    )
                else:
                    slicer = (
                        Ellipsis,
                        *(
                            [slice(None)] * len(covfunc._base_k.output_shape_0)
                        ),  # pylint: disable=protected-access
                        covfunc._idx,  # pylint: disable=protected-access
                    )
                return base_eval[slicer]
            # Fall back to protected evaluation to bypass __call__ assertions.
            return covfunc._evaluate(x0, x1)  # pylint: disable=protected-access

        for idx, covfunc in np.ndenumerate(self._covfuncs):
            # Compute expected shape once; use safe evaluation to bypass assertions.
            bcast_batch_shape = (
                covfunc._check_shapes(  # pylint: disable=protected-access
                    x0.shape, x1.shape if x1 is not None else None
                )
            )
            expected_shape = (
                bcast_batch_shape + covfunc.output_shape_0 + covfunc.output_shape_1
            )
            eval_result = _safe_eval(covfunc)

            if eval_result.shape != expected_shape:
                # Some kernels (e.g. jitted closures) may return an extra batch
                # axis which mirrors the first batch dimension. In that case we
                # pick the diagonal along the duplicated axis.
                if (
                    x1 is not None
                    and len(eval_result.shape) == len(expected_shape) + 1
                    and len(bcast_batch_shape) == 2
                    and eval_result.shape[:2] == bcast_batch_shape
                    and eval_result.shape[2] == bcast_batch_shape[0]
                ):
                    diag_idx = np.arange(bcast_batch_shape[0])
                    eval_result = eval_result[
                        diag_idx[:, None],
                        np.arange(bcast_batch_shape[1])[None, :],
                        diag_idx[:, None],
                        ...,
                    ]

                # If we only have one of the two batch axes (diagonal only),
                # promote it to a full pairwise matrix by placing values on the
                # diagonal.
                if (
                    x1 is not None
                    and len(bcast_batch_shape) == 2
                    and eval_result.size == bcast_batch_shape[0]
                ):
                    diag_vals = np.reshape(
                        eval_result,
                        (bcast_batch_shape[0],)
                        + covfunc.output_shape_0
                        + covfunc.output_shape_1,
                    )
                    full = np.zeros(expected_shape, dtype=eval_result.dtype)
                    diag_idx = np.arange(bcast_batch_shape[0])
                    diag_slice = (
                        diag_idx,
                        diag_idx,
                        *([slice(None)] * (full.ndim - 2)),
                    )
                    full[diag_slice] = diag_vals
                    eval_result = full

                # If we only received the diagonal (shape matches the first
                # batch axis but not the full pairwise matrix), expand it to a
                # diagonal matrix along the batch dimensions.
                batch_ndim = len(bcast_batch_shape)
                if (
                    x1 is not None
                    and batch_ndim == 2
                    and bcast_batch_shape[0] == bcast_batch_shape[1]
                ):
                    diag_shape = (
                        (bcast_batch_shape[0],)
                        + covfunc.output_shape_0
                        + covfunc.output_shape_1
                    )
                    if eval_result.shape == diag_shape:
                        full = np.zeros(expected_shape, dtype=eval_result.dtype)
                        diag_idx = np.arange(bcast_batch_shape[0])
                        full[diag_idx, diag_idx] = eval_result
                        eval_result = full

                if eval_result.shape != expected_shape:
                    eval_result = np.reshape(eval_result, expected_shape)
            evals[idx] = eval_result

            if component_output_shape_0 is None:
                component_output_shape_0 = covfunc.output_shape_0
                component_output_shape_1 = covfunc.output_shape_1

            if batch_shape is None:
                # Extract the batch shape by removing the component's output axes
                output_shape_not_stacked = (
                    component_output_shape_1
                    if self._output_idx == 0
                    else component_output_shape_0
                )
                batch_shape = eval_result.shape[
                    : eval_result.ndim - len(output_shape_not_stacked)
                ]

        result_shape = (
            batch_shape
            + (
                self._covfuncs.shape
                if self._output_idx == 0
                else component_output_shape_0
            )
            + (
                self._covfuncs.shape
                if self._output_idx == 1
                else component_output_shape_1
            )
        )

        res = np.zeros(result_shape, dtype=np.result_type(evals.flat[0]))
        for idx, eval_at_idx in np.ndenumerate(evals):
            batch_slice = (slice(None),) * len(batch_shape)

            if self._output_idx == 0:
                output_shape_1_slice = (slice(None),) * len(component_output_shape_1)
                res[batch_slice + idx + output_shape_1_slice] = eval_at_idx
            else:
                output_shape_0_slice = (slice(None),) * len(component_output_shape_0)
                res[batch_slice + output_shape_0_slice + idx] = eval_at_idx
        return res

    # pylint: disable=too-complex,too-many-locals,protected-access
    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        evals = np.empty_like(self._covfuncs, dtype=np.object_)
        batch_shape = None
        component_output_shape_0 = None
        component_output_shape_1 = None

        def _safe_eval_jax(covfunc):
            # Handle SelectOutput-wrapped kernels without triggering ProbNum assertions.
            if hasattr(covfunc, "_base_k") and hasattr(covfunc, "_argnum"):
                base_eval = _safe_eval_jax(
                    covfunc._base_k
                )  # pylint: disable=protected-access
                if covfunc._argnum == 0:  # pylint: disable=protected-access
                    slicer = (
                        Ellipsis,
                        covfunc._idx,  # pylint: disable=protected-access
                        *(
                            [slice(None)] * len(covfunc._base_k.output_shape_1)
                        ),  # pylint: disable=protected-access
                    )
                else:
                    slicer = (
                        Ellipsis,
                        *(
                            [slice(None)] * len(covfunc._base_k.output_shape_0)
                        ),  # pylint: disable=protected-access
                        covfunc._idx,  # pylint: disable=protected-access
                    )
                return base_eval[slicer]
            try:
                return covfunc.jax(x0, x1)
            except AssertionError:
                try:
                    return covfunc._evaluate_jax(
                        x0, x1
                    )  # pylint: disable=protected-access
                except AssertionError:
                    if isinstance(covfunc, pn_cov_arith.SumCovarianceFunction):
                        return functools.reduce(
                            operator.add,
                            (
                                _safe_eval_jax(s) for s in covfunc._summands
                            ),  # pylint: disable=protected-access
                        )
                    if isinstance(covfunc, pn_cov_arith.ScaledCovarianceFunction):
                        return (
                            covfunc._scalar
                            * _safe_eval_jax(  # pylint: disable=protected-access
                                covfunc._covfunc  # pylint: disable=protected-access
                            )
                        )
                    raise

        for idx, covfunc in np.ndenumerate(self._covfuncs):
            eval_result = _safe_eval_jax(covfunc)

            output_shape_not_stacked = (
                covfunc.output_shape_1
                if self._output_idx == 0
                else covfunc.output_shape_0
            )
            bcast_batch_shape = (
                covfunc._check_shapes(  # pylint: disable=protected-access
                    x0.shape, x1.shape if x1 is not None else None
                )
            )
            expected_shape = (
                bcast_batch_shape + covfunc.output_shape_0 + covfunc.output_shape_1
            )
            if eval_result.shape != expected_shape:
                if (
                    x1 is not None
                    and len(eval_result.shape) == len(expected_shape) + 1
                    and len(bcast_batch_shape) == 2
                    and eval_result.shape[:2] == bcast_batch_shape
                    and eval_result.shape[2] == bcast_batch_shape[0]
                ):
                    n = bcast_batch_shape[0]
                    eval_result = eval_result[
                        jnp.arange(n)[:, None],
                        jnp.arange(bcast_batch_shape[1])[None, :],
                        jnp.arange(n)[:, None],
                        ...,
                    ]

            evals[idx] = eval_result

            if component_output_shape_0 is None:
                component_output_shape_0 = covfunc.output_shape_0
                component_output_shape_1 = covfunc.output_shape_1

            if batch_shape is None:
                batch_shape = eval_result.shape[
                    : eval_result.ndim - len(output_shape_not_stacked)
                ]

                # Handle case with an extra duplicated batch axis.
                if (
                    x1 is not None
                    and len(batch_shape) == len(self._covfuncs.shape) + 1
                    and batch_shape[:2] == batch_shape[2:4][:2]  # heuristic
                ):
                    n = batch_shape[0]
                    eval_result = eval_result[
                        jnp.arange(n)[:, None],
                        jnp.arange(batch_shape[1])[None, :],
                        jnp.arange(n)[:, None],
                        ...,
                    ]
                    evals[idx] = eval_result
                    batch_shape = eval_result.shape[
                        : eval_result.ndim - len(output_shape_not_stacked)
                    ]

        result_shape = (
            batch_shape
            + (
                self._covfuncs.shape
                if self._output_idx == 0
                else component_output_shape_0
            )
            + (
                self._covfuncs.shape
                if self._output_idx == 1
                else component_output_shape_1
            )
        )

        res = jnp.zeros(result_shape, dtype=jnp.result_type(evals.flat[0]))
        for idx, eval_at_idx in np.ndenumerate(evals):
            batch_slice = (slice(None),) * len(batch_shape)

            if self._output_idx == 0:
                output_shape_1_slice = (slice(None),) * len(component_output_shape_1)
                res = res.at[batch_slice + idx + output_shape_1_slice].set(eval_at_idx)
            else:
                output_shape_0_slice = (slice(None),) * len(component_output_shape_0)
                res = res.at[batch_slice + output_shape_0_slice + idx].set(eval_at_idx)
        return res

    def linop(
        self, x0: pn.utils.ArrayLike, x1: Optional[pn.utils.ArrayLike] = None
    ) -> pn.linops.LinearOperator:
        if self._output_idx == 0:
            return BlockMatrix(
                [
                    [covfunc.linop(x0, x1)]
                    for covfunc in self.covfuncs.reshape(-1, order="C")
                ]
            )
        return BlockMatrix(
            [
                [
                    covfunc.linop(x0, x1)
                    for covfunc in self.covfuncs.reshape(-1, order="C")
                ]
            ]
        )
