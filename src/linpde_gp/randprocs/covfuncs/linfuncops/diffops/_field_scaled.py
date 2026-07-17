"""Structured kernel for ``c(x_{argnum}) * k(x_0, x_1)``.

The naïve approach of wrapping the product in a ``JaxLambdaCovarianceFunction``
fails for downstream operators that need to differentiate ``k`` because, e.g.,
the Matern kernel has a square-root non-smoothness at the diagonal that the
specialized handlers in this package work around. Generic JAX tracing through
that path produces ``NaN`` on the diagonal.

This module provides
:class:`FieldScaledCovarianceFunction`, a thin wrapper that keeps the
``c``-field and the base kernel as separate components, plus registry
entries that route subsequent diffop applications along the *opposite*
``argnum`` back to the base kernel's specialized handlers.
"""

from __future__ import annotations

from typing import Optional

from jax import numpy as jnp
import numpy as np
import probnum as pn

from linpde_gp import functions as _functions

from ..._jax import JaxCovarianceFunction


class FieldScaledCovarianceFunction(JaxCovarianceFunction):
    r"""Represents :math:`c(x_{\mathrm{argnum}})\, k(x_0, x_1)`.

    Parameters
    ----------
    coefficient_field :
        Scalar ``pn.functions.Function`` over the kernel's spatial domain.
    base_kernel :
        The kernel that is being scaled.
    scaled_argnum :
        Which input axis (0 or 1) of the kernel ``c`` depends on.
    """

    def __init__(
        self,
        coefficient_field: pn.functions.Function,
        base_kernel: pn.randprocs.covfuncs.CovarianceFunction,
        scaled_argnum: int,
    ) -> None:
        if scaled_argnum not in (0, 1):
            raise ValueError(f"scaled_argnum must be 0 or 1, got {scaled_argnum}.")
        if coefficient_field.input_shape != base_kernel.input_shape:
            raise ValueError(
                "coefficient_field.input_shape must match base_kernel.input_shape."
            )
        if coefficient_field.output_shape != ():
            raise ValueError(
                "coefficient_field must be scalar-valued, got "
                f"output_shape={coefficient_field.output_shape}."
            )

        self._coefficient_field = coefficient_field
        self._base_kernel = base_kernel
        self._scaled_argnum = int(scaled_argnum)

        super().__init__(
            input_shape=base_kernel.input_shape,
            output_shape_0=base_kernel.output_shape_0,
            output_shape_1=base_kernel.output_shape_1,
        )

    @property
    def coefficient_field(self) -> pn.functions.Function:
        return self._coefficient_field

    @property
    def base_kernel(self) -> pn.randprocs.covfuncs.CovarianceFunction:
        return self._base_kernel

    @property
    def scaled_argnum(self) -> int:
        return self._scaled_argnum

    def _c_jax(self, x):
        c = self._coefficient_field
        if isinstance(c, _functions.JaxFunction):
            return c.jax(x)
        return jnp.asarray(np.asarray(c(np.asarray(x))))

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            x1 = x0
        if self._scaled_argnum == 0:
            cvals = np.asarray(self._coefficient_field(x0))
        else:
            cvals = np.asarray(self._coefficient_field(x1))
        return cvals * self._base_kernel(x0, x1)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            x1 = x0
        if self._scaled_argnum == 0:
            return self._c_jax(x0) * self._base_kernel.jax(x0, x1)
        return self._c_jax(x1) * self._base_kernel.jax(x0, x1)
