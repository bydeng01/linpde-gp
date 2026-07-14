from __future__ import annotations

from typing import Optional

from jax import numpy as jnp
import numpy as np

from ._jax import JaxCovarianceFunction


class Real2FromScalarKernel(JaxCovarianceFunction):
    """Wrap a scalar kernel into a 2-output kernel for (Re, Im)."""

    def __init__(self, scalar_kernel: JaxCovarianceFunction) -> None:
        if scalar_kernel.output_shape_0 != () or scalar_kernel.output_shape_1 != ():
            raise ValueError("Scalar kernel must be single-output.")

        self._scalar_kernel = scalar_kernel

        super().__init__(
            input_shape=scalar_kernel.input_shape,
            output_shape_0=(2,),
            output_shape_1=(2,),
        )

    @property
    def scalar_kernel(self) -> JaxCovarianceFunction:
        return self._scalar_kernel

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        cov = self.scalar_kernel(x0, x1)
        result = np.zeros(
            cov.shape + self.output_shape_0 + self.output_shape_1, dtype=cov.dtype
        )
        result[..., 0, 0] = cov / 2.0
        result[..., 1, 1] = cov / 2.0
        return result

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        cov = self.scalar_kernel.jax(x0, x1)
        result = jnp.zeros(
            cov.shape + self.output_shape_0 + self.output_shape_1, dtype=cov.dtype
        )
        result = result.at[..., 0, 0].set(cov / 2.0)
        result = result.at[..., 1, 1].set(cov / 2.0)
        return result
