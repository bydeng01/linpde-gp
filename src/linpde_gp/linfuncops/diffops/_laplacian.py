from __future__ import annotations

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from linpde_gp import functions

from ._coefficients import MultiIndex, PartialDerivativeCoefficients
from ._lindiffop import LinearDifferentialOperator

if TYPE_CHECKING:
    import linpde_gp


class WeightedLaplacian(LinearDifferentialOperator):
    r"""Generalization of the Laplacian operator, which multiplies each partial
    derivative with an individual weight.

    .. math::
        \Delta_w := \sum_{i = 1}^d w_i \frac{\partial^2}{\partial x_i^2}
    """

    def __init__(self, weights: ArrayLike) -> None:
        weights = np.asarray(weights)

        coefficients = PartialDerivativeCoefficients(
            {
                (): {
                    MultiIndex.from_index(domain_index, weights.shape, 2): coefficient
                    for domain_index, coefficient in np.ndenumerate(weights)
                    if coefficient != 0.0
                }
            },
            input_domain_shape=weights.shape,
            input_codomain_shape=(),
        )

        super().__init__(coefficients=coefficients, input_shapes=(weights.shape, ()))

        self._weights = weights

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(  # pylint: disable=arguments-differ
        self, f: Callable, /, *, argnum: int = 0, **kwargs
    ) -> Callable:
        f_hessian = jax.hessian(f, argnums=argnum)

        def _laplacian_unbatched(*args, **kwargs):
            hess = f_hessian(*args, **kwargs)
            hess = jnp.asarray(hess)

            if hess.ndim < 2:
                return jnp.asarray(self._weights) * hess

            # Hessian shape (unbatched): output_shape + input_shape + input_shape.
            # Take diagonal over the last two axes and sum with weights.
            hess_diag = jnp.diagonal(hess, axis1=-2, axis2=-1)

            weights = jnp.asarray(self._weights)
            if weights.ndim == 0:
                return weights * hess_diag

            return jnp.sum(
                weights * hess_diag,
                axis=tuple(range(-weights.ndim, 0)),
            )

        def f_laplacian(*args, **kwargs):
            args = list(args)

            target_arg = jnp.asarray(args[argnum])
            batch_ndim = target_arg.ndim - self.input_domain_ndim
            if batch_ndim < 0:
                raise ValueError("Input has fewer dimensions than the operator domain.")

            if batch_ndim == 0:
                args[argnum] = target_arg
                return _laplacian_unbatched(*args, **kwargs)

            args[argnum] = target_arg

            in_axes = [None] * len(args)
            in_axes[argnum] = 0

            vmapped_laplacian = _laplacian_unbatched
            for _ in range(batch_ndim):
                vmapped_laplacian = jax.vmap(vmapped_laplacian, in_axes=tuple(in_axes))

            return jax.jit(vmapped_laplacian)(*args, **kwargs)

        return f_laplacian

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()


class Laplacian(WeightedLaplacian):
    def __init__(self, domain_shape: ShapeLike) -> None:
        super().__init__(np.ones(domain_shape, dtype=np.double))

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()

    @weak_form.register(functions.bases.UnivariateLinearInterpolationBasis)
    def _weak_form_univariate_interpolation_basis(
        self, test_basis: functions.bases.UnivariateLinearInterpolationBasis
    ):
        from linpde_gp.linfunctls.weak_forms import (  # pylint: disable=import-outside-toplevel
            WeakForm_Laplacian_UnivariateInterpolationBasis,
        )

        return WeakForm_Laplacian_UnivariateInterpolationBasis(test_basis)


class SpatialLaplacian(WeightedLaplacian):
    def __init__(self, domain_shape: ShapeLike) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        if len(domain_shape) != 1 or domain_shape[0] < 2:
            raise ValueError()

        self._laplacian = Laplacian(domain_shape=(domain_shape[0] - 1,))

        weights = np.ones(domain_shape, dtype=np.double)
        weights[0] = 0

        super().__init__(weights)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()
