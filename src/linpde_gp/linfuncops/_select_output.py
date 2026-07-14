import functools

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

from jax import numpy as jnp
from linpde_gp import functions
from ._linfuncop import LinearFunctionOperator


class SelectOutput(LinearFunctionOperator):
    def __init__(
        self,
        input_shapes: tuple[ShapeLike, ShapeLike],
        idx: tuple[int, ...] | int,
    ) -> None:
        self._idx = idx

        input_domain_shape = pn.utils.as_shape(input_shapes[0])
        input_codomain_shape = pn.utils.as_shape(input_shapes[1])

        # Shape of the selected output after applying `idx`
        output_codomain_shape = np.empty(input_codomain_shape, dtype=int)[self._idx].shape

        super().__init__(
            (input_domain_shape, input_codomain_shape),
            output_shapes=(input_domain_shape, output_codomain_shape),
        )

    @property
    def idx(self) -> tuple[int, ...] | int:
        return self._idx

    def adjoint(self):
        """Adjoint inserts the selected component back into the full output."""

        class _AdjointSelectOutput(LinearFunctionOperator):
            def __init__(inner_self, base: SelectOutput):
                inner_self._base = base
                super().__init__(
                    input_shapes=(
                        base.output_domain_shape,
                        base.output_codomain_shape,
                    ),
                    output_shapes=(
                        base.output_domain_shape,
                        base.input_codomain_shape,
                    ),
                )

            @functools.singledispatchmethod
            def __call__(inner_self, obj, /, **kwargs):
                return super().__call__(obj, **kwargs)

            @__call__.register
            def _(inner_self, f: functions.JaxFunction, /):
                if f.input_shape != inner_self.input_domain_shape:
                    raise ValueError()
                if f.output_shape != inner_self.input_codomain_shape:
                    raise ValueError()

                class InsertOutputFunction(functions.JaxFunction):
                    def __init__(self):
                        super().__init__(
                            input_shape=inner_self.output_domain_shape,
                            output_shape=inner_self.output_codomain_shape,
                        )
                        self._f = f
                        self._idx = inner_self._base.idx

                    def _evaluate(self, x):
                        base_vals = self._f(x)
                        batch_ndim = base_vals.ndim - len(
                            inner_self.input_codomain_shape
                        )
                        batch_shape = base_vals.shape[:batch_ndim]
                        result = np.zeros(
                            batch_shape + inner_self.output_codomain_shape,
                            dtype=base_vals.dtype,
                        )
                        result[(Ellipsis, self._idx)] = base_vals
                        return result

                    def _evaluate_jax(self, x):
                        base_vals = self._f.jax(x)
                        batch_ndim = base_vals.ndim - len(
                            inner_self.input_codomain_shape
                        )
                        batch_shape = base_vals.shape[:batch_ndim]
                        result = jnp.zeros(
                            batch_shape + inner_self.output_codomain_shape,
                            dtype=base_vals.dtype,
                        )
                        result = result.at[(Ellipsis, self._idx)].set(base_vals)
                        return result

                return InsertOutputFunction()

            @__call__.register
            def _(
                inner_self,
                k: pn.randprocs.covfuncs.CovarianceFunction,
                /,
                *,
                argnum: int = 0,
            ):
                from linpde_gp.randprocs import (
                    covfuncs as rp_covfuncs,
                )  # pylint: disable=import-outside-toplevel

                output_shape_0 = (
                    inner_self.output_codomain_shape
                    if argnum == 0
                    else k.output_shape_0
                )
                output_shape_1 = (
                    inner_self.output_codomain_shape
                    if argnum == 1
                    else k.output_shape_1
                )

                input_shape = k.input_shape_0 if argnum == 0 else k.input_shape_1

                class InsertedCovarianceFunction(rp_covfuncs.JaxCovarianceFunction):
                    def __init__(self):
                        super().__init__(
                            input_shape=input_shape,
                            output_shape_0=output_shape_0,
                            output_shape_1=output_shape_1,
                        )
                        self._base_k = k
                        self._idx = inner_self._base.idx
                        self._argnum = argnum

                    def _evaluate(self, x0, x1):
                        base_eval = self._base_k(x0, x1)
                        batch_ndim = (
                            base_eval.ndim
                            - len(self._base_k.output_shape_0)
                            - len(self._base_k.output_shape_1)
                        )
                        batch_shape = base_eval.shape[:batch_ndim]
                        result = np.zeros(
                            batch_shape + output_shape_0 + output_shape_1,
                            dtype=base_eval.dtype,
                        )

                        if self._argnum == 0:
                            slicer = (
                                Ellipsis,
                                self._idx,
                                *([slice(None)] * len(self._base_k.output_shape_1)),
                            )
                        else:
                            slicer = (
                                Ellipsis,
                                *([slice(None)] * len(self._base_k.output_shape_0)),
                                self._idx,
                            )

                        result[slicer] = base_eval
                        return result

                    def _evaluate_jax(self, x0, x1):
                        base_eval = self._base_k.jax(x0, x1)
                        batch_ndim = (
                            base_eval.ndim
                            - len(self._base_k.output_shape_0)
                            - len(self._base_k.output_shape_1)
                        )
                        batch_shape = base_eval.shape[:batch_ndim]
                        result = jnp.zeros(
                            batch_shape + output_shape_0 + output_shape_1,
                            dtype=base_eval.dtype,
                        )

                        if self._argnum == 0:
                            slicer = (
                                Ellipsis,
                                self._idx,
                                *([slice(None)] * len(self._base_k.output_shape_1)),
                            )
                        else:
                            slicer = (
                                Ellipsis,
                                *([slice(None)] * len(self._base_k.output_shape_0)),
                                self._idx,
                            )

                        result = result.at[slicer].set(base_eval)
                        return result

                return InsertedCovarianceFunction()

        return _AdjointSelectOutput(self)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: functions.JaxFunction, /):
        if f.input_shape != self.input_domain_shape:
            raise ValueError(
                f"Function input shape {f.input_shape} does not match "
                f"operator domain shape {self.input_domain_shape}"
            )
        if f.output_shape != self.input_codomain_shape:
            raise ValueError(
                f"Function output shape {f.output_shape} does not match "
                f"operator input codomain shape {self.input_codomain_shape}"
            )

        class SelectOutputFunction(functions.JaxFunction):
            def __init__(inner_self):
                super().__init__(
                    input_shape=self.output_domain_shape,
                    output_shape=self.output_codomain_shape,
                )
                inner_self._f = f
                inner_self._idx = self.idx

            def _evaluate(inner_self, x):
                result = inner_self._f(x)
                return result[..., inner_self._idx]

            def _evaluate_jax(inner_self, x):
                result = inner_self._f.jax(x)
                return result[..., inner_self._idx]

        return SelectOutputFunction()

    @__call__.register
    def _(self, k: pn.randprocs.covfuncs.CovarianceFunction, /, *, argnum: int = 0):
        """Select an output component of a covariance function."""
        from linpde_gp.randprocs import covfuncs as rp_covfuncs  # pylint: disable=import-outside-toplevel

        # If the selected side is scalar-valued already, selection is a no-op.
        if (argnum == 0 and len(k.output_shape_0) == 0) or (
            argnum == 1 and len(k.output_shape_1) == 0
        ):
            return k

        # Determine new output shapes depending on which argument we act on
        output_shape_0 = (
            self.output_codomain_shape if argnum == 0 else k.output_shape_0
        )
        output_shape_1 = (
            self.output_codomain_shape if argnum == 1 else k.output_shape_1
        )

        input_shape = k.input_shape_0 if argnum == 0 else k.input_shape_1

        # Create a proper covariance function class that handles x1=None correctly
        class SelectedCovarianceFunction(rp_covfuncs.JaxCovarianceFunction):
            def __init__(inner_self):
                super().__init__(
                    input_shape=input_shape,
                    output_shape_0=output_shape_0,
                    output_shape_1=output_shape_1,
                )
                inner_self._base_k = k
                inner_self._idx = self.idx
                inner_self._argnum = argnum

            def _evaluate(inner_self, x0, x1):
                k_vals = inner_self._base_k(x0, x1)

                # For x1 is None, probnum interprets this as a pointwise covariance
                # k(x, x), but the return array must still have shape
                #   batch_shape + output_shape_0 + output_shape_1
                # of this *selected* kernel. We therefore do not collapse both
                # output dimensions to a scalar, but only select along the
                # requested argument.
                if inner_self._argnum == 0:
                    slicer = (
                        Ellipsis,
                        inner_self._idx,
                        *([slice(None)] * len(k.output_shape_1)),
                    )
                else:
                    slicer = (
                        Ellipsis,
                        *([slice(None)] * len(k.output_shape_0)),
                        inner_self._idx,
                    )

                return k_vals[slicer]

            def _evaluate_jax(inner_self, x0, x1):
                k_vals = inner_self._base_k.jax(x0, x1)

                if inner_self._argnum == 0:
                    slicer = (
                        Ellipsis,
                        inner_self._idx,
                        *([slice(None)] * len(k.output_shape_1)),
                    )
                else:
                    slicer = (
                        Ellipsis,
                        *([slice(None)] * len(k.output_shape_0)),
                        inner_self._idx,
                    )

                return k_vals[slicer]

        return SelectedCovarianceFunction()

    def __repr__(self) -> str:
        return f"SelectOutput(idx={self.idx})"
