from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

import linpde_gp  # pylint: disable=unused-import # for type hints
from linpde_gp import functions
from linpde_gp.functions import JaxFunction, JaxLambdaFunction

from .._arithmetic import SumLinearFunctionOperator
from .._linfuncop import LinearFunctionOperator
from .._select_output import SelectOutput
from ._coefficients import PartialDerivativeCoefficients

if TYPE_CHECKING:
    from typing import Union

    from .._arithmetic import CompositeLinearFunctionOperator
    from ._partial_derivative import PartialDerivative, _PartialDerivativeNoJax


class LinearDifferentialOperator(LinearFunctionOperator):
    """Linear differential operator that maps to functions with codomain R."""

    def __init__(
        self,
        coefficients: PartialDerivativeCoefficients,
        input_shapes: tuple[ShapeLike, ShapeLike],
    ) -> None:
        if coefficients.input_domain_shape != input_shapes[0]:
            raise ValueError()
        if coefficients.input_codomain_shape != input_shapes[1]:
            raise ValueError()

        super().__init__(
            input_shapes=input_shapes,
            output_shapes=(input_shapes[0], ()),
        )

        self._coefficients = coefficients

    @property
    def coefficients(self) -> PartialDerivativeCoefficients:
        return self._coefficients

    @property
    def has_mixed(self) -> bool:
        return self._coefficients.has_mixed

    # PartialDerivative or PartialDerivative @ SelectOutput
    PartialDerivativeSummand = "Union[PartialDerivative, CompositeLinearFunctionOperator]"  # pylint: disable=line-too-long

    def to_sum(
        self,
    ) -> (
        "SumLinearFunctionOperator[SumLinearFunctionOperator[PartialDerivativeSummand]]"
    ):
        from ._partial_derivative import (  # pylint: disable=import-outside-toplevel
            PartialDerivative,
        )

        outer_summands = []
        for output_index in self.coefficients:
            inner_summands = []
            for multi_index in self.coefficients[output_index]:
                partial_diffop = PartialDerivative(multi_index)
                if output_index != ():  # pylint: disable=comparison-with-callable
                    partial_diffop = partial_diffop @ SelectOutput(
                        self.input_shapes, output_index
                    )
                inner_summands.append(
                    self.coefficients[output_index][multi_index] * partial_diffop
                )
            outer_summands.append(SumLinearFunctionOperator(*inner_summands))
        return SumLinearFunctionOperator(*outer_summands)

    def _to_sum_no_jax(
        self,
    ) -> (
        "SumLinearFunctionOperator[SumLinearFunctionOperator[_PartialDerivativeNoJax]]"
    ):
        from ._partial_derivative import (  # pylint: disable=import-outside-toplevel
            _PartialDerivativeNoJax,
        )

        outer_summands = []
        for output_index in self.coefficients:
            inner_summands = []
            for multi_index in self.coefficients[output_index]:
                partial_diffop = _PartialDerivativeNoJax(multi_index)
                if output_index != ():  # pylint: disable=comparison-with-callable
                    partial_diffop = partial_diffop @ SelectOutput(
                        self.input_shapes, output_index
                    )
                inner_summands.append(
                    self.coefficients[output_index][multi_index] * partial_diffop
                )
            outer_summands.append(SumLinearFunctionOperator(*inner_summands))
        return SumLinearFunctionOperator(*outer_summands)

    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        try:
            return self._call_no_jax(f, **kwargs)
        except NotImplementedError:
            pass

        if isinstance(f, JaxFunction):
            if f.input_shape != self.input_domain_shape:
                raise ValueError()

            if f.output_shape != self.input_codomain_shape:
                raise ValueError()

            return JaxLambdaFunction(
                self._jax_fallback(f.jax, **kwargs),
                input_shape=self.output_domain_shape,
                output_shape=self.output_codomain_shape,
                vectorize=True,
            )

        return JaxLambdaFunction(
            self._jax_fallback(f, **kwargs),
            input_shape=self.output_domain_shape,
            output_shape=self.output_codomain_shape,
            vectorize=True,
        )

    def _call_no_jax(self, f, **kwargs):
        try:
            return super().__call__(f, **kwargs)
        except NotImplementedError:
            pass

        from ._partial_derivative import (  # pylint: disable=import-outside-toplevel
            PartialDerivative,
        )

        if isinstance(self, PartialDerivative):
            raise NotImplementedError()
        return self._to_sum_no_jax()(f, **kwargs)

    def _jax_fallback(  # pylint: disable=arguments-differ
        self, f: Callable, /, *, argnum: int = 0, **kwargs
    ) -> Callable:
        return self.to_sum()(f, argnum=argnum, **kwargs)

    def __rmul__(self, other) -> LinearFunctionOperator:
        if np.ndim(other) == 0:
            from ._arithmetic import (  # pylint: disable=import-outside-toplevel
                ScaledLinearDifferentialOperator,
            )

            return ScaledLinearDifferentialOperator(self, scalar=other)

        return NotImplemented

    @functools.singledispatchmethod
    def weak_form(
        self, basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()


class StackedLinearDifferentialOperator(LinearFunctionOperator):
    """Stack scalar linear differential operators into a vector-valued operator."""

    def __init__(self, *row_operators: LinearFunctionOperator) -> None:
        if not row_operators:
            raise ValueError("At least one row operator must be provided.")

        input_shapes = row_operators[0].input_shapes
        output_domain_shape = row_operators[0].output_domain_shape

        if any(row.input_shapes != input_shapes for row in row_operators):
            raise ValueError("All row operators must share the same input shapes.")

        if any(row.output_domain_shape != output_domain_shape for row in row_operators):
            raise ValueError("All row operators must share the same output domain.")

        if any(row.output_codomain_shape != () for row in row_operators):
            raise ValueError("Row operators must map to scalar codomains.")

        self._row_operators = tuple(row_operators)

        super().__init__(
            input_shapes=input_shapes,
            output_shapes=(output_domain_shape, (len(self._row_operators),)),
        )

        # Register handlers for types that would cause circular imports
        self._register_covariance_handler()
        self._register_crosscov_handler()

    @property
    def row_operators(self) -> tuple[LinearFunctionOperator, ...]:
        return self._row_operators

    def adjoint(self):
        # pylint: disable=import-outside-toplevel,redefined-outer-name
        from .._arithmetic import SumLinearFunctionOperator
        from .._select_output import SelectOutput

        selectors = [
            SelectOutput(self.output_shapes, idx)
            for idx in range(len(self._row_operators))
        ]
        return SumLinearFunctionOperator(
            *(
                row_op.adjoint() @ selector
                for row_op, selector in zip(self._row_operators, selectors)
            )
        )

    @functools.singledispatchmethod
    def __call__(self, obj, /, **kwargs):
        return super().__call__(obj, **kwargs)

    @__call__.register
    def _(self, f: pn.functions.Function, /):
        if f.input_shape != self.input_domain_shape:
            raise ValueError()

        if f.output_shape != self.input_codomain_shape:
            raise ValueError()

        components = tuple(row_op(f) for row_op in self.row_operators)
        return functions.stack(components, axis=-1)

    def _register_covariance_handler(self):
        """Deferred registration to avoid circular imports."""
        from linpde_gp.randprocs import (  # pylint: disable=import-outside-toplevel
            covfuncs as rp_covfuncs,
        )

        # pylint: disable=no-member
        @self.__call__.register(pn.randprocs.covfuncs.CovarianceFunction)
        def _(self, covfunc, *, argnum: int = 0):
            components = tuple(
                row_op(covfunc, argnum=argnum) for row_op in self.row_operators
            )
            return rp_covfuncs.StackCovarianceFunction(components, output_idx=argnum)

    def _register_crosscov_handler(self):
        """Deferred registration to avoid circular imports."""
        from linpde_gp.randprocs import (  # pylint: disable=import-outside-toplevel
            crosscov as rp_crosscov,
        )

        # pylint: disable=no-member
        @self.__call__.register(rp_crosscov.ProcessVectorCrossCovariance)
        def _(self, pv_crosscov):
            stacked = rp_crosscov.StackedProcessVectorCrossCovariance(
                tuple(row_op(pv_crosscov) for row_op in self.row_operators)
            )
            return stacked
