"""Pointwise multiplication by a (possibly complex-valued) coefficient field.

This module provides :class:`CoefficientFieldOperator`, the variable-coefficient
analogue of :class:`IdentityOperator`. It represents the zero-order linear
differential operator

.. math::
    \\mathcal{C}[f](x) = c(x) \\cdot f(x),

where ``c`` is an arbitrary ``pn.functions.Function`` of the same input domain
as ``f``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

from linpde_gp import functions

from ._coefficients import MultiIndex, PartialDerivativeCoefficients
from ._lindiffop import LinearDifferentialOperator

if TYPE_CHECKING:
    import linpde_gp


class CoefficientFieldOperator(LinearDifferentialOperator):
    r"""Zero-order operator: pointwise multiplication by ``c(x)``.

    This is the variable-coefficient generalization of
    :class:`~linpde_gp.linfuncops.diffops.IdentityOperator`. Whereas
    ``IdentityOperator(domain_shape, scalar)`` represents
    :math:`f \mapsto \alpha\, f` for a scalar :math:`\alpha`, this operator
    represents

    .. math::
        f \mapsto c(\cdot)\, f(\cdot)

    where :math:`c` is an arbitrary scalar-valued function of the spatial
    coordinate.

    The coefficient field may be real- or complex-valued.

    Parameters
    ----------
    domain_shape :
        Shape of the spatial domain (matches the input shape of the functions
        this operator acts on).
    coefficient_field :
        A ``pn.functions.Function`` whose ``input_shape`` equals
        ``domain_shape`` and whose ``output_shape`` is the empty tuple
        ``()`` (scalar-valued).

    Notes
    -----
    The underlying :class:`PartialDerivativeCoefficients` table stores a unit
    coefficient on the zero-multi-index so that the parent-class invariants
    are satisfied. The actual ``__call__`` semantics are overridden to use the
    function-valued ``coefficient_field`` directly; falling back to the scalar
    coefficient table would be incorrect, and ``to_sum`` / ``_jax_fallback``
    are not invoked through the override.
    """

    def __init__(
        self,
        domain_shape: ShapeLike,
        coefficient_field: pn.functions.Function,
    ) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        if not isinstance(coefficient_field, pn.functions.Function):
            raise TypeError(
                "`coefficient_field` must be a probnum.functions.Function, "
                f"got {type(coefficient_field).__name__}."
            )

        if coefficient_field.input_shape != domain_shape:
            raise ValueError(
                f"coefficient_field.input_shape ({coefficient_field.input_shape}) "
                f"must match domain_shape ({domain_shape})."
            )

        if coefficient_field.output_shape != ():
            raise ValueError(
                "coefficient_field must be scalar-valued, i.e. have "
                f"output_shape=(), got output_shape={coefficient_field.output_shape}."
            )

        # Infer dtype from the coefficient field. We probe a sample point to
        # see whether the field is complex. Probing is cheap and avoids
        # special-casing every Function subclass.
        sample_input = np.zeros(domain_shape, dtype=np.double)
        try:
            sample_value = np.asarray(coefficient_field(sample_input))
            is_complex = np.iscomplexobj(sample_value)
        except Exception:  # pylint: disable=broad-except
            # If the probe fails (e.g. domain excludes the origin), default
            # to real. Tests requiring complex dtype will override this branch
            # via Constant / explicit dtype-bearing functions.
            is_complex = False

        dtype = np.complex128 if is_complex else np.double

        # Build a minimal-but-valid coefficient table so that base class
        # invariants hold. We never dispatch through this table because
        # __call__ is overridden below.
        zero_multi_index = MultiIndex(np.zeros(domain_shape, dtype=int))
        coefficients = PartialDerivativeCoefficients(
            {(): {zero_multi_index: np.asarray(1.0, dtype=dtype)}},
            input_domain_shape=domain_shape,
            input_codomain_shape=(),
        )

        super().__init__(
            coefficients=coefficients,
            input_shapes=(domain_shape, ()),
        )

        self._coefficient_field = coefficient_field
        self._domain_shape = domain_shape
        self._dtype = dtype

    @property
    def coefficient_field(self) -> pn.functions.Function:
        """The variable coefficient field ``c(x)``."""
        return self._coefficient_field

    @property
    def domain_shape(self) -> tuple:
        """Shape of the spatial domain."""
        return self._domain_shape

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype the operator output is expected to take."""
        return np.dtype(self._dtype)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):  # type: ignore[override]
        # Function inputs always use the variable-coefficient product
        # semantic; we must NOT fall through to the LinearDifferentialOperator
        # base class because the placeholder unit-coefficient table there
        # would silently produce the wrong result (an identity map).
        if isinstance(f, pn.functions.Function):
            return self._call_on_function(f)

        # For non-Function inputs (random processes, GPs, ...), defer to the
        # parent dispatch so that the existing operator-on-stochastic-objects
        # machinery has a chance to handle the call.
        return super().__call__(f, **kwargs)

    def _call_on_function(self, f: pn.functions.Function) -> pn.functions.Function:
        """Return the function ``x -> c(x) * f(x)``."""
        if f.input_shape != self.input_domain_shape:
            raise ValueError(
                f"f.input_shape ({f.input_shape}) does not match operator "
                f"input_domain_shape ({self.input_domain_shape})."
            )
        if f.output_shape != self.input_codomain_shape:
            raise ValueError(
                f"f.output_shape ({f.output_shape}) does not match operator "
                f"input_codomain_shape ({self.input_codomain_shape})."
            )

        c_field = self._coefficient_field

        # Specialization 1: f is Zero -> result is Zero.
        if isinstance(f, functions.Zero):
            return functions.Zero(
                input_shape=self.output_domain_shape,
                output_shape=self.output_codomain_shape,
            )

        # Specialization 2: f is a JaxFunction and c is also a JaxFunction.
        # Compose them as a JaxLambdaFunction so JAX tracing is preserved.
        if isinstance(f, functions.JaxFunction) and isinstance(
            c_field, functions.JaxFunction
        ):

            def _product(x):
                return c_field.jax(x) * f.jax(x)

            return functions.JaxLambdaFunction(
                _product,
                input_shape=self.output_domain_shape,
                output_shape=self.output_codomain_shape,
                vectorize=True,
            )

        # General fallback: wrap as a plain JaxLambdaFunction that calls back
        # into both factors via their numpy interface.
        def _np_product(x):
            return jnp.asarray(np.asarray(c_field(np.asarray(x)))) * jnp.asarray(
                np.asarray(f(np.asarray(x)))
            )

        return functions.JaxLambdaFunction(
            _np_product,
            input_shape=self.output_domain_shape,
            output_shape=self.output_codomain_shape,
            vectorize=True,
        )

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":  # noqa: F821
        raise NotImplementedError()

    def __repr__(self) -> str:
        return (
            f"CoefficientFieldOperator(domain_shape={self._domain_shape}, "
            f"coefficient_field={self._coefficient_field!r})"
        )


# When applied to a Constant function the product c(x) * v is just a rescaled
# coefficient field. We return a JaxLambdaFunction so that callers can still
# evaluate it pointwise.
@CoefficientFieldOperator.__call__.register  # pylint: disable=no-member
def _(self, f: functions.Constant, /) -> pn.functions.Function:
    assert f.input_shape == self.input_domain_shape
    assert f.output_shape == self.input_codomain_shape

    if np.all(np.asarray(f.value) == 0):
        return functions.Zero(
            input_shape=self.output_domain_shape,
            output_shape=self.output_codomain_shape,
        )

    return self._call_on_function(f)  # pylint: disable=protected-access


# NOTE: the CovarianceFunction-input handler for CoefficientFieldOperator is
# registered in
# `linpde_gp.randprocs.covfuncs.linfuncops.diffops._registry` to avoid a
# circular import between this module and `linpde_gp.randprocs.covfuncs`.
