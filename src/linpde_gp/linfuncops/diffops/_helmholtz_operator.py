from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Union

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ShapeLike

from linpde_gp import functions as _functions  # for JaxLambdaFunction wrapping

from .._arithmetic import SumLinearFunctionOperator
from .._select_output import SelectOutput
from ._coefficient_field_operator import CoefficientFieldOperator
from ._coefficients import MultiIndex, PartialDerivativeCoefficients
from ._laplacian import Laplacian
from ._lindiffop import LinearDifferentialOperator, StackedLinearDifferentialOperator

if TYPE_CHECKING:
    import linpde_gp


class IdentityOperator(LinearDifferentialOperator):
    r"""Identity operator (zero-order differential operator).

    This operator represents multiplication by a scalar, which corresponds
    to a differential operator with no derivatives. It is defined as:

    .. math::
        \mathcal{I}[f] = \alpha f

    where :math:`\alpha` is a scalar coefficient.

    Parameters
    ----------
    domain_shape : ShapeLike
        Shape of the spatial domain.
    scalar : Union[float, complex], optional
        The scalar multiplier. Default is 1.0.
    """

    def __init__(
        self, domain_shape: ShapeLike, scalar: Union[float, complex] = 1.0
    ) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        # Determine appropriate dtype based on scalar type
        dtype = np.complex128 if np.iscomplex(scalar) else np.double

        # Zero-order derivative (no differentiation) with given coefficient
        zero_multi_index = MultiIndex(np.zeros(domain_shape, dtype=int))

        coefficients = PartialDerivativeCoefficients(
            {(): {zero_multi_index: np.asarray(scalar, dtype=dtype)}},
            input_domain_shape=domain_shape,
            input_codomain_shape=(),
        )

        super().__init__(coefficients=coefficients, input_shapes=(domain_shape, ()))

        self._scalar = np.asarray(scalar, dtype=dtype)
        self._domain_shape = domain_shape

    @property
    def scalar(self) -> Union[float, complex, np.number]:
        """The scalar multiplier."""
        return self._scalar

    @property
    def domain_shape(self) -> tuple:
        """Shape of the domain."""
        return self._domain_shape

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()

    def __repr__(self) -> str:
        return (
            f"IdentityOperator(domain_shape={self._domain_shape}, "
            f"scalar={self._scalar})"
        )


class HelmholtzOperator(SumLinearFunctionOperator):
    r"""Helmholtz operator: ∇² + k²I.

    The Helmholtz equation arises in many physical contexts including
    acoustics, electromagnetics, and quantum mechanics. The operator
    is defined as:

    .. math::
        \mathcal{H} = \nabla^2 + k^2 I

    where :math:`k` is the wave number and :math:`I` is the identity operator.

    The Helmholtz equation :math:`(\nabla^2 + k^2)u = f` appears in:

    * Acoustics: sound wave propagation
    * Electromagnetics: time-harmonic Maxwell's equations
    * Quantum mechanics: time-independent Schrödinger equation
    * Seismology: elastic wave propagation

    Parameters
    ----------
    domain_shape : ShapeLike
        Shape of the spatial domain.
    k_squared : ScalarLike
        The wave number squared (k²). Can be real or complex.
        Complex values model lossy media with attenuation.

    Examples
    --------
    >>> # Create a 2D Helmholtz operator with k² = 2.5
    >>> helmholtz = HelmholtzOperator(domain_shape=(2,), k_squared=2.5)

    >>> # Complex wave number for lossy media
    >>> helmholtz_complex = HelmholtzOperator(domain_shape=(3,), k_squared=1+0.5j)

    >>> # Using the factory method with wave number k
    >>> helmholtz = HelmholtzOperator.from_wave_number(domain_shape=(2,), k=1.58)
    """

    def __init__(self, domain_shape: ShapeLike, k_squared: ScalarLike) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        # Preserve the type of k_squared (real or complex)
        self._k_squared = k_squared
        self._domain_shape = domain_shape

        # Create the two components: Laplacian and k²*Identity
        laplacian = Laplacian(domain_shape)
        k_squared_identity = IdentityOperator(domain_shape, scalar=self._k_squared)

        # Helmholtz operator is the sum: ∇² + k²I
        super().__init__(laplacian, k_squared_identity)

    @classmethod
    def from_wave_number(
        cls, domain_shape: ShapeLike, k: ScalarLike
    ) -> "HelmholtzOperator":
        """Create a Helmholtz operator from the wave number k.

        Parameters
        ----------
        domain_shape : ShapeLike
            Shape of the spatial domain.
        k : ScalarLike
            The wave number (not squared). Can be real or complex.

        Returns
        -------
        HelmholtzOperator
            A Helmholtz operator with k_squared = k².

        Examples
        --------
        >>> # Create operator with wave number k = 2π/λ
        >>> wavelength = 0.5
        >>> k = 2 * np.pi / wavelength
        >>> helmholtz = HelmholtzOperator.from_wave_number((2,), k)
        """
        return cls(domain_shape, k_squared=k**2)

    @classmethod
    def from_coefficient_field(
        cls,
        domain_shape: ShapeLike,
        k_squared_field: pn.functions.Function,
    ) -> "HelmholtzOperator":
        r"""Build a Helmholtz operator with a spatially-varying coefficient.

        Constructs the operator

        .. math::
            \mathcal{H}[f](x) = \Delta f(x) + k^2(x)\, f(x),

        where :math:`k^2(\cdot)` is given by ``k_squared_field``. The instance
        is created via ``cls.__new__`` so that the existing scalar
        ``__init__`` signature is preserved verbatim; the
        :class:`SumLinearFunctionOperator` machinery is then initialized
        directly on the sum of the Laplacian and a
        :class:`CoefficientFieldOperator`.

        Parameters
        ----------
        domain_shape :
            Shape of the spatial domain (same convention as the scalar
            constructor).
        k_squared_field :
            A ``pn.functions.Function`` representing the variable
            :math:`k^2(x)`. Must have ``input_shape == domain_shape`` and
            ``output_shape == ()``.

        Returns
        -------
        HelmholtzOperator
            An operator whose ``k_squared`` attribute is the
            ``k_squared_field`` function itself, not a scalar.
        """
        domain_shape = pn.utils.as_shape(domain_shape)

        if not isinstance(k_squared_field, pn.functions.Function):
            raise TypeError(
                "`k_squared_field` must be a probnum.functions.Function, "
                f"got {type(k_squared_field).__name__}. Use the scalar "
                "constructor `HelmholtzOperator(domain_shape, k_squared=...)` "
                "for constant k²."
            )

        instance = cls.__new__(cls)
        # Mirror the scalar `__init__` invariants but store the field instead.
        instance._k_squared = k_squared_field
        instance._domain_shape = domain_shape

        laplacian = Laplacian(domain_shape)
        coeff_op = CoefficientFieldOperator(
            domain_shape, coefficient_field=k_squared_field
        )

        # Initialize the SumLinearFunctionOperator state directly; we skip the
        # scalar HelmholtzOperator.__init__ to preserve its signature for
        # backward compatibility.
        SumLinearFunctionOperator.__init__(instance, laplacian, coeff_op)
        return instance

    @property
    def k_squared(self):
        """The wave number squared (k²).

        Returns either a scalar (when the operator was built with a constant
        coefficient) or a ``pn.functions.Function`` (when built via
        :meth:`from_coefficient_field`).
        """
        return self._k_squared

    @property
    def is_variable_coefficient(self) -> bool:
        """True if the operator was built with a function-valued k²."""
        return isinstance(self._k_squared, pn.functions.Function)

    @property
    def domain_shape(self) -> tuple:
        """Shape of the spatial domain."""
        return self._domain_shape

    @property
    def wave_number(self) -> Union[float, complex, np.number]:
        """The wave number k (computed as sqrt(k²)).

        Only well-defined for the scalar (constant-coefficient) case. For a
        variable coefficient field, evaluate ``k_squared`` at a point first
        and take the square root, e.g. ``np.sqrt(op.k_squared(x))``.
        """
        if self.is_variable_coefficient:
            raise ValueError(
                "`wave_number` is only defined for scalar k². This operator "
                "has a variable k²(x) field; evaluate "
                "`np.sqrt(op.k_squared(x))` at the point of interest."
            )
        return np.sqrt(self._k_squared)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        r"""Compute the weak form of the Helmholtz operator.

        The weak form is obtained by integration by parts:

        .. math::
            \langle \mathcal{H}u, v \rangle = -\langle \nabla u, \nabla v \rangle
            + k^2 \langle u, v \rangle

        for suitable test functions v with appropriate boundary conditions.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        if self.is_variable_coefficient:
            return (
                f"HelmholtzOperator(domain_shape={self._domain_shape}, "
                f"k_squared=<variable: {type(self._k_squared).__name__}>)"
            )
        return (
            f"HelmholtzOperator(domain_shape={self._domain_shape}, "
            f"k_squared={self._k_squared})"
        )


class HelmholtzReal2Operator(StackedLinearDifferentialOperator):
    """Real-valued representation of the complex Helmholtz operator."""

    def __init__(self, domain_shape: ShapeLike, k_squared: ScalarLike) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)
        k_squared = complex(k_squared)

        self._domain_shape = domain_shape
        self._k_squared = k_squared

        alpha = np.real(k_squared)
        beta = float(np.imag(k_squared))

        laplacian = Laplacian(domain_shape)
        identity = IdentityOperator(domain_shape, scalar=1.0)

        lap_plus_alpha = SumLinearFunctionOperator(
            laplacian,
            IdentityOperator(domain_shape, scalar=alpha),
        )

        selector_real = SelectOutput((domain_shape, (2,)), 0)
        selector_imag = SelectOutput((domain_shape, (2,)), 1)

        diag_real = lap_plus_alpha @ selector_real
        diag_imag = lap_plus_alpha @ selector_imag

        row_real_terms = [diag_real]
        row_imag_terms = [diag_imag]

        if beta != 0.0:
            row_real_terms.append((-beta) * identity @ selector_imag)
            row_imag_terms.append(beta * identity @ selector_real)

        row_real = (
            SumLinearFunctionOperator(*row_real_terms)
            if len(row_real_terms) > 1
            else row_real_terms[0]
        )
        row_imag = (
            SumLinearFunctionOperator(*row_imag_terms)
            if len(row_imag_terms) > 1
            else row_imag_terms[0]
        )

        super().__init__(row_real, row_imag)

    @classmethod
    def from_coefficient_field(
        cls,
        domain_shape: ShapeLike,
        k_squared_field: pn.functions.Function,
    ) -> "HelmholtzReal2Operator":
        r"""Build a real2 Helmholtz operator with a spatially-varying k²(x).

        Constructs the block operator

        .. math::
            \begin{bmatrix} \Delta + \alpha(x) & -\beta(x) \\
                              \beta(x)         & \Delta + \alpha(x)
            \end{bmatrix},

        where :math:`\alpha(x) = \mathrm{Re}\, k^2(x)` and
        :math:`\beta(x)  = \mathrm{Im}\, k^2(x)`. Internally the two diagonal
        :math:`\Delta + \alpha(x)` blocks share a single
        :class:`CoefficientFieldOperator` instance, and the off-diagonal
        couplings each use a second one (one for :math:`-\beta`, one for
        :math:`\beta`).

        Parameters
        ----------
        domain_shape :
            Shape of the spatial domain.
        k_squared_field :
            ``pn.functions.Function`` representing :math:`k^2(x)`. The field
            may be complex-valued; its real and imaginary parts are extracted
            internally into separate real-valued JAX functions so that all
            downstream arithmetic stays real.
        """
        domain_shape = pn.utils.as_shape(domain_shape)

        if not isinstance(k_squared_field, pn.functions.Function):
            raise TypeError(
                "`k_squared_field` must be a probnum.functions.Function, "
                f"got {type(k_squared_field).__name__}."
            )

        if k_squared_field.input_shape != domain_shape:
            raise ValueError(
                f"k_squared_field.input_shape ({k_squared_field.input_shape}) "
                f"must match domain_shape ({domain_shape})."
            )
        if k_squared_field.output_shape != ():
            raise ValueError(
                "k_squared_field must be scalar-valued, got "
                f"output_shape={k_squared_field.output_shape}."
            )

        # Split into real-valued α(x) and β(x). Wrapping via JaxLambdaFunction
        # gives us scalar real-output Functions that compose cleanly with
        # CoefficientFieldOperator.
        if isinstance(k_squared_field, _functions.JaxFunction):
            alpha_field = _functions.JaxLambdaFunction(
                lambda x, _f=k_squared_field: jnp.real(_f.jax(x)),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )
            beta_field = _functions.JaxLambdaFunction(
                lambda x, _f=k_squared_field: jnp.imag(_f.jax(x)),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )
            neg_beta_field = _functions.JaxLambdaFunction(
                lambda x, _f=k_squared_field: -jnp.imag(_f.jax(x)),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )
        else:
            alpha_field = _functions.JaxLambdaFunction(
                lambda x, _f=k_squared_field: jnp.real(jnp.asarray(_f(np.asarray(x)))),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )
            beta_field = _functions.JaxLambdaFunction(
                lambda x, _f=k_squared_field: jnp.imag(jnp.asarray(_f(np.asarray(x)))),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )
            neg_beta_field = _functions.JaxLambdaFunction(
                lambda x, _f=k_squared_field: -jnp.imag(jnp.asarray(_f(np.asarray(x)))),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )

        laplacian = Laplacian(domain_shape)
        alpha_op = CoefficientFieldOperator(domain_shape, alpha_field)
        beta_op = CoefficientFieldOperator(domain_shape, beta_field)
        neg_beta_op = CoefficientFieldOperator(domain_shape, neg_beta_field)

        lap_plus_alpha = SumLinearFunctionOperator(laplacian, alpha_op)

        selector_real = SelectOutput((domain_shape, (2,)), 0)
        selector_imag = SelectOutput((domain_shape, (2,)), 1)

        diag_real = lap_plus_alpha @ selector_real
        diag_imag = lap_plus_alpha @ selector_imag

        row_real = SumLinearFunctionOperator(diag_real, neg_beta_op @ selector_imag)
        row_imag = SumLinearFunctionOperator(diag_imag, beta_op @ selector_real)

        # Construct without invoking the scalar __init__: mirror the spec's
        # pattern from Phase 2.
        instance = cls.__new__(cls)
        instance._domain_shape = domain_shape
        instance._k_squared = k_squared_field
        # pylint: disable=attribute-defined-outside-init
        instance._alpha_field = alpha_field
        instance._beta_field = beta_field
        StackedLinearDifferentialOperator.__init__(instance, row_real, row_imag)
        return instance

    @property
    def k_squared(self):
        """The wave number squared.

        Scalar (``complex``) when built via the scalar constructor, or a
        ``pn.functions.Function`` when built via
        :meth:`from_coefficient_field`.
        """
        return self._k_squared

    @property
    def is_variable_coefficient(self) -> bool:
        return isinstance(self._k_squared, pn.functions.Function)

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return self._domain_shape

    def adjoint(self) -> "HelmholtzReal2Operator":
        """Return the adjoint operator.

        In the real2 representation the off-diagonal couplings encode the
        imaginary part of k^2 with opposite signs in the transpose. Flipping
        the sign of ``beta`` therefore yields the conjugate-transpose operator.
        For a variable field, we conjugate the field pointwise.
        """
        if self.is_variable_coefficient:
            k_field = self._k_squared

            if isinstance(k_field, _functions.JaxFunction):
                conj_field = _functions.JaxLambdaFunction(
                    lambda x, _f=k_field: jnp.conjugate(_f.jax(x)),
                    input_shape=self._domain_shape,
                    output_shape=(),
                    vectorize=True,
                )
            else:
                conj_field = _functions.JaxLambdaFunction(
                    lambda x, _f=k_field: jnp.conjugate(jnp.asarray(_f(np.asarray(x)))),
                    input_shape=self._domain_shape,
                    output_shape=(),
                    vectorize=True,
                )

            return HelmholtzReal2Operator.from_coefficient_field(
                domain_shape=self._domain_shape,
                k_squared_field=conj_field,
            )

        return HelmholtzReal2Operator(
            domain_shape=self._domain_shape,
            k_squared=np.conjugate(self._k_squared),
        )
