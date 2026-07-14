"""Unit tests for CoefficientFieldOperator (Phase 1)."""

from jax import numpy as jnp
import numpy as np
import pytest

import linpde_gp
from linpde_gp.functions import Constant, JaxLambdaFunction
from linpde_gp.linfuncops.diffops import (
    CoefficientFieldOperator,
    IdentityOperator,
)


# ---------------------------------------------------------------------------
# 1. Constant-field equivalence with IdentityOperator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain_shape, scalar",
    [
        ((), 1.0),
        ((), 3.5),
        ((1,), 2.0),
        ((2,), 2.5),
        ((3,), -1.25),
    ],
)
def test_constant_field_matches_identity_operator_on_polynomial(domain_shape, scalar):
    """A constant coefficient field must behave identically to IdentityOperator."""

    if domain_shape == ():
        def _poly(x):
            return x ** 2 + 1.0
    else:
        def _poly(x):
            return jnp.sum(x ** 2, axis=-1) + 1.0

    f = JaxLambdaFunction(
        _poly, input_shape=domain_shape, output_shape=(), vectorize=True
    )

    ident_op = IdentityOperator(domain_shape, scalar=scalar)
    coeff_op = CoefficientFieldOperator(
        domain_shape, Constant(domain_shape, value=scalar)
    )

    ident_out = ident_op(f)
    coeff_out = coeff_op(f)

    # Evaluate both on a small batch and compare
    if domain_shape == ():
        xs = np.linspace(-1.0, 1.0, 7)
    else:
        d = int(domain_shape[0])
        xs = np.linspace(-1.0, 1.0, 5 * d).reshape(5, d)

    np.testing.assert_allclose(
        np.asarray(coeff_out(xs)),
        np.asarray(ident_out(xs)),
        atol=1e-12,
    )


def test_constant_field_matches_identity_operator_on_exponential():
    """Constant-field equivalence on a JAX-traceable exponential function."""

    domain_shape = ()
    scalar = 0.75

    f = JaxLambdaFunction(
        lambda x: jnp.exp(-(x ** 2)), input_shape=(), output_shape=(), vectorize=True
    )

    ident_op = IdentityOperator(domain_shape, scalar=scalar)
    coeff_op = CoefficientFieldOperator(
        domain_shape, Constant(domain_shape, value=scalar)
    )

    xs = np.linspace(-2.0, 2.0, 11)
    np.testing.assert_allclose(
        np.asarray(coeff_op(f)(xs)),
        np.asarray(ident_op(f)(xs)),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 2. Variable-field correctness
# ---------------------------------------------------------------------------


def test_variable_field_correctness_1d():
    """c(x) = sin(x), f(x) = x^2 => (cf)(x) = sin(x) x^2."""

    c = JaxLambdaFunction(
        jnp.sin, input_shape=(), output_shape=(), vectorize=True
    )
    f = JaxLambdaFunction(
        lambda x: x ** 2, input_shape=(), output_shape=(), vectorize=True
    )

    op = CoefficientFieldOperator((), c)
    cf = op(f)

    xs = np.array([-1.0, -0.3, 0.0, 0.5, 1.7])
    expected = np.sin(xs) * xs ** 2
    np.testing.assert_allclose(np.asarray(cf(xs)), expected, atol=1e-12)


def test_variable_field_correctness_2d():
    """c(x,y) = x*y, f(x,y) = sin(x)+cos(y)."""

    c = JaxLambdaFunction(
        lambda x: x[..., 0] * x[..., 1],
        input_shape=(2,), output_shape=(), vectorize=True,
    )
    f = JaxLambdaFunction(
        lambda x: jnp.sin(x[..., 0]) + jnp.cos(x[..., 1]),
        input_shape=(2,), output_shape=(), vectorize=True,
    )

    op = CoefficientFieldOperator((2,), c)
    cf = op(f)

    pts = np.array([[0.0, 0.0], [1.0, 1.0], [-0.5, 0.3], [0.7, -1.2]])
    expected = pts[:, 0] * pts[:, 1] * (np.sin(pts[:, 0]) + np.cos(pts[:, 1]))
    np.testing.assert_allclose(np.asarray(cf(pts)), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Complex-valued coefficient field
# ---------------------------------------------------------------------------


def test_complex_coefficient_field_dtype_and_values():
    """c(x) = exp(i x), f(x) = 1 + x^2; check dtype is complex and values match."""

    c = JaxLambdaFunction(
        lambda x: jnp.exp(1j * x),
        input_shape=(), output_shape=(), vectorize=True,
    )
    f = JaxLambdaFunction(
        lambda x: 1.0 + x ** 2, input_shape=(), output_shape=(), vectorize=True
    )

    op = CoefficientFieldOperator((), c)
    assert np.issubdtype(op.dtype, np.complexfloating)

    cf = op(f)
    xs = np.linspace(-2.0, 2.0, 9)
    expected = np.exp(1j * xs) * (1.0 + xs ** 2)
    np.testing.assert_allclose(
        np.asarray(cf(xs)), expected, atol=1e-10
    )


# ---------------------------------------------------------------------------
# 4. Constant input function: result should evaluate to c(x) * v(x)
# ---------------------------------------------------------------------------


def test_constant_input_returns_scaled_field():
    """Applying op to a non-zero Constant returns a function equal to v * c(x)."""

    c = JaxLambdaFunction(
        jnp.cos, input_shape=(), output_shape=(), vectorize=True
    )
    v = 2.5
    f = Constant((), value=v)

    op = CoefficientFieldOperator((), c)
    out = op(f)

    xs = np.linspace(-1.0, 1.0, 5)
    np.testing.assert_allclose(np.asarray(out(xs)), v * np.cos(xs), atol=1e-12)


def test_zero_input_returns_zero():
    """Applying op to a Zero function returns Zero."""
    from linpde_gp.functions import Zero

    c = JaxLambdaFunction(
        jnp.sin, input_shape=(), output_shape=(), vectorize=True
    )
    f = Zero(input_shape=(), output_shape=())

    op = CoefficientFieldOperator((), c)
    out = op(f)

    assert isinstance(out, Zero)


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------


def test_shape_mismatch_in_output_shape_raises():
    """Passing a coefficient_field with non-scalar output_shape must raise."""

    vector_field = JaxLambdaFunction(
        lambda x: jnp.array([x, x ** 2]),
        input_shape=(),
        output_shape=(2,),
        vectorize=True,
    )

    with pytest.raises(ValueError, match="scalar-valued"):
        CoefficientFieldOperator((), vector_field)


def test_input_shape_mismatch_raises():
    """coefficient_field.input_shape must match domain_shape."""
    c1d = JaxLambdaFunction(
        jnp.sin, input_shape=(), output_shape=(), vectorize=True
    )

    with pytest.raises(ValueError, match="input_shape"):
        CoefficientFieldOperator((2,), c1d)


def test_non_function_argument_raises():
    """Passing something that is not a probnum.functions.Function must raise."""

    with pytest.raises(TypeError, match="Function"):
        CoefficientFieldOperator((), 1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. Public-API export
# ---------------------------------------------------------------------------


def test_class_is_exported_from_diffops():
    """Symbol must be available from the diffops public module."""
    assert (
        linpde_gp.linfuncops.diffops.CoefficientFieldOperator
        is CoefficientFieldOperator
    )
