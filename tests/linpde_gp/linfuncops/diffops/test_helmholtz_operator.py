from jax import numpy as jnp
import numpy as np

import pytest

import linpde_gp
from linpde_gp.functions import Constant, JaxLambdaFunction
from linpde_gp.linfuncops.diffops import HelmholtzOperator


def test_helmholtz_real2_operator_matches_manual_split():
    domain_shape = (1,)
    k_squared = 2.0 + 0.5j
    operator = linpde_gp.linfuncops.diffops.HelmholtzReal2Operator(
        domain_shape=domain_shape,
        k_squared=k_squared,
    )

    def _f(x):
        x_scalar = jnp.atleast_1d(x)[..., 0]
        return jnp.stack(
            (
                jnp.sin(x_scalar),
                jnp.cos(x_scalar),
            ),
            axis=-1,
        )

    f = linpde_gp.functions.JaxLambdaFunction(
        _f,
        input_shape=domain_shape,
        output_shape=(2,),
        vectorize=True,
    )

    result = operator(f)

    xs = np.linspace(-1.0, 1.0, 5)[:, None]

    alpha = np.real(k_squared)
    beta = np.imag(k_squared)

    f_r = np.sin(xs[:, 0])
    f_i = np.cos(xs[:, 0])
    lap_f_r = -f_r
    lap_f_i = -f_i

    expected_real = lap_f_r + alpha * f_r - beta * f_i
    expected_imag = lap_f_i + alpha * f_i + beta * f_r

    evaluated = result(xs)

    np.testing.assert_allclose(evaluated[..., 0], expected_real)
    np.testing.assert_allclose(evaluated[..., 1], expected_imag)


# ---------------------------------------------------------------------------
# Phase 2: HelmholtzOperator.from_coefficient_field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain_shape, k0",
    [
        ((), 1.0),
        ((), 4.7),
        ((1,), 2.5),
        ((2,), 2.5),
        ((3,), 0.8),
    ],
)
def test_from_coefficient_field_matches_scalar_for_constant_field(domain_shape, k0):
    """Constant k²-field must produce results identical to scalar k²."""

    if domain_shape == ():
        f = JaxLambdaFunction(
            lambda x: jnp.sin(x) + 0.5 * x**2,
            input_shape=(),
            output_shape=(),
            vectorize=True,
        )
    else:
        d = int(domain_shape[0])
        if d == 1:
            f = JaxLambdaFunction(
                lambda x: jnp.sin(x[..., 0]),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )
        elif d == 2:
            f = JaxLambdaFunction(
                lambda x: jnp.sin(x[..., 0]) * jnp.cos(x[..., 1]),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )
        else:
            f = JaxLambdaFunction(
                lambda x: jnp.sum(x**2, axis=-1),
                input_shape=domain_shape,
                output_shape=(),
                vectorize=True,
            )

    scalar_op = HelmholtzOperator(domain_shape, k_squared=k0)
    variable_op = HelmholtzOperator.from_coefficient_field(
        domain_shape, Constant(domain_shape, value=k0)
    )

    if domain_shape == ():
        xs = np.linspace(-1.0, 1.0, 11)
    else:
        d = int(domain_shape[0])
        xs = np.linspace(-1.0, 1.0, 7 * d).reshape(7, d)

    np.testing.assert_allclose(
        np.asarray(variable_op(f)(xs)),
        np.asarray(scalar_op(f)(xs)),
        atol=1e-12,
    )


def test_from_coefficient_field_manufactured_solution_real():
    r"""1D, k²(x) = (x+2)², f(x) = (x+2)^{-2}.

    Then
        f'(x)  = -2(x+2)^{-3}
        f''(x) =  6(x+2)^{-4}
        k² f   =  (x+2)² * (x+2)^{-2} = 1
        (Δ + k² I) f = 6(x+2)^{-4} + 1.
    """

    k_squared = JaxLambdaFunction(
        lambda x: (x + 2.0) ** 2, input_shape=(), output_shape=(), vectorize=True
    )
    f = JaxLambdaFunction(
        lambda x: (x + 2.0) ** (-2.0),
        input_shape=(),
        output_shape=(),
        vectorize=True,
    )

    op = HelmholtzOperator.from_coefficient_field((), k_squared)
    result = op(f)

    xs = np.linspace(-0.9, 0.9, 9)
    expected = 6.0 * (xs + 2.0) ** (-4.0) + 1.0
    got = np.asarray(result(xs))
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_from_coefficient_field_manufactured_solution_complex():
    r"""Complex variable field, real f.

    k²(x) = (1 + 0.1j)(x+2)², f(x) = (x+2)^{-2}. Then
        Δf  = 6(x+2)^{-4}                     (real)
        k²f = (1 + 0.1j)                      (complex, constant)
        (Δ + k² I) f = 6(x+2)^{-4} + (1 + 0.1j).

    The Laplacian-of-complex-f path is intentionally avoided here because
    JAX's `jacrev` requires real-valued outputs; the same restriction exists
    in the scalar-k² baseline. Phase 3's real2 lift handles complex f via
    the {Re, Im} stack and is what production code uses for fully complex
    BVPs.
    """

    k_squared = JaxLambdaFunction(
        lambda x: (1.0 + 0.1j) * ((x + 2.0) ** 2),
        input_shape=(),
        output_shape=(),
        vectorize=True,
    )
    f = JaxLambdaFunction(
        lambda x: (x + 2.0) ** (-2.0),
        input_shape=(),
        output_shape=(),
        vectorize=True,
    )

    op = HelmholtzOperator.from_coefficient_field((), k_squared)
    assert op.is_variable_coefficient
    result = op(f)

    xs = np.linspace(-0.9, 0.9, 9)
    expected = 6.0 * (xs + 2.0) ** (-4.0) + (1.0 + 0.1j)
    got = np.asarray(result(xs))
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_wave_number_raises_for_variable_field():
    k_squared = JaxLambdaFunction(
        lambda x: (x + 2.0) ** 2, input_shape=(), output_shape=(), vectorize=True
    )
    op = HelmholtzOperator.from_coefficient_field((), k_squared)

    with pytest.raises(ValueError, match="variable"):
        _ = op.wave_number


def test_wave_number_still_works_for_scalar():
    op = HelmholtzOperator(domain_shape=(), k_squared=4.0)
    assert not op.is_variable_coefficient
    np.testing.assert_allclose(op.wave_number, 2.0)


def test_from_coefficient_field_rejects_nonfunction():
    with pytest.raises(TypeError, match="Function"):
        HelmholtzOperator.from_coefficient_field((), 1.0)  # type: ignore[arg-type]


def test_repr_distinguishes_variable_and_scalar():
    scalar_repr = repr(HelmholtzOperator((), k_squared=1.5))
    assert "1.5" in scalar_repr and "variable" not in scalar_repr

    k_field = JaxLambdaFunction(
        lambda x: (x + 1.0) ** 2, input_shape=(), output_shape=(), vectorize=True
    )
    variable_repr = repr(HelmholtzOperator.from_coefficient_field((), k_field))
    assert "variable" in variable_repr


# ---------------------------------------------------------------------------
# Phase 3: HelmholtzReal2Operator.from_coefficient_field
# ---------------------------------------------------------------------------


def test_real2_from_coefficient_field_matches_scalar_for_constant_field():
    """A constant complex k² field must produce identical results to the
    existing scalar real2 constructor."""

    domain_shape = (1,)
    k_squared = 2.0 + 0.5j

    def _f(x):
        x_scalar = jnp.atleast_1d(x)[..., 0]
        return jnp.stack(
            (
                jnp.sin(x_scalar),
                jnp.cos(x_scalar),
            ),
            axis=-1,
        )

    f = linpde_gp.functions.JaxLambdaFunction(
        _f, input_shape=domain_shape, output_shape=(2,), vectorize=True
    )

    scalar_op = linpde_gp.linfuncops.diffops.HelmholtzReal2Operator(
        domain_shape=domain_shape, k_squared=k_squared
    )

    k_field = linpde_gp.functions.Constant(domain_shape, value=k_squared)
    variable_op = (
        linpde_gp.linfuncops.diffops.HelmholtzReal2Operator.from_coefficient_field(
            domain_shape=domain_shape, k_squared_field=k_field
        )
    )

    xs = np.linspace(-1.0, 1.0, 5)[:, None]
    expected = np.asarray(scalar_op(f)(xs))
    got = np.asarray(variable_op(f)(xs))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_real2_from_coefficient_field_variable_matches_numpy_reference():
    """Variable α(x), β(x) — compare to a hand-built numpy reference."""

    domain_shape = (1,)

    # k²(x) = (2 + x²) + i (0.5 + 0.1 x)
    k_field = linpde_gp.functions.JaxLambdaFunction(
        lambda x: (2.0 + jnp.atleast_1d(x)[..., 0] ** 2)
        + 1j * (0.5 + 0.1 * jnp.atleast_1d(x)[..., 0]),
        input_shape=domain_shape,
        output_shape=(),
        vectorize=True,
    )

    # f(x) = [sin(x), cos(x)] — stacked Re/Im
    f = linpde_gp.functions.JaxLambdaFunction(
        lambda x: jnp.stack(
            (jnp.sin(jnp.atleast_1d(x)[..., 0]), jnp.cos(jnp.atleast_1d(x)[..., 0])),
            axis=-1,
        ),
        input_shape=domain_shape,
        output_shape=(2,),
        vectorize=True,
    )

    op = linpde_gp.linfuncops.diffops.HelmholtzReal2Operator.from_coefficient_field(
        domain_shape=domain_shape, k_squared_field=k_field
    )
    assert op.is_variable_coefficient

    xs = np.linspace(-1.0, 1.0, 7)[:, None]
    xv = xs[:, 0]

    alpha = 2.0 + xv**2
    beta = 0.5 + 0.1 * xv

    f_r = np.sin(xv)
    f_i = np.cos(xv)
    lap_r = -np.sin(xv)
    lap_i = -np.cos(xv)

    expected_real = lap_r + alpha * f_r - beta * f_i
    expected_imag = lap_i + alpha * f_i + beta * f_r

    got = np.asarray(op(f)(xs))
    np.testing.assert_allclose(got[..., 0], expected_real, atol=1e-10)
    np.testing.assert_allclose(got[..., 1], expected_imag, atol=1e-10)


def test_real2_from_coefficient_field_rejects_nonfunction():
    with pytest.raises(TypeError, match="Function"):
        linpde_gp.linfuncops.diffops.HelmholtzReal2Operator.from_coefficient_field(
            (1,), 1.0 + 0.0j  # type: ignore[arg-type]
        )


def test_real2_adjoint_for_variable_field():
    """Adjoint must conjugate the field pointwise."""

    domain_shape = (1,)
    k_field = linpde_gp.functions.JaxLambdaFunction(
        lambda x: (1.0 + jnp.atleast_1d(x)[..., 0] ** 2)
        + 1j * (0.2 + 0.3 * jnp.atleast_1d(x)[..., 0]),
        input_shape=domain_shape,
        output_shape=(),
        vectorize=True,
    )

    op = linpde_gp.linfuncops.diffops.HelmholtzReal2Operator.from_coefficient_field(
        domain_shape=domain_shape, k_squared_field=k_field
    )
    adj = op.adjoint()

    xs = np.linspace(-0.5, 0.5, 4)[:, None]
    # The adjoint's k_squared field should equal conjugate of the original.
    orig_vals = k_field(xs)
    adj_vals = adj.k_squared(xs)
    np.testing.assert_allclose(adj_vals, np.conjugate(orig_vals), atol=1e-12)
