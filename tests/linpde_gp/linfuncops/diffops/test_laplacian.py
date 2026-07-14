from pytest_cases import fixture

from linpde_gp.linfuncops.diffops import MultiIndex, WeightedLaplacian


@fixture
def laplacian() -> WeightedLaplacian:
    return WeightedLaplacian([1.0, 2.0, 3.0])


def test_coefficients(laplacian: WeightedLaplacian) -> WeightedLaplacian:
    assert len(laplacian.coefficients) == 1
    assert laplacian.coefficients[()] == {
        MultiIndex((2, 0, 0)): 1.0,
        MultiIndex((0, 2, 0)): 2.0,
        MultiIndex((0, 0, 2)): 3.0,
    }


def test_jax_fallback_respects_batch_shape(laplacian: WeightedLaplacian) -> None:
    import jax.numpy as jnp

    def f(x):
        return jnp.sum(x**2)

    laplacian_f = laplacian(f)

    x = jnp.ones((5,) + laplacian.input_domain_shape)

    result = laplacian_f.jax(x)

    assert result.shape == (5,)
    assert jnp.allclose(result, 12.0)
