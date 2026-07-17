from linpde_gp import functions

from ._directional_derivative import DirectionalDerivative
from ._helmholtz_operator import IdentityOperator
from ._laplacian import Laplacian, SpatialLaplacian, WeightedLaplacian
from ._partial_derivative import PartialDerivative


@DirectionalDerivative.__call__.register  # pylint: disable=no-member
@Laplacian.__call__.register  # pylint: disable=no-member
@PartialDerivative.__call__.register  # pylint: disable=no-member
@SpatialLaplacian.__call__.register  # pylint: disable=no-member
@WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, f: functions.Constant, /) -> functions.Zero:
    assert f.input_shape == self.input_domain_shape
    assert f.output_shape == self.input_codomain_shape

    return functions.Zero(
        input_shape=self.output_domain_shape,
        output_shape=self.output_codomain_shape,
    )


@IdentityOperator.__call__.register  # pylint: disable=no-member
def _(self, f: functions.Constant, /) -> functions.Constant:
    """Identity operator on constant function returns scaled constant."""
    assert f.input_shape == self.input_domain_shape
    assert f.output_shape == self.input_codomain_shape

    return functions.Constant(
        value=self.scalar * f.value,
        input_shape=self.output_domain_shape,
    )


# The HelmholtzOperator inherits behavior from SumLinearFunctionOperator
