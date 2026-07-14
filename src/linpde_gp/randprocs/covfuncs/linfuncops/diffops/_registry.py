from jax import numpy as jnp
import numpy as np
from probnum.randprocs import covfuncs as pn_covfuncs

from linpde_gp import functions as _functions
from linpde_gp.linfuncops import diffops
from linpde_gp.randprocs import covfuncs

from . import _expquad, _matern, _tensor_product
from ._field_scaled import FieldScaledCovarianceFunction
from ..._utils import validate_covfunc_transformation

########################################################################################
# General `LinearFunctionOperators` ####################################################
########################################################################################


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxCovarianceFunctionMixin, /, *, argnum=0):
    validate_covfunc_transformation(self, k, argnum)

    try:
        return self._call_no_jax(k, argnum=argnum)  # pylint: disable=protected-access
    except NotImplementedError:
        return covfuncs.JaxLambdaCovarianceFunction(
            self._jax_fallback(  # pylint: disable=protected-access
                k.jax, argnum=argnum
            ),
            input_shape=self.output_domain_shape,
            output_shape_0=k.output_shape_0,
            output_shape_1=k.output_shape_1,
            vectorize=True,
        )


########################################################################################
# Structured kernel composition: scalar-codomain LDOs distribute over multi-output #####
# kernel structure (Stack / JaxSum / JaxScaled / Zero) so that downstream specialized ##
# handlers (e.g. the Matern WeightedLaplacian closed-forms) remain reachable for the ###
# variable-coefficient Helmholtz pipeline against 2-component priors. ##################
########################################################################################


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    """Distribute a scalar-codomain LDO over a :class:`StackCovarianceFunction`.

    Each component of the stack is itself a scalar covariance, so applying the
    operator independently to each component preserves the stack. This holds
    regardless of whether ``argnum`` matches ``k.output_idx``: because the
    operator's codomain is scalar, it does not introduce a new output axis,
    so the stacked axis is unchanged.

    Vector-codomain LDOs (e.g. :class:`StackedLinearDifferentialOperator`) have
    their own dispatch path and are not handled here; we defer to the generic
    fallback by raising :class:`NotImplementedError`.
    """
    if self.output_codomain_shape != ():
        raise NotImplementedError(
            "Scalar-codomain LDO required to distribute over StackCovarianceFunction; "
            f"got output_codomain_shape={self.output_codomain_shape}."
        )
    L_covfuncs = np.copy(k.covfuncs)
    for idx, covfunc in np.ndenumerate(L_covfuncs):
        L_covfuncs[idx] = self(covfunc, argnum=argnum)
    return covfuncs.StackCovarianceFunction(L_covfuncs, output_idx=k.output_idx)


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxSumCovarianceFunction, /, *, argnum: int = 0):
    """Distribute an LDO over the summands of a :class:`JaxSumCovarianceFunction`."""
    return covfuncs.JaxSumCovarianceFunction(
        *(self(summand, argnum=argnum) for summand in k.summands)
    )


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxScaledCovarianceFunction, /, *, argnum: int = 0):
    """Pull the scalar out of a :class:`JaxScaledCovarianceFunction`."""
    return k.scalar * self(k.covfunc, argnum=argnum)


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.Zero, /, *, argnum: int = 0):
    """Apply an LDO to :class:`Zero`: return another Zero with the operator's codomain."""
    return covfuncs.Zero(
        input_shape_0=self.output_domain_shape if argnum == 0 else k.input_shape_0,
        input_shape_1=self.output_domain_shape if argnum == 1 else k.input_shape_1,
        output_shape_0=self.output_codomain_shape if argnum == 0 else k.output_shape_0,
        output_shape_1=self.output_codomain_shape if argnum == 1 else k.output_shape_1,
    )


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _tensor_product.TensorProduct,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    D0 = (
        self
        if argnum == 0
        else diffops.PartialDerivative(diffops.MultiIndex(np.zeros(k.input_shape_0)))
    )
    D1 = (
        self
        if argnum == 1
        else diffops.PartialDerivative(diffops.MultiIndex(np.zeros(k.input_shape_1)))
    )
    return _tensor_product.TensorProduct_LinDiffOp_LinDiffOp(k, L0=D0, L1=D1)


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _tensor_product.TensorProduct_LinDiffOp_LinDiffOp,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    if argnum == 0 and k.L0.order == 0:
        D0 = self
        D1 = k.L1
    elif argnum == 1 and k.L1.order == 0:
        D0 = k.L0
        D1 = self
    else:
        return NotImplemented
    return _tensor_product.TensorProduct_LinDiffOp_LinDiffOp(k.k, L0=D0, L1=D1)


########################################################################################
# CoefficientFieldOperator #############################################################
########################################################################################


@diffops.CoefficientFieldOperator.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: covfuncs.JaxCovarianceFunctionMixin,
    /,
    *,
    argnum: int = 0,
):
    r"""Apply pointwise multiplication by c(x) along one input axis of a kernel.

    Returns a :class:`FieldScaledCovarianceFunction` so that subsequent diffop
    applications along the *opposite* ``argnum`` can still dispatch to the
    specialized handlers of the base kernel.
    """
    validate_covfunc_transformation(self, k, argnum)

    return FieldScaledCovarianceFunction(
        coefficient_field=self.coefficient_field,
        base_kernel=k,
        scaled_argnum=argnum,
    )


# Distribute the coefficient-field multiplication over multi-output structure so
# that each leaf is wrapped as a scalar FieldScaledCovarianceFunction. This is
# necessary for the variable-coefficient Helmholtz pipeline against 2-component
# priors: without it the JaxCovarianceFunctionMixin handler above is applied to
# a (2,)-output covariance and downstream LDO applications fail validation.

@diffops.CoefficientFieldOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    L_covfuncs = np.copy(k.covfuncs)
    for idx, covfunc in np.ndenumerate(L_covfuncs):
        L_covfuncs[idx] = self(covfunc, argnum=argnum)
    return covfuncs.StackCovarianceFunction(L_covfuncs, output_idx=k.output_idx)


@diffops.CoefficientFieldOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxSumCovarianceFunction, /, *, argnum: int = 0):
    return covfuncs.JaxSumCovarianceFunction(
        *(self(summand, argnum=argnum) for summand in k.summands)
    )


@diffops.CoefficientFieldOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxScaledCovarianceFunction, /, *, argnum: int = 0):
    return k.scalar * self(k.covfunc, argnum=argnum)


@diffops.CoefficientFieldOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.Zero, /, *, argnum: int = 0):
    # Multiplication by c(x) leaves Zero a Zero, with the same shapes.
    return covfuncs.Zero(
        input_shape_0=k.input_shape_0,
        input_shape_1=k.input_shape_1,
        output_shape_0=k.output_shape_0,
        output_shape_1=k.output_shape_1,
    )


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: FieldScaledCovarianceFunction,
    /,
    *,
    argnum: int = 0,
):
    """Apply any linear differential operator to a ``FieldScaledCovarianceFunction``.

    If ``argnum`` differs from the kernel's ``scaled_argnum`` then ``c`` is a
    spectator with respect to the differentiation, and we route the operator
    application through the base kernel (which preserves specialized handlers
    for Matern, ExpQuad, etc.). Otherwise we fall back to the generic JAX
    path — this is the case ``L (c(x_arg) k)`` along the same axis ``c``
    depends on; for the variable-coefficient Helmholtz pipeline this branch
    is *not* exercised because the only consumer applies the operator along
    one argnum at a time.
    """
    validate_covfunc_transformation(self, k, argnum)

    if argnum != k.scaled_argnum:
        inner = self(k.base_kernel, argnum=argnum)
        return FieldScaledCovarianceFunction(
            coefficient_field=k.coefficient_field,
            base_kernel=inner,
            scaled_argnum=k.scaled_argnum,
        )

    # Same-argnum case: fall through to the JAX fallback path.
    try:
        return self._call_no_jax(k, argnum=argnum)  # pylint: disable=protected-access
    except NotImplementedError:
        return covfuncs.JaxLambdaCovarianceFunction(
            self._jax_fallback(  # pylint: disable=protected-access
                k.jax, argnum=argnum
            ),
            input_shape=self.output_domain_shape,
            output_shape_0=k.output_shape_0,
            output_shape_1=k.output_shape_1,
            vectorize=True,
        )


@diffops.CoefficientFieldOperator.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: FieldScaledCovarianceFunction,
    /,
    *,
    argnum: int = 0,
):
    """Multiply a FieldScaledCovarianceFunction by another coefficient field."""
    validate_covfunc_transformation(self, k, argnum)

    if argnum != k.scaled_argnum:
        # The two coefficient fields act on different input axes, so we can
        # simply nest: c_outer(x_argnum) * (c_inner(x_other) * k_base).
        inner = FieldScaledCovarianceFunction(
            coefficient_field=self.coefficient_field,
            base_kernel=k.base_kernel,
            scaled_argnum=argnum,
        )
        return FieldScaledCovarianceFunction(
            coefficient_field=k.coefficient_field,
            base_kernel=inner,
            scaled_argnum=k.scaled_argnum,
        )

    # Same-argnum case: multiply two functions of the same coordinate. We
    # wrap the resulting product field as a JaxLambdaFunction.
    c_outer = self.coefficient_field
    c_inner = k.coefficient_field

    if isinstance(c_outer, _functions.JaxFunction) and isinstance(
        c_inner, _functions.JaxFunction
    ):
        def _prod(x, _co=c_outer, _ci=c_inner):
            return _co.jax(x) * _ci.jax(x)
    else:
        def _prod(x, _co=c_outer, _ci=c_inner):
            co = _co.jax(x) if hasattr(_co, "jax") else jnp.asarray(
                np.asarray(_co(np.asarray(x)))
            )
            ci = _ci.jax(x) if hasattr(_ci, "jax") else jnp.asarray(
                np.asarray(_ci(np.asarray(x)))
            )
            return co * ci

    new_field = _functions.JaxLambdaFunction(
        _prod,
        input_shape=self.output_domain_shape,
        output_shape=(),
        vectorize=True,
    )

    return FieldScaledCovarianceFunction(
        coefficient_field=new_field,
        base_kernel=k.base_kernel,
        scaled_argnum=argnum,
    )


########################################################################################
# Partial Derivative ###################################################################
########################################################################################


def _partial_derivative_fallback(
    D: diffops.PartialDerivative, k: pn_covfuncs.CovarianceFunction, argnum: int = 0
):
    validate_covfunc_transformation(D, k, argnum)

    if D.order == 0:
        return k
    if int(np.prod(D.input_domain_shape)) == 1:
        if D.order == 1:
            return diffops.DirectionalDerivative(1.0)(k, argnum=argnum)
        if D.order == 2:
            return diffops.WeightedLaplacian(1.0)(k, argnum=argnum)
    return NotImplemented


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.HalfIntegerMatern_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.HalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.ExpQuad, /, *, argnum: int = 0):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    return _partial_derivative_fallback(self, k, argnum=argnum)


########################################################################################
# Directional Derivative ###############################################################
########################################################################################


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if k.p is not None:
        return _matern.HalfIntegerMatern_Identity_DirectionalDerivative(
            k,
            direction=self.direction,
            reverse=(argnum == 0),
        )

    return super(diffops.DirectionalDerivative, self).__init__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.HalfIntegerMatern_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None

    if argnum == 0 and not k.reverse:
        return (
            _matern.UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative  # pylint: disable=line-too-long
            if k.matern.input_size == 1
            else _matern.HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
        )(
            k.matern,
            direction0=self.direction,
            direction1=k.direction,
        )

    if argnum == 1 and k.reverse:
        return (
            _matern.UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative  # pylint: disable=line-too-long
            if k.matern.input_size == 1
            else _matern.HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
        )(
            k.matern,
            direction0=k.direction,
            direction1=self.direction,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return (
            _matern.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian(
                k.matern,
                direction=self.direction,
                L1=k.L,
                reverse=(argnum == 1),
            )
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.ExpQuad, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    return _expquad.ExpQuad_Identity_DirectionalDerivative(
        expquad=k,
        direction=self.direction,
        reverse=(argnum == 0),
    )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if argnum == 0 and not k.reverse:
        return _expquad.ExpQuad_DirectionalDerivative_DirectionalDerivative(
            expquad=k.expquad,
            direction0=self.direction,
            direction1=k.direction,
        )

    if argnum == 1 and k.reverse:
        return _expquad.ExpQuad_DirectionalDerivative_DirectionalDerivative(
            expquad=k.expquad,
            direction0=k.direction,
            direction1=self.direction,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return _expquad.ExpQuad_DirectionalDerivative_WeightedLaplacian(
            k.expquad,
            direction=self.direction,
            L1=k.L,
            reverse=(argnum == 1),
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


########################################################################################
# (Weighted) Laplacian #################################################################
########################################################################################


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if k.input_size == 1:
        if k.p is not None:
            return _matern.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian(
                k, L=self, reverse=(argnum == 0)
            )
    elif k.p is not None and k.p >= 2:
        # Multi-D half-integer Matern with smoothness >= 5/2: use the
        # closed-form handler so subsequent diffop applications (e.g. for
        # Helmholtz operators) avoid the JAX hessian fallback that NaNs
        # the Gram diagonal.
        return _matern.HalfIntegerMatern_Identity_WeightedLaplacian(
            k, L=self, reverse=(argnum == 0)
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None
    assert k.input_size == 1

    if argnum == 0 and not k.reverse:
        return _matern.UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=self, L1=k.L
        )

    if argnum == 1 and k.reverse:
        return _matern.UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=k.L, L1=self
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.HalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None
    assert k.matern.p >= 2
    assert k.input_size > 1

    if argnum == 0 and not k.reverse:
        return _matern.HalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=self, L1=k.L
        )

    if argnum == 1 and k.reverse:
        return _matern.HalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=k.L, L1=self
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.HalfIntegerMatern_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None
    assert k.input_size == 1

    if k.input_size == 1:
        if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
            return _matern.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian(  # pylint: disable=line-too-long
                k.matern,
                direction=k.direction,
                L1=self,
                reverse=(argnum == 0),
            )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.ExpQuad, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    return _expquad.ExpQuad_Identity_WeightedLaplacian(k, L=self, reverse=argnum == 0)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if argnum == 0 and not k.reverse:
        return _expquad.ExpQuad_WeightedLaplacian_WeightedLaplacian(
            k.expquad, L0=self, L1=k.L
        )

    if argnum == 1 and k.reverse:
        return _expquad.ExpQuad_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=k.L, L1=self
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return _expquad.ExpQuad_DirectionalDerivative_WeightedLaplacian(
            k.expquad,
            direction=k.direction,
            L1=self,
            reverse=(argnum == 0),
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)
