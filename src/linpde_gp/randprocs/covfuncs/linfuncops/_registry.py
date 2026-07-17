import numpy as np
from probnum.randprocs import covfuncs as pn_covfuncs

from linpde_gp import linfuncops
from linpde_gp.randprocs import covfuncs

from .._utils import validate_covfunc_transformation

########################################################################################
# General `LinearFunctionOperators` ####################################################
########################################################################################


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(
    self, k: covfuncs.JaxScaledCovarianceFunction, /, *, argnum: int = 0
) -> covfuncs.JaxScaledCovarianceFunction:
    validate_covfunc_transformation(self, k, argnum)

    return k.scalar * self(k.covfunc, argnum=argnum)


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(
    self, k: covfuncs.JaxSumCovarianceFunction, /, *, argnum: int = 0
) -> covfuncs.JaxSumCovarianceFunction:
    validate_covfunc_transformation(self, k, argnum)

    return covfuncs.JaxSumCovarianceFunction(
        *(self(summand, argnum=argnum) for summand in k.summands)
    )


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and k.output_idx == 1) or (argnum == 1 and k.output_idx == 0):
        L_covfuncs = np.copy(k.covfuncs)
        for idx, covfunc in np.ndenumerate(L_covfuncs):
            L_covfuncs[idx] = self(covfunc, argnum=argnum)

        return covfuncs.StackCovarianceFunction(
            L_covfuncs,
            output_idx=k.output_idx,
        )

    raise NotImplementedError()


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.Zero, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    return covfuncs.Zero(
        input_shape_0=self.output_domain_shape if argnum == 0 else k.input_shape_0,
        input_shape_1=self.output_domain_shape if argnum == 1 else k.input_shape_1,
        output_shape_0=self.output_codomain_shape if argnum == 0 else k.output_shape_0,
        output_shape_1=self.output_codomain_shape if argnum == 1 else k.output_shape_1,
    )


########################################################################################
# `Identity` ###########################################################################
########################################################################################


@linfuncops.Identity.__call__.register  # pylint: disable=no-member
def _(
    self, covfunc: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0
) -> pn_covfuncs.CovarianceFunction:
    validate_covfunc_transformation(self, covfunc, argnum)

    return covfunc


########################################################################################
# `SelectOutput` #######################################################################
########################################################################################


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and k.output_idx == 0) or (argnum == 1 and k.output_idx == 1):
        return k.covfuncs[self.idx]

    return super(linfuncops.SelectOutput, self).__call__(k, argnum=argnum)


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: covfuncs.IndependentMultiOutputCovarianceFunction,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    zero_cov = covfuncs.Zero(
        input_shape_0=k.input_shape_0,
        input_shape_1=k.input_shape_1,
        output_shape_0=(),
        output_shape_1=(),
    )

    assert isinstance(self.idx, int)

    return covfuncs.StackCovarianceFunction(
        tuple(
            (
                *([zero_cov] * self.idx),
                k.covfuncs[self.idx],
                *([zero_cov] * (len(k.covfuncs) - self.idx - 1)),
            )
        ),
        output_idx=1 - argnum,
    )


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: covfuncs.CoregionalizedMultiOutputCovarianceFunction,
    /,
    *,
    argnum: int = 0,
):
    """Select one output channel of an ICM (coregionalized) covariance.

    Phase 8b. ICM has ``Cov[u_c(x), u_c'(x')] = B[c, c'] * base(x, x')``, so
    selecting channel ``idx`` along ``argnum`` leaves a vector over the *other*
    channel: component ``c`` is ``B[idx, c] * base`` (``B`` is symmetric, so
    the row and column of ``B`` coincide). This parallels the
    ``IndependentMultiOutputCovarianceFunction`` handler above, except the
    off-diagonal entries are scaled copies of the shared base kernel rather
    than ``Zero`` — that is exactly how the coregionalization matrix ``B``
    rides along, untouched, into the downstream specialized handlers.

    Each component is either a ``Zero`` (for an exactly-zero coupling, which
    makes ``B = I`` reduce to the IID structure) or a
    ``JaxScaledCovarianceFunction`` wrapping the shared base kernel. The latter
    is resolved by the existing Phase 7b ``JaxScaledCovarianceFunction``
    operator handlers (``scalar * L(base)``), so ``L`` always reaches the base
    kernel's closed-form path and never falls through to ``JaxLambda``.
    """
    validate_covfunc_transformation(self, k, argnum)

    assert isinstance(self.idx, int)

    B = k.B
    base = k.base
    couplings = B[self.idx, :]

    zero_cov = covfuncs.Zero(
        input_shape_0=k.input_shape_0,
        input_shape_1=k.input_shape_1,
        output_shape_0=(),
        output_shape_1=(),
    )

    components = tuple(
        (zero_cov if coupling == 0.0 else float(coupling) * base)
        for coupling in couplings
    )

    return covfuncs.StackCovarianceFunction(components, output_idx=1 - argnum)


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxSumCovarianceFunction, /, *, argnum: int = 0):
    """Distribute :class:`SelectOutput` over the summands of a JaxSumCovarianceFunction.

    SelectOutput has its own singledispatch table that does not inherit the
    LinearFunctionOperator-level handler for JaxSumCovarianceFunction. Without
    this explicit registration, selection on a sum collapses into a generic
    ``SelectedCovarianceFunction`` wrapper and downstream LDO handlers can no
    longer reach the closed-form Matern paths.
    """
    return covfuncs.JaxSumCovarianceFunction(
        *(self(summand, argnum=argnum) for summand in k.summands)
    )


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxScaledCovarianceFunction, /, *, argnum: int = 0):
    """Pull the scalar through :class:`SelectOutput` on a scaled covfunc."""
    return k.scalar * self(k.covfunc, argnum=argnum)


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.Zero, /, *, argnum: int = 0):
    """Selecting from a Zero kernel yields another Zero with the selected codomain.

    SelectOutput's singledispatch table shadows the LinearFunctionOperator-level
    Zero handler unless we register explicitly here.
    """
    return covfuncs.Zero(
        input_shape_0=k.input_shape_0,
        input_shape_1=k.input_shape_1,
        output_shape_0=self.output_codomain_shape if argnum == 0 else k.output_shape_0,
        output_shape_1=self.output_codomain_shape if argnum == 1 else k.output_shape_1,
    )
