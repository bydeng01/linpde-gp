"""
linpde_gp_constrained: integration layer between linpde-gp (Pförtner et al.,
arXiv 2212.12474) and gp_constr (Agrell, JMLR 2019).

Contents
--------
ConstrainedLinPDEPosterior
    The end-user-facing wrapper. Conditions a linpde-gp posterior u | E on a
    list of inequality constraints C and exposes constrained posterior
    means and samples. Internally builds the kernel adapter and a
    populated ``gp_constr.GPmodel``; folds m_E into the inequality bounds
    automatically; re-adds m_E on output.

InequalityConstraint
    Dataclass describing one constraint a ≤ L_ineq u(X_v) ≤ b. Supported
    operators: identity (boundedness) and first-order single-axis partial
    derivatives.

LinPDEKernelAdapter
    Wraps a linpde-gp ConditionalGaussianProcess so it can be passed to
    gp_constr's GPmodel. Translates between the two libraries' kernel APIs
    without modifying upstream code.

PriorKernelAdapter
    A subclass of LinPDEKernelAdapter that exposes a *prior* GP (no equality
    conditioning) under the same kernel API. Used for the inequality-only arm
    of the 3-way ablation (equality-only vs inequality-only vs both): the prior
    is truncated by the shape constraints directly, with no PDE/BC information,
    so the conditional kernel collapses to the prior kernel.

transform_bounds_for_zero_mean
    Helper for the mean-subtraction trick: passes mean=0 to gp_constr while
    folding the LinPDE-GP posterior mean m_E into spatially-varying inequality
    bounds. Mathematically equivalent to the original constraint.

Design notes
------------
The integration is a thin wrapper. By design we never modify
linpde-gp/src/linpde_gp/ or gp_constr/GPConstr/.

Two-stage conditional inference is enforced: equality-side hyperparameters are
fit and frozen before this adapter is built. The adapter raises if any caller
attempts to invoke gp_constr's optimize() path through set_params /
get_params / K_gradients. Calling GPmodel.optimize() on this adapter is a bug.
"""

from .constrained_posterior import ConstrainedLinPDEPosterior, InequalityConstraint
from .kernel_adapter import (
    LinPDEKernelAdapter,
    NoHyperparamOptimization,
    PriorKernelAdapter,
)
from .mean_handling import transform_bounds_for_zero_mean
from .pde_equality_posterior import (
    make_pde_collocation_posterior,
    make_pde_equality_posterior,
)

__all__ = [
    "ConstrainedLinPDEPosterior",
    "InequalityConstraint",
    "LinPDEKernelAdapter",
    "NoHyperparamOptimization",
    "PriorKernelAdapter",
    "make_pde_collocation_posterior",
    "make_pde_equality_posterior",
    "transform_bounds_for_zero_mean",
]
