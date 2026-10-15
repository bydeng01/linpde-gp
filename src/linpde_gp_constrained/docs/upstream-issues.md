# Backend compatibility

The integration package uses the conditional-kernel formula to avoid unsupported
JAX paths in the equality posterior. It also wraps NumPy percentile calls made by
`gp_constr`. These workarounds are implemented in the wrapper; they do not edit the
submodule source files.

## Derivatives of the conditional covariance

The following `_evaluate_jax` methods call NumPy interfaces that cannot accept
JAX tracers:

| Class | Source file | NumPy call inside the JAX method |
| --- | --- | --- |
| `PriorPredictiveCrossCovariance` | `src/linpde_gp/randprocs/_gaussian_process/_conditional.py` | `kLa(x)` |
| `CovarianceFunction_Identity_Evaluation` | `src/linpde_gp/randprocs/crosscov/linfunctls/_evaluation.py` | `self._covfunc(x, X)` |
| `CovarianceFunction_Evaluation_Identity` | `src/linpde_gp/randprocs/crosscov/linfunctls/_evaluation.py` | `self._covfunc(x, X)` |

These calls can reach `np.asarray(tracer)` and raise
`TracerArrayConversionError`. Their JAX counterparts are `kLa.jax(x)` and
`self._covfunc.jax(x, X)`. Changing those calls alone would not make the full
conditional covariance traceable: its `_evaluate_jax` method also calls
`self._gram_matrix.solve(...)`, backed by NumPy-based `probnum` linear operators.

`LinPDEKernelAdapter` computes derivative kernels from

```text
k_E(x, x') = k(x, x') - kLa(x)^T G^{-1} kLa(x'),
```

where `kLa` applies the equality functionals to the prior kernel and `G` is their
Gram matrix including equality-observation noise. It differentiates the prior
kernel and its cross-covariances, then solves on NumPy arrays. ExpQuad and Matérn
priors use the available closed-form derivative handlers. This avoids tracing
the conditional covariance through JAX. An adapter that instead differentiates
`posterior.cov` directly would encounter the limitations above.

See [kernel_adapter.py](../linpde_gp_constrained/kernel_adapter.py) and
[test_kernel_adapter_integration.py](../tests/test_kernel_adapter_integration.py).

## Derivatives of the posterior mean

`pn.functions.Zero` is not a `JaxFunction`; differentiating it through the JAX
fallback can also reach `np.asarray(tracer)`. The weak-form equality constructor
requires a zero-mean prior and supplies representer weights directly to avoid
this path.

For derivative inequality bounds, `ConstrainedLinPDEPosterior` evaluates the
posterior mean by central finite differences with step `1e-4`. Identity bounds
use the posterior mean directly. This numerical mean derivative is distinct
from the analytic covariance derivatives above.

See [pde_equality_posterior.py](../linpde_gp_constrained/pde_equality_posterior.py)
and [constrained_posterior.py](../linpde_gp_constrained/constrained_posterior.py).

## NumPy percentiles on `gp_constr` samples

`gp_constr/GPConstr/model.py` stores constrained samples as `np.matrix`.
With NumPy 2.x, percentile interpolation can invoke matrix multiplication
instead of elementwise multiplication and raise a shape-mismatch error. A
minimal reproducer under NumPy 2.x is:

```python
np.percentile(np.matrix(np.zeros((5, 8))), [10, 50, 90], axis=1)
```

`_matrix_safe_numpy_percentile` temporarily rebinds `np.percentile` around the
`gp_constr` call and converts matrix inputs with `np.asarray`. It restores the
original binding on exit. The binding is module-wide while the context is
active, so concurrent NumPy calls can see the wrapper. Converting the sample
matrix to an ndarray before interpolation avoids the incompatible multiplication.
