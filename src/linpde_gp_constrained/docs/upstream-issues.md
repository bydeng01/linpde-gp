# Upstream linpde-gp issues encountered during the gp_constr integration

This document is the migrated content of the former `_compat.py` module.
These issues are documented here as candidates for upstream PRs against
[linpde-gp](https://github.com/marvinpfoertner/linpde-gp). The integration
package (`linpde_gp_constrained`) **no longer applies any of these patches
at runtime** — the kernel adapter now bypasses the broken JAX paths by
computing derivative kernels directly from the conditional-kernel formula
applied to the prior covariance function. See
`linpde_gp_constrained/kernel_adapter.py` for the architectural choice.

## Common pattern

All three patches fix the same upstream pattern: a method named
`_evaluate_jax` calls a sub-object's numpy `__call__` instead of its
JAX-aware `.jax(x)`. When the surrounding trace passes a JAX tracer in,
the `__call__` path eventually does `np.asarray(tracer)` and crashes.

These bugs are triggered the moment any caller applies a `PartialDerivative`
(or any other `LinearDifferentialOperator` that uses the JAX autodiff
fallback) to `ConditionalGaussianProcess.cov` — exactly what a kernel
adapter would do to build `Ki0` and `Kij` if it routed through the
`posterior.cov` JAX path. linpde-gp's own tests do not exercise this
because they apply operators to the posterior GP itself (`L(posterior)`),
not to its bivariate covariance function.

## Patch 1 — `PriorPredictiveCrossCovariance._evaluate_jax`

- File: `src/linpde_gp/randprocs/_gaussian_process/_conditional.py`, line 161.
- Bug: calls `kLa(x)` (numpy `__call__`) inside a JAX-traced method.
- Fix: replace with `kLa.jax(x)`. One-character change.
- Verified-equivalent JAX implementation (matches the structure of the
  parallel `Mean._evaluate_jax` in the same file):

```python
def _evaluate_jax(self, x):
    batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]
    return jnp.concatenate(
        [
            jnp.reshape(
                kLa.jax(x),
                batch_shape + self.randproc_output_shape + (-1,),
                "C",
            )
            for kLa in self._kLas
        ],
        axis=-1,
    )
```

## Patch 2 — `CovarianceFunction_Identity_Evaluation._evaluate_jax`

- File: `src/linpde_gp/randprocs/crosscov/linfunctls/_evaluation.py`, line 136.
- Bug: calls `self._covfunc(x, X)` (numpy `__call__`) inside a JAX-traced method.
- Fix: replace with `self._covfunc.jax(x, X)`. One-character change.
- The full method body is correct other than this single call:

```python
def _evaluate_jax(self, x):
    # `kxX.shape` layout:
    # x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
    x_batch_shape = x.shape[: x.ndim - self._covfunc.input_ndim]
    x_batch_ndim = len(x_batch_shape)
    x_batch_offset = 0

    X_batch_shape = self._evaluation_fctl.X_batch_shape
    X_batch_ndim = self._evaluation_fctl.X_batch_ndim
    X_batch_offset = x_batch_offset + x_batch_ndim

    x_output_shape = self.randproc_output_shape
    x_output_ndim = self.randproc_output_ndim
    x_output_offset = X_batch_offset + X_batch_ndim

    X_output_shape = self._evaluation_fctl.input_codomain_shape
    X_output_ndim = self._evaluation_fctl.input_codomain_ndim
    X_output_offset = x_output_offset + x_output_ndim

    assert x.shape == x_batch_shape + self._covfunc.input_shape

    x = jnp.expand_dims(
        x,
        axis=tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim)),
    )

    assert (
        x.shape == x_batch_shape + X_batch_ndim * (1,) + self._covfunc.input_shape
    )

    X = self._evaluation_fctl.X

    assert X.shape == X_batch_shape + self._covfunc.input_shape

    # Patched: was ``self._covfunc(x, X)``.
    kxX = self._covfunc.jax(x, X)

    assert kxX.shape == (
        x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
    )

    kxX = kxX.transpose(
        tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
        + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
        + tuple(range(X_output_offset, X_output_offset + X_output_ndim))
        + tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
    )

    assert kxX.shape == (
        x_batch_shape + x_output_shape + X_output_shape + X_batch_shape
    )

    return kxX
```

## Patch 3 — `CovarianceFunction_Evaluation_Identity._evaluate_jax`

- File: `src/linpde_gp/randprocs/crosscov/linfunctls/_evaluation.py`, line 291.
- Bug: same one-character bug as Patch 2 in the symmetric (transposed) sibling.
- Fix: replace `self._covfunc(x, X)` with `self._covfunc.jax(x, X)`.

```python
def _evaluate_jax(self, x):
    # `kXx.shape` layout:
    # X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
    X_batch_shape = self._evaluation_fctl.X_batch_shape
    X_batch_ndim = self._evaluation_fctl.X_batch_ndim
    X_batch_offset = 0

    x_batch_shape = x.shape[: x.ndim - self._covfunc.input_ndim]
    x_batch_ndim = len(x_batch_shape)
    x_batch_offset = X_batch_offset + X_batch_ndim

    X_output_shape = self._evaluation_fctl.input_codomain_shape
    X_output_ndim = self._evaluation_fctl.input_codomain_ndim
    X_output_offset = x_batch_offset + x_batch_ndim

    x_output_shape = self.randproc_output_shape
    x_output_ndim = self.randproc_output_ndim
    x_output_offset = X_output_offset + X_output_ndim

    assert x.shape == x_batch_shape + self._covfunc.input_shape

    X = jnp.expand_dims(
        self._evaluation_fctl.X,
        axis=tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim)),
    )

    assert X.shape == (
        self._evaluation_fctl.X_batch_shape
        + x_batch_ndim * (1,)
        + self._covfunc.input_shape
    )

    # Patched: was ``self._covfunc(x, X)``.
    kxX = self._covfunc.jax(x, X)

    assert kxX.shape == (
        X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
    )

    kxX = kxX.transpose(
        tuple(range(X_output_offset, X_output_offset + X_output_ndim))
        + tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
        + tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
        + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
    )

    assert kxX.shape == (
        X_output_shape + X_batch_shape + x_batch_shape + x_output_shape
    )

    return kxX
```

## Patch 4 (architectural, not a typo)

`ConditionalGaussianProcess.CovarianceFunction._evaluate_jax` calls
`self._gram_matrix.solve(...)`, which dispatches to probnum's numpy-only
`LinearOperator.solve`. That method can never accept JAX tracers, so the
JAX path is structurally broken — the bug is not a one-character typo but
the absence of a JAX-traceable solve on the gram matrix. A proper upstream
fix would either provide a JAX-aware solve on the gram matrix when
needed, or cleanly mark the JAX path of the conditional covariance as
unsupported.

## Patch 5 (gp_constr × numpy 2.x — `np.percentile` on `np.matrix`)

- File: `gp_constr/GPConstr/model.py`, line 380, inside
  `calc_posterior_constrained`.
- Bug: `gp_constr` stores the sample tensor `fs_sim` as `np.matrix`. In
  numpy 2.x, `np.percentile` internally evaluates `diff * t` on a slice
  of its input. If the input is `np.matrix`, the slice is also
  `np.matrix`, whose `__mul__` performs matrix multiplication (`np.dot`)
  instead of elementwise multiplication. The shape mismatch raises
  `ValueError: shapes (k, n) and (k, 1) not aligned`.
- Reproduction: `np.percentile(np.matrix(np.zeros((5, 8))), [10, 50, 90],
  axis=1)` under numpy ≥ 2.0.
- Two clean upstream fixes:
  1. `gp_constr` converts `fs_sim` to plain `np.ndarray` before computing
     percentiles (`np.asarray(fs_sim)` once at the top of the routine).
  2. Or: replace all `np.matrix` usage in gp_constr with plain
     `np.ndarray` (matrix has been deprecated since NumPy 1.15, see
     https://numpy.org/doc/stable/reference/generated/numpy.matrix.html).
     This is the ergonomically correct fix but a much bigger change.
- Also affects the same module's `_constrprob_xs_2*` helpers if they ever
  surface `fs_sim`-style matrices into percentile calls (none do at
  present, but the np.matrix legacy is project-wide).

### Wrapper-side workaround (used at runtime)

`linpde_gp_constrained.constrained_posterior._matrix_safe_numpy_percentile`
is a context manager that locally rebinds `np.percentile` to a wrapper
which calls `np.asarray(a)` when `a` is `np.matrix`. We open this
context only around the gp_constr call; numpy state is restored on
exit, no global side effect.

## Why we no longer patch the JAX bugs at runtime

(Patches 1–4 only — patch 5 is still applied at runtime via the context
manager described above, scoped to the single gp_constr call.)

The kernel adapter avoids the JAX paths above by computing the conditional
kernel and its derivatives directly from

```
k_E(x, x') = k(x, x') - k(x, X_E)^T G^{-1} k(X_E, x'),
```

differentiating only the *prior* kernel `k` (which has clean closed-form
handlers for ExpQuad/Matern in linpde-gp). The gram-matrix solve is on
numpy arrays (no JAX tracing). This bypasses Patches 1–3 entirely and
sidesteps the architectural Patch 4.

If a future change to the kernel adapter starts routing through
`ConditionalGaussianProcess.CovarianceFunction._evaluate_jax` again, these
patches (or proper upstream fixes) will become necessary again.
