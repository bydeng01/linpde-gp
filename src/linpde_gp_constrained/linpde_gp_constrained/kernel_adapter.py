"""
Kernel adapter exposing the gp_constr kernel API on a linpde-gp posterior GP.

The mathematical object being implemented is the conditional covariance
function

    k_E(x, x') = k(x, x') - kLa(x)^T G^{-1} kLa(x')

where ``kLa(x)`` is the row vector obtained by applying each equality
functional ``L_l`` (l = 1, ..., L) to the second argument of the prior
kernel and concatenating:

    kLa(x) = [ L_1 k(x, ·), L_2 k(x, ·), ..., L_L k(x, ·) ] in R^{n_E_total}

with ``n_E_total = sum_l output_size(L_l)``. ``G`` is the full block gram
matrix (already assembled by linpde-gp inside ``posterior_gp.gram``).

When every ``L_l`` is an ``_EvaluationFunctional`` —
i.e., point measurement / point boundary condition — so

    L_l k(x, ·) = k(x, X_E^{(l)}),     kLa(x) = k(x, X_E_total).

This adapter generalises to ``L_l`` arising from PDE residuals discretised on
a finite test basis (``L2Projection_UnivariateLinearInterpolationBasis``,
``CompositeLinearFunctional`` of the form ``l2_proj @ Laplacian``, etc.).
The Schur formula above is unchanged; only ``kLa(x)`` and its first
derivative ``∂_{x_i} kLa(x)`` need to be computed via linpde-gp's cross-cov
dispatch instead of by direct kernel evaluation at evaluation points.

The adapter exposes derivative kernels under the method names that
gp_constr's ``kernel_Stationary`` interface requires:

    K(X1, X2)         shape (n1, n2)   k_E(X1[a], X2[b])
    K_diag(X)         shape (n,)       k_E(X[a], X[a])
    Ki0(X1, X2, i)    shape (n1, n2)   (d/dx_i) k_E(x, x')                  | first arg
                                         evaluated at (X1[a], X2[b])
    Kij(X1, X2, i, j) shape (n1, n2)   (d/dx_i)(d/dx'_j) k_E(x, x')         | first by i,
                                         evaluated at (X1[a], X2[b])           second by j
    Kii_diag(X, i)    shape (n,)       diag of Kij(X, X, i, i)

Derivative kernels use the conditional-covariance formula above, with
linpde-gp's PartialDerivative dispatch on the prior kernel and cross-covariances (see
src/linpde_gp/randprocs/covfuncs/linfuncops/diffops/_registry.py).
For ExpQuad / Matern priors with explicit handlers this hits a closed-form
path; otherwise linpde-gp falls back to a JAX autodiff implementation
(_partial_derivative.py:95). Both routes are exact — no finite-difference
approximation is used.

Argument-number convention reconciliation
-----------------------------------------
gp_constr's Ki0 differentiates the FIRST kernel argument (verified
empirically: see verify_gpconstr_conventions.py; for an RBF
k(x, x') = sigma^2 exp(-0.5 ||x - x'||^2/ell^2), gp_constr.kernel_RBF.Ki0
returns -k * (x_i - x'_i) / ell_i^2, which matches d/d x_i analytically.).
linpde-gp's argnum=0 also means "differentiate the first kernel argument"
(see randprocs/covfuncs/linfuncops/_abstract.py:62 -
'argnum' returns 0 if reverse else 1, and reverse=True corresponds to
operator-on-first-argument). The conventions therefore agree directly:
    gp_constr Ki0  <->  linpde-gp PartialDerivative(...)(k_E, argnum=0)
    gp_constr Kij  <->  linpde-gp Partial_i(argnum=0) o Partial_j(argnum=1)

Hyperparameter-optimization regression guard
--------------------------------------------
gp_constr's GPmodel.optimize() reaches into the kernel via
set_params/get_params/K_gradients. The two-stage integration framework
freezes equality-side hyperparameters before this adapter is built; calling
optimize() through the adapter would silently break that contract. We raise
NoHyperparamOptimization on these calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NoHyperparamOptimization(RuntimeError):
    """Raised when a caller attempts hyperparameter optimization via the adapter.

    The two-stage LinPDE-GP × gp_constr integration freezes equality-side
    hyperparameters before the inequality-conditioning step. Hyperparameter
    optimization through gp_constr's GPmodel.optimize() would bypass that
    framing, so we raise instead of silently allowing it.
    """


class LinPDEKernelAdapter:
    """Adapter exposing a linpde-gp posterior GP under gp_constr's kernel API.

    Parameters
    ----------
    posterior_gp
        A linpde-gp ConditionalGaussianProcess (the result of
        ``prior.condition_on_observations(...)``). Hyperparameters of the
        underlying prior covariance must already be frozen before this adapter
        is constructed; the adapter does not provide a path to retune them.

    Notes
    -----
    The adapter is read-only: it does not mutate posterior_gp. Calls to K and
    K_diag use posterior_gp.cov; Ki0, Kij, and Kii_diag use the conditional-kernel
    formula with derivatives of the prior kernel and cross-covariances. Output
    shapes match gp_constr's documented conventions.
    """

    def __init__(self, posterior_gp: Any) -> None:
        in_shape = posterior_gp.cov.input_shape
        # Conventions:
        #   input_shape == ()    -> scalar-input prior, linpde-gp wants X of shape (n,)
        #   input_shape == (d,)  -> d-dim input prior,  linpde-gp wants X of shape (n, d)
        # gp_constr always passes (n, d). We translate at the boundary.
        if in_shape == ():
            self._d = 1
            self._scalar_input = True
        elif len(in_shape) == 1:
            self._d = int(in_shape[0])
            self._scalar_input = False
        else:
            raise NotImplementedError(
                "LinPDEKernelAdapter only supports input_shape == () or (d,); "
                f"got {in_shape}."
            )

        self._gp = posterior_gp

        # Derivative-kernel machinery (Ki0 / Kij) is deferred to the first
        # call so that K and K_diag work against any object exposing only
        # ``.cov`` (e.g., the covariance-only unit-test mock).
        #
        # Populated by _ensure_derivative_setup():
        #   self._prior_cov   : the prior kernel k(x, x')
        #   self._Ls          : tuple of equality functionals L_l
        #   self._block_sizes : output_size(L_l) per block (for kLa assembly)
        #   self._eval_only   : True iff every L_l is _EvaluationFunctional;
        #                       enables the fast path that bypasses per-L
        #                       cross-cov dispatch.
        #   self._X_E_total   : concatenated X_E (eval-only path); else None.
        #   self._gram        : G = Cov(L_E u, L_E u); supports .solve(b)
        #                       on numpy.
        self._prior_cov = None
        self._Ls: list | None = None
        self._block_sizes: list[int] | None = None
        self._eval_only: bool = False
        self._X_E_total: np.ndarray | None = None
        self._gram = None
        self._derivative_setup_done = False

    def _ensure_derivative_setup(self) -> None:
        """Populate the prior-cov / Ls / gram triple needed for Ki0 / Kij.

        Lazy: only invoked when the caller asks for a derivative kernel.
        Idempotent.

        For each functional ``L_l`` in the equality list we record:
          * its ``output_size`` (so we know the block width contributed to
            ``kLa(x)``);
          * a flag ``_eval_only`` which is True iff every ``L_l`` is a plain
            point-evaluation functional. The all-eval case admits a fast
            path that bypasses linpde-gp's per-functional cross-cov dispatch
            and directly evaluates the prior kernel at the concatenated
            ``X_E``.
        """
        if self._derivative_setup_done:
            return
        # Lazy import: keep this module importable without linpde-gp.
        from linpde_gp.linfunctls._evaluation import _EvaluationFunctional

        self._prior_cov = self._gp._prior.cov  # pylint: disable=protected-access
        self._Ls = list(self._gp._Ls)  # pylint: disable=protected-access
        self._gram = self._gp.gram

        self._block_sizes = [int(np.prod(L.output_shape, dtype=int)) for L in self._Ls]
        self._eval_only = all(isinstance(L, _EvaluationFunctional) for L in self._Ls)

        # Eval-only fast path: store concatenated X_E for direct prior-kernel
        # evaluation. Skipped (set to None) in the generic path where some L
        # is a non-evaluation functional and the per-block cross-cov dispatch
        # is the only valid route.
        if self._eval_only and len(self._Ls) > 0:
            self._X_E_total = np.concatenate(
                [np.asarray(L.X) for L in self._Ls], axis=0
            )
        else:
            self._X_E_total = None
        self._derivative_setup_done = True

    # -------------------------------------------------------------------------
    # Public attributes that gp_constr's model.py reads outside optimize()
    # -------------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """Input dimensionality d (gp_constr passes (n, d)-shaped X)."""
        return self._d

    def __str__(self) -> str:
        return (
            f"LinPDEKernelAdapter(input_dim={self._d}, "
            f"posterior_gp={type(self._gp).__name__})"
        )

    # -------------------------------------------------------------------------
    # Shape adapter: gp_constr always passes (n, d). probnum's
    # CovarianceFunction.__call__ does *broadcasting* over batch dims, not an
    # outer product. To get an (n1, n2) Gram matrix we therefore have to
    # reshape inputs so their batch dims broadcast outward:
    #
    #   input_shape == ()    -> X1: (n1, 1)        X2: (1, n2)
    #                              => bcast batch (n1, n2)
    #   input_shape == (d,)  -> X1: (n1, 1, d)     X2: (1, n2, d)
    #                              => probnum strips trailing (d,), bcast (n1, n2)
    #
    # If we instead passed X1 with shape (n1,) (or (n1, d)) and X2 with shape
    # (n2,) (or (n2, d)), probnum would try to broadcast (n1,) against (n2,)
    # and fail (or, for n1 == n2, return only the diagonal — silently wrong).
    # -------------------------------------------------------------------------
    def _format_for_pair(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reshape gp_constr-style (n, d) inputs to broadcast to an (n1, n2) Gram."""
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        if X1.ndim != 2 or X1.shape[1] != self._d:
            raise ValueError(f"Expected X1 of shape (n, {self._d}); got {X1.shape}.")
        if X2.ndim != 2 or X2.shape[1] != self._d:
            raise ValueError(f"Expected X2 of shape (n, {self._d}); got {X2.shape}.")
        if self._scalar_input:
            # input_shape == () : strip the trailing 1 axis, then broadcast
            # outward so probnum sees batch shapes (n1, 1) and (1, n2).
            return X1[:, 0][:, None], X2[:, 0][None, :]
        # input_shape == (d,) : keep the d axis as the trailing input dim,
        # broadcast the batch axes outward.
        return X1[:, None, :], X2[None, :, :]

    # -------------------------------------------------------------------------
    # Kernel API: K, K_diag, Ki0, Kij, Kii_diag.
    # -------------------------------------------------------------------------
    def K(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """k_E(X1[a], X2[b]) — shape (n1, n2).

        We delegate to ``posterior.cov(...)`` directly. This goes through the
        numpy ``_evaluate`` path (not the broken ``_evaluate_jax`` path) and
        is mathematically the conditional kernel
        ``k_E(x, x') = k(x,x') - k(x,X_E)^T G^{-1} k(X_E, x')``.
        """
        x0, x1 = self._format_for_pair(X1, X2)
        return np.asarray(self._gp.cov(x0, x1))

    def K_diag(self, X: np.ndarray) -> np.ndarray:
        """diag of k_E(X, X) — shape (n,).

        Computed by forming the full (n, n) Gram and extracting the diagonal.
        For very large virtual-point sets a dedicated diagonal evaluator
        would be faster, but correctness first.
        """
        return np.diag(self.K(X, X))

    def Ki0(self, X1: np.ndarray, X2: np.ndarray, i: int) -> np.ndarray:
        """(∂/∂x_i) k_E(x, x') | x=X1[a], x'=X2[b] — shape (n1, n2).

        Implementation: directly from the formula
            ∂_{x_i} k_E(x, x') = ∂_{x_i} k(x, x')
                               - (∂_{x_i} kLa)(x)^T · G^{-1} · kLa(x').
        Only the *prior* kernel is differentiated, so we never go through
        ConditionalGaussianProcess.CovarianceFunction._evaluate_jax (which
        has multiple JAX-tracing bugs in upstream linpde-gp). The gram-matrix
        solve is on numpy arrays — also no JAX tracing.

        ``kLa(x)`` is the concatenation of ``L_l(prior_cov(x, ·))`` over the
        equality functionals (point evaluation, L2 projection, L2 projection
        of the Laplacian, ...); see :meth:`_eval_kLa`. Its first derivative
        is built via the same dispatch chain after differentiating the
        prior; see :meth:`_eval_dki_kLa`.

        Mirror of gp_constr's kernel_Stationary.Ki0
        (gp_constr/GPConstr/kern.py:134-143). First kernel argument
        differentiated.
        """
        if not (0 <= int(i) < self._d):
            raise IndexError(f"i must be in [0, {self._d}); got {i}.")

        self._ensure_derivative_setup()

        Di = self._partial_along(int(i))
        d_prior = Di(self._prior_cov, argnum=0)  # ∂_{x_i} k(·, ·) as a covfunc

        x0, x1 = self._format_for_pair(X1, X2)

        # Term 1: (∂_{x_i} k_prior)(X1, X2) — shape (n1, n2)
        d_kxx = np.asarray(d_prior(x0, x1))

        # Term 2: (∂_{x_i} kLa)(X1) · G^{-1} · kLa(X2)^T   — shape (n1, n2)
        if self._eval_only:
            # Fast path: bivariate-kernel evaluation at concatenated X_E.
            XE_arg0, XE_arg1 = self._format_evaluation_points()
            d_kLa_X1 = np.asarray(d_prior(x0, XE_arg1))  # (n1, n_E_total)
            kLa_X2_T = np.asarray(self._prior_cov(XE_arg0, x1))  # (n_E_total, n2)
        else:
            # Generic path: per-L block assembly via cross-cov dispatch.
            d_kLa_X1 = self._eval_dki_kLa(X1, int(i))  # (n1, n_E_total)
            kLa_X2_T = self._eval_kLa(X2).T  # (n_E_total, n2)

        correction = d_kLa_X1 @ np.asarray(self._gram.solve(kLa_X2_T))
        return d_kxx - correction

    def Kij(self, X1: np.ndarray, X2: np.ndarray, i: int, j: int) -> np.ndarray:
        """(∂/∂x_i)(∂/∂x'_j) k_E(x, x') | x=X1[a], x'=X2[b] — shape (n1, n2).

        Implementation: directly from the formula
            ∂_{x_i} ∂_{x'_j} k_E(x, x') = ∂_{x_i} ∂_{x'_j} k(x, x')
                - (∂_{x_i} kLa)(x)^T · G^{-1} · (∂_{x'_j} kLa)(x').
        Only the prior kernel is differentiated.

        Note that ``kLa`` is a single function of one argument; ``∂_{x_i}
        kLa`` and ``∂_{x'_j} kLa`` denote partial derivatives of that one
        function, evaluated at ``X1`` and ``X2`` respectively (and along
        possibly different axes if the input is multidimensional).

        Mirror of gp_constr's kernel_Stationary.Kij
        (gp_constr/GPConstr/kern.py:145-161). First kernel argument
        differentiated by ``i``, second by ``j``.
        """
        if not (0 <= int(i) < self._d):
            raise IndexError(f"i must be in [0, {self._d}); got {i}.")
        if not (0 <= int(j) < self._d):
            raise IndexError(f"j must be in [0, {self._d}); got {j}.")

        self._ensure_derivative_setup()

        Di = self._partial_along(int(i))
        Dj = self._partial_along(int(j))

        # Build the mixed-derivative prior kernel: ∂_{x_i} ∂_{x'_j} k(·, ·)
        d_prior_argnum1 = Dj(self._prior_cov, argnum=1)
        d_prior_mixed = Di(d_prior_argnum1, argnum=0)

        x0, x1 = self._format_for_pair(X1, X2)

        # Term 1: (∂_{x_i} ∂_{x'_j} k_prior)(X1, X2) — shape (n1, n2)
        d_kxx_ij = np.asarray(d_prior_mixed(x0, x1))

        # Term 2: (∂_{x_i} kLa)(X1) · G^{-1} · (∂_{x'_j} kLa)(X2)^T
        if self._eval_only:
            d_prior_argnum0 = Di(self._prior_cov, argnum=0)
            XE_arg0, XE_arg1 = self._format_evaluation_points()
            d_kLa_X1_i = np.asarray(d_prior_argnum0(x0, XE_arg1))  # (n1, n_E_total)
            d_kLa_X2_j = np.asarray(d_prior_argnum1(XE_arg0, x1))  # (n_E_total, n2)
        else:
            d_kLa_X1_i = self._eval_dki_kLa(X1, int(i))  # (n1, n_E_total)
            d_kLa_X2_j = self._eval_dki_kLa(X2, int(j)).T  # (n_E_total, n2)

        correction = d_kLa_X1_i @ np.asarray(self._gram.solve(d_kLa_X2_j))
        return d_kxx_ij - correction

    def Kii_diag(self, X: np.ndarray, i: int) -> np.ndarray:
        """diag of Kij(X, X, i, i) — shape (n,).

        Implemented via Kij; this is correct but not the most efficient route
        (it builds the full (n, n) matrix and extracts the diagonal). If
        performance on large virtual-point sets becomes an issue, register a
        dedicated diagonal evaluator.
        """
        return np.diag(self.Kij(X, X, i, i))

    # -------------------------------------------------------------------------
    # Hyperparameter regression guards.
    # -------------------------------------------------------------------------
    def set_params(self, theta: np.ndarray) -> None:
        raise NoHyperparamOptimization(
            "set_params() is not supported by LinPDEKernelAdapter. The "
            "two-stage integration freezes equality-side hyperparameters; do "
            "not call GPmodel.optimize() on this adapter."
        )

    def get_params(self) -> np.ndarray:
        raise NoHyperparamOptimization(
            "get_params() is not supported by LinPDEKernelAdapter (see "
            "set_params for context)."
        )

    def K_gradients(self, X1: np.ndarray, X2: np.ndarray) -> list:
        raise NoHyperparamOptimization(
            "K_gradients() is not supported by LinPDEKernelAdapter (see "
            "set_params for context)."
        )

    # -------------------------------------------------------------------------
    # Hook for subclasses that expose a *prior* (no equality conditioning).
    # The base class always reduces to the conditional kernel k_E; the
    # PriorKernelAdapter below short-circuits the Schur correction so that
    # k_E = k.  Subclasses overriding Ki0 / Kij call _partial_along and
    # _format_for_pair, which are inherited unchanged.
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Internal: build PartialDerivative for axis i.
    # -------------------------------------------------------------------------
    def _partial_along(self, i: int):
        """Return a first-order differential operator along axis i.

        For ``input_shape == ()`` (scalar input) this is the univariate
        ``PartialDerivative(MultiIndex(1))``, which the linpde-gp registry
        converts to a closed-form ``DirectionalDerivative(1.0)`` via
        ``_partial_derivative_fallback`` (the scalar branch).

        For ``input_shape == (d,)`` (vector input) we return
        ``DirectionalDerivative(e_i)`` with ``e_i`` the i-th unit vector,
        rather than ``PartialDerivative(MultiIndex(e_i))``. The reason is a
        gap in upstream dispatch: in *multi-D*,
        ``PartialDerivative.__call__`` on an ``ExpQuad`` routes to
        ``_partial_derivative_fallback``, which only synthesises a closed-form
        operator when ``prod(input_domain_shape) == 1`` (scalar). For
        ``d > 1`` it returns ``NotImplemented``, so the generic Ki0/Kij block
        path would break. A first-order partial derivative along axis ``i`` is
        exactly the directional derivative along ``e_i``, and the
        ``DirectionalDerivative`` handlers for ExpQuad / half-integer Matern
        are dimension-generic and closed-form
        (``ExpQuad_Identity_DirectionalDerivative`` and its compositions).
        This keeps the entire derivative chain inside registered closed-form
        handlers in 2D+, with zero upstream edits. The sign convention matches
        the scalar path (and is guarded by the Ki0/Kij finite-difference
        tests).
        """
        # Lazy import: this module can be inspected without a working
        # linpde-gp install (e.g. for unit testing the shape adapter against a
        # mock cov).
        from linpde_gp.linfuncops.diffops import (
            DirectionalDerivative,
            PartialDerivative,
        )
        from linpde_gp.linfuncops.diffops._coefficients import MultiIndex

        if self._scalar_input:
            assert i == 0, f"scalar-input GP must use i=0, got {i}"
            return PartialDerivative(MultiIndex(1))
        e_i = np.zeros((self._d,), dtype=float)
        e_i[i] = 1.0
        return DirectionalDerivative(e_i)

    def _format_evaluation_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Reshape stored X_E for use as kernel arg-0 vs arg-1.

        Used by the evaluation-only fast path. ``self._X_E_total`` is the
        concatenation of every ``_EvaluationFunctional.X``; this method
        reshapes it for probnum's broadcasting convention.

        Returns
        -------
        XE_arg0
            X_E reshaped to broadcast as the *first* kernel argument: shape
            ``(n_E, 1)`` for scalar input, ``(n_E, 1, d)`` for vector input.
        XE_arg1
            X_E reshaped to broadcast as the *second* kernel argument: shape
            ``(1, n_E)`` for scalar input, ``(1, n_E, d)`` for vector input.
        """
        if self._X_E_total is None:
            raise RuntimeError(
                "_format_evaluation_points called outside the eval-only fast "
                "path; this is a bug in the adapter."
            )
        XE = self._X_E_total
        n_E = XE.shape[0]
        if self._scalar_input:
            # X_E shape: (n_E,) (scalar input prior)
            return XE.reshape(n_E, 1), XE.reshape(1, n_E)
        # X_E shape: (n_E, d)
        return XE[:, None, :], XE[None, :, :]

    # -------------------------------------------------------------------------
    # Per-L block assembly for kLa and ∂_{x_i} kLa.
    #
    # Used in the generic (non-eval-only) path. The cross-cov object
    # ``L_l(prior_cov, argnum=1)`` is a ``ProcessVectorCrossCovariance`` whose
    # ``_evaluate(x)`` returns ``L_l(k(x, ·))`` — shape
    # ``batch_shape + (output_size(L_l),)``. Concatenating across l along the
    # last axis gives the row vector ``kLa(x)`` of length ``n_E_total``.
    #
    # For the differentiated kLa, we differentiate the *prior* kernel first
    # (a clean closed-form path for ExpQuad/Matern) and only then apply L_l
    # via dispatch. This sidesteps the missing
    # ``LinearFunctionOperator(L2Projection_crosscov)`` handler in upstream
    # linpde-gp and keeps the entire computation inside dispatched routes
    # that do exist (PartialDerivative on ExpQuad → ExpQuad_*_*; then L_l on
    # that via the generic L2-projection / evaluation handlers).
    # -------------------------------------------------------------------------
    def _input_view(self, X: np.ndarray) -> np.ndarray:
        """Convert gp_constr-style ``(n, d)`` input to linpde-gp's input view.

        Scalar-input GPs want ``(n,)``; vector-input GPs want ``(n, d)``.
        """
        X = np.asarray(X)
        if self._scalar_input:
            return X[:, 0]
        return X

    def _eval_kLa(self, X: np.ndarray) -> np.ndarray:
        """Evaluate concatenated ``kLa(X)`` — shape ``(n, n_E_total)``.

        Delegates to ``posterior_gp._kLas`` (a
        ``PriorPredictiveCrossCovariance``) which does the per-L
        concatenation internally. Output shape is
        ``batch_shape + (n_E_total,)`` for a scalar-output GP.
        """
        Xin = self._input_view(X)
        kLas_eval = self._gp._kLas(Xin)  # pylint: disable=protected-access
        return np.asarray(kLas_eval).reshape(X.shape[0], -1)

    def _eval_dki_kLa(self, X: np.ndarray, i: int) -> np.ndarray:
        """Evaluate ``(∂/∂x_i kLa)(X)`` — shape ``(n, n_E_total)``.

        Built per L_l: differentiate the prior kernel along axis i (argnum=0)
        to get ``∂_{x_i} k(x, t)``, then apply L_l on argnum=1 to obtain
        ``L_l_t(∂_{x_i} k(x, t))`` — a cross-covariance in x. Evaluating at X
        and concatenating the L blocks yields the differentiated row.
        """
        Di = self._partial_along(int(i))
        Di_prior = Di(self._prior_cov, argnum=0)

        Xin = self._input_view(X)
        n = X.shape[0]

        blocks = []
        for L, sz in zip(self._Ls, self._block_sizes):
            crosscov_di_l = L(Di_prior, argnum=1)
            block = np.asarray(crosscov_di_l(Xin))
            # block shape may be batch + L.output_shape; flatten L axes:
            blocks.append(block.reshape(n, sz))
        if not blocks:
            return np.zeros((n, 0))
        return np.concatenate(blocks, axis=1)


# -----------------------------------------------------------------------------
# Prior adapter — gp_constr kernel API on a PRIOR GP (no equality conditioning)
# -----------------------------------------------------------------------------
class PriorKernelAdapter(LinPDEKernelAdapter):
    r"""Expose a *prior* GP under gp_constr's kernel API (no equality side).

    The base :class:`LinPDEKernelAdapter` wraps a ``ConditionalGaussianProcess``
    ``u | E`` and exposes the conditional kernel

        k_E(x, x') = k(x, x') - kLa(x)^T G^{-1} kLa(x').

    For the *inequality-only* ablation arm there is no equality information at
    all: we truncate the prior ``u ~ GP(0, k)`` directly by the shape
    constraints ``C``. The conditional kernel then collapses to the prior
    kernel itself (the representer correction term is empty), so this adapter
    overrides :meth:`Ki0` / :meth:`Kij` to differentiate only the prior
    covariance, with no Schur correction. ``K`` and ``K_diag`` are inherited
    unchanged — they already delegate to ``self._gp.cov``, which for a prior GP
    *is* ``k``.

    This keeps the inequality-only arm on exactly the same numerical path as the
    other arms (same ``ConstrainedLinPDEPosterior`` / gp_constr machinery, same
    closed-form ``DirectionalDerivative`` handlers), isolating the single
    difference that matters for the ablation: the absence of the PDE/BC
    equality information.

    Parameters
    ----------
    prior_gp
        A ``pn.randprocs.GaussianProcess`` with ``input_shape`` ``()`` or
        ``(d,)`` and a covariance whose first/second derivatives dispatch to a
        closed form (ExpQuad in any dimension; half-integer Matern with
        sufficient smoothness). Its mean must be evaluable on numpy input
        (``pn.functions.Zero`` is fine and is the intended use).

    Notes
    -----
    Like the base adapter this is read-only and raises
    :class:`NoHyperparamOptimization` from ``set_params`` / ``get_params`` /
    ``K_gradients`` (inherited), so the two-stage freeze contract is preserved.
    """

    def __init__(self, prior_gp: Any) -> None:
        in_shape = prior_gp.cov.input_shape
        if in_shape == ():
            self._d = 1
            self._scalar_input = True
        elif len(in_shape) == 1:
            self._d = int(in_shape[0])
            self._scalar_input = False
        else:
            raise NotImplementedError(
                "PriorKernelAdapter only supports input_shape == () or (d,); "
                f"got {in_shape}."
            )

        # ``self._gp`` is the prior GP; ``K`` (inherited) reads ``self._gp.cov``.
        self._gp = prior_gp
        self._prior_cov = prior_gp.cov

        # No equality functionals: there is nothing to set up and no
        # representer correction. We populate the same private attributes the
        # base class would, so any inherited code that inspects them is happy.
        self._Ls = []
        self._block_sizes = []
        self._eval_only = False
        self._X_E_total = None
        self._gram = None
        self._derivative_setup_done = True

    def _ensure_derivative_setup(self) -> None:
        # No equality side — nothing to assemble.
        return

    def Ki0(self, X1: np.ndarray, X2: np.ndarray, i: int) -> np.ndarray:
        """(∂/∂x_i) k(x, x') | x=X1[a], x'=X2[b] — shape (n1, n2).

        Prior kernel only; the Schur correction term of the base class is
        absent (no equality information).
        """
        if not (0 <= int(i) < self._d):
            raise IndexError(f"i must be in [0, {self._d}); got {i}.")
        Di = self._partial_along(int(i))
        d_prior = Di(self._prior_cov, argnum=0)  # ∂_{x_i} k(·, ·)
        x0, x1 = self._format_for_pair(X1, X2)
        return np.asarray(d_prior(x0, x1))

    def Kij(self, X1: np.ndarray, X2: np.ndarray, i: int, j: int) -> np.ndarray:
        """(∂/∂x_i)(∂/∂x'_j) k(x, x') | x=X1[a], x'=X2[b] — shape (n1, n2).

        Prior kernel only; no representer correction.
        """
        if not (0 <= int(i) < self._d):
            raise IndexError(f"i must be in [0, {self._d}); got {i}.")
        if not (0 <= int(j) < self._d):
            raise IndexError(f"j must be in [0, {self._d}); got {j}.")
        Di = self._partial_along(int(i))
        Dj = self._partial_along(int(j))
        d_prior_mixed = Di(Dj(self._prior_cov, argnum=1), argnum=0)
        x0, x1 = self._format_for_pair(X1, X2)
        return np.asarray(d_prior_mixed(x0, x1))
