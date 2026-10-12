"""
ConstrainedLinPDEPosterior — wrapper that conditions a linpde-gp posterior
(equality-side) on inequality constraints via gp_constr.

Mathematical setup
------------------
Let u ~ GP(m, k) be a prior over a single latent process and let

    E := finite collection of equality information (data, BCs, weak-form PDE
         residuals, ...)
    C := finite collection of inequality information of the form
         a_l <= L_{ineq,l} u(X_v^{(l)}) <= b_l   (l = 1, ..., L)

We exploit the sequential-conditioning identity

    (u | E) | C  =  u | (E, C).

After equality conditioning, u | E ~ GP(m_E, k_E) is Gaussian. The
truncation against the polyhedron ⋃_l { a_l ≤ L_{ineq,l} v(X_v^{(l)}) ≤ b_l }
breaks Gaussianity; this wrapper builds the constrained posterior u | (E, C)
by feeding (k_E, m_E) into Agrell's gp_constr machinery.

Mean-subtraction trick
----------------------
gp_constr's GPmodel only accepts a constant scalar ``mean``. To support the
spatially-varying posterior mean m_E without modifying upstream code, we
center the process: pass ``mean = 0`` to gp_constr and fold the mean into the
inequality bounds via

    a' := a - L_ineq m_E(X_v),    b' := b - L_ineq m_E(X_v).

By linearity of L_ineq, the event {a ≤ L_ineq u ≤ b} is the same as
{a' ≤ L_ineq (u - m_E) ≤ b'}. After gp_constr returns a sample of the
centered process, we add m_E(X) back at output time.

Equality noise is NOT re-applied. The kernel adapter exposes k_E directly,
which has already absorbed the equality observation noise (sigma_E^2) inside
the gram matrix. We therefore pass ``likelihood = 0`` to gp_constr.

Empty equality conditioning slot
--------------------------------
gp_constr always conditions on training data Y at points X. We have already
conditioned on E at the linpde-gp side, so we want gp_constr to perform
*only* the inequality conditioning. We achieve this by passing
``X_training = empty((0, d))``, ``Y_training = empty(0)``: with zero
equality observations, gp_constr's intermediate quantities (v_1, A_1, ...)
that depend on training data collapse cleanly to zero / empty, and the
constrained-posterior expression reduces to inequality conditioning of the
base GP defined by the kernel adapter (which IS u | E). See the docstring of
:class:`_EmptyEqualityGPmodel` for the algebra.

Wrapper boundary
----------------
This module contains *no* re-derivation of the constrained-posterior moments
or sampler — those live in gp_constr. The wrapper's responsibilities are
strictly:
  (i)   build the kernel adapter,
  (ii)  evaluate L_ineq m_E at virtual points and transform bounds,
  (iii) populate gp_constr.GPmodel with our adapter, our (transformed)
        constraints, and the empty equality slot,
  (iv)  re-add m_E to gp_constr's outputs.

It also exposes :attr:`gp_constr_model` for advanced use, but explicitly
forbids calling :meth:`optimize` on it: the kernel adapter raises
:class:`NoHyperparamOptimization` from set_params/get_params/K_gradients,
making any such attempt loud.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .kernel_adapter import LinPDEKernelAdapter
from .mean_handling import transform_bounds_for_zero_mean


# -----------------------------------------------------------------------------
# Workaround for a numpy 2.x × np.matrix incompatibility surfaced in gp_constr.
# -----------------------------------------------------------------------------
# gp_constr stores the sample tensor ``fs_sim`` as ``np.matrix`` (its codebase
# was written when matrix was the idiomatic 2-D container). In numpy 2.x,
# ``np.percentile`` internally does ``diff * t`` on a sliced view of the input
# array; if the input is ``np.matrix`` the slice is also ``np.matrix``, whose
# ``__mul__`` is matrix multiplication, not elementwise. The shapes
# disagree and a ``ValueError: shapes (k, n) and (k, 1) not aligned`` is
# raised. This happens at gp_constr/GPConstr/model.py:380 inside
# ``calc_posterior_constrained``.
#
# Wrapper-side workaround: monkey-patch ``np.percentile`` for the duration of
# the gp_constr call, coercing ``np.matrix`` inputs to plain ``np.ndarray``.
# Restored on exit. Scoped to one method invocation, no global side effect.
@contextlib.contextmanager
def _matrix_safe_numpy_percentile():
    """Context manager: ``np.percentile`` accepts ``np.matrix`` for the duration.

    Inside the block, ``np.percentile(a, q, ...)`` first calls
    ``np.asarray(a)`` if ``a`` is an ``np.matrix``; outside the block,
    behavior is unchanged. Idempotent under nesting.
    """
    original = np.percentile

    def _patched(a, q, *args, **kwargs):
        if isinstance(a, np.matrix):
            a = np.asarray(a)
        return original(a, q, *args, **kwargs)

    np.percentile = _patched
    try:
        yield
    finally:
        np.percentile = original


def _seed_r_rng(seed: int = 0) -> None:
    """Pin R's RNG before delegating to gp_constr's R-backed routines.

    tmvtnorm::mtmvnorm and mvtnorm::pmvnorm (GenzBretz) use *randomized*
    quasi-Monte Carlo, and rtmvnorm's samplers consume R's RNG stream, so
    without a fixed seed repeated calls differ at the quadrature tolerance
    (~0.5% observed run-to-run). Seeding wrapper-side keeps gp_constr
    untouched and makes every moment-based summary bitwise reproducible.
    No-op when rpy2/R is unavailable; the delegated call raises its own
    informative error in that case.
    """
    try:
        import rpy2.robjects as ro  # noqa: PLC0415 (lazy: R optional)

        ro.r(f"set.seed({int(seed)})")
    except Exception:  # noqa: BLE001
        pass


if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike

    import linpde_gp


# -----------------------------------------------------------------------------
# Public dataclass for inequality constraints
# -----------------------------------------------------------------------------


@dataclass
class InequalityConstraint:
    """One inequality constraint a ≤ L_ineq u(X_v) ≤ b at virtual points X_v.

    Attributes
    ----------
    operator
        A linpde-gp ``LinearFunctionOperator``. Two cases are supported:

        * ``Identity`` (``order == 0``) — a boundedness constraint
          ``a ≤ u(X_v) ≤ b``. Mapped to gp_constr's ``constr_bounded``.
        * ``PartialDerivative`` of order 1 along a single axis (including
          the univariate ``Derivative(order=1)``) — a derivative
          constraint ``a ≤ ∂u/∂x_i (X_v) ≤ b``. Mapped to gp_constr's
          ``constr_deriv[i]``.

        Higher-order or mixed derivative constraints, integral functionals,
        and divergence/Laplacian/grad operators are not currently supported
        because gp_constr's per-axis ``∂/∂x_i`` constraint slots do not
        admit them. Extension would require a richer constraint API on the
        gp_constr side.
    Xv
        Virtual points. Shape ``(m, d)`` where ``d`` matches the GP's
        ``input_shape`` (1 for scalar input).
    bounds
        Tuple ``(a, b)`` of lower / upper bounds. Each is broadcastable to
        shape ``(m,)``. Use ``np.inf`` / ``-np.inf`` for one-sided
        constraints.
    """

    operator: Any  # linpde_gp.linfuncops.LinearFunctionOperator
    Xv: np.ndarray
    bounds: tuple["ArrayLike", "ArrayLike"]

    # -- Internal: classification of the operator --------------------------

    def kind_and_axis(self, dim: int) -> tuple[str, int]:
        """Classify the operator as ('identity', 0) or ('partial', i).

        Raises
        ------
        NotImplementedError
            If the operator is neither identity nor a first-order
            single-axis partial derivative.
        """
        # Lazy imports — keep this module importable without linpde-gp.
        from linpde_gp.linfuncops.diffops import PartialDerivative

        op = self.operator
        if isinstance(op, PartialDerivative):
            if op.order == 0:
                return ("identity", 0)
            if op.order == 1 and not op.is_mixed:
                # Find the single nonzero axis. For scalar input, multi_index.array
                # is a 0-d array containing the order; the "axis" is 0.
                arr = np.asarray(op.multi_index.array)
                if arr.ndim == 0:
                    # Univariate derivative; axis is 0 by convention.
                    if dim != 1:
                        raise NotImplementedError(
                            "Univariate Derivative used with vector-input GP "
                            f"(dim={dim}); use PartialDerivative(MultiIndex(e_i))."
                        )
                    return ("partial", 0)
                axes = np.flatnonzero(arr)
                if axes.size != 1:
                    raise NotImplementedError(
                        f"Expected single-axis first-order partial; got "
                        f"multi-index {arr}."
                    )
                return ("partial", int(axes[0]))
        raise NotImplementedError(
            f"InequalityConstraint.operator must be a PartialDerivative of "
            f"order 0 or 1 (single axis). Got {type(op).__name__}."
        )


# -----------------------------------------------------------------------------
# Numerical PD repair for the inequality-side gram.
# -----------------------------------------------------------------------------
def _make_positive_definite(A: np.ndarray, base_jitter: float) -> np.ndarray:
    """Return a symmetric PD matrix close to ``A`` by adding the smallest
    diagonal jitter (>= ``base_jitter``) for which a Cholesky factorisation
    succeeds.

    The inequality-side gram ``B1 = Cov(L_ineq u(X_v)) + eps_v I`` becomes
    severely ill-conditioned when several derivative functionals are evaluated
    at the same (or nearby) virtual points — e.g. enforcing BOTH ``d/dx`` and
    ``d/dy`` monotonicity on one virtual grid stacks two strongly correlated
    derivative blocks. The truncated-MVN routines in gp_constr (R's mtmvnorm /
    rtmvnorm / pmvnorm) require a strictly positive-definite ``sigma`` and
    reject borderline matrices. We therefore symmetrise and raise the jitter
    geometrically until the factorisation succeeds. The added jitter has the
    same modelling interpretation as ``constr_likelihood`` (virtual-observation
    noise eps_v); it only ever grows it, never shrinks it.
    """
    A = 0.5 * (np.asarray(A) + np.asarray(A).T)
    m = A.shape[0]
    scale = float(np.mean(np.diag(A))) if m else 1.0
    scale = scale if scale > 0.0 else 1.0
    jitter = float(base_jitter)
    for _ in range(12):
        B = A + jitter * np.eye(m)
        try:
            np.linalg.cholesky(B)
            return B
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-12 * scale)
    # Last resort: eigenvalue floor.
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    w = np.clip(w, 1e-10 * scale, None)
    return (V * w) @ V.T


# -----------------------------------------------------------------------------
# Subclass of GPmodel for the empty-equality case
# -----------------------------------------------------------------------------


def _empty_safe_gpmodel(adapter: LinPDEKernelAdapter, constr_likelihood: float):
    """Build a gp_constr.GPmodel with empty training data and patched preps.

    The wrapper hands gp_constr a kernel that *already* encodes the
    linpde-gp equality posterior. To prevent gp_constr from re-conditioning,
    we pass empty (0, d) X_training and length-0 Y_training. We then patch
    the methods that LAPACK would otherwise choke on with empty matrices:

        _prep_K_w        — K_w must be a (0, 0) np.matrix.
        _prep_K_w_factor — K_w_chol = (0, 0) np.matrix; jitchol on empty
                           is unreliable across SciPy versions.
        _prep_LLY        — LLY = empty (0, 1) np.matrix.
        _prep_1          — v2 / A2 / B2: v2 is (0, n_xs); A2 is (n_xs, 0);
                           B2 is K(XS, XS) (no Y correction).
        _prep_2          — v1 = (0, m); A1 = (m, 0); B1 = L1L2T_K_xv_xv +
                           constr_likelihood * I.

    All other gp_constr internals (_prep_3, _calc_constr_mean,
    calc_posterior_constrained, calc_posterior_constrained_moments,
    constrprob_Xv) consume these via standard matrix algebra; with the
    shapes above the implicit "B * Y_centered" and "A_1 * Y_centered" terms
    are (n_xs, 0) @ (0, 1) = (n_xs, 1) of zeros, which is the correct
    contribution of an empty equality slot.
    """
    # Lazy import: keep this module importable without gp_constr available.
    from GPConstr.model import GPmodel

    d = adapter.dim

    class _EmptyEqualityGPmodel(GPmodel):
        # gp_constr stores X_training / Y_training as private dunder
        # attributes. After calling super().__init__, the property setters
        # have already been invoked with the constructor defaults; we
        # overwrite them below explicitly.

        def _prep_K_w(self, verbatim: bool = False) -> None:
            if self.K_w is None:
                self.K_w = np.matrix(np.zeros((0, 0)))

        def _prep_K_w_factor(self, verbatim: bool = False) -> None:
            if self.K_w_chol is None:
                self.K_w_chol = np.matrix(np.zeros((0, 0)))

        def _prep_LLY(self) -> None:
            if self.LLY is None:
                self.LLY = np.matrix(np.zeros((0, 1)))

        def _prep_1(self, XS: np.ndarray, verbatim: bool = False) -> None:
            n_xs = XS.shape[0]
            self.v2 = np.matrix(np.zeros((0, n_xs)))
            K_xs_xs = np.asarray(self.kernel.K(XS, XS))
            self.B2 = np.matrix(K_xs_xs)
            self.A2 = np.matrix(np.zeros((n_xs, 0)))

        def _prep_2(self, verbatim: bool = False) -> None:
            if self._p2:
                return
            # L1L2T_K_xv_xv is the cross-cov between virtual points under
            # all sub-operators; gp_constr already implements this via
            # _calc_L1L2() against self.kernel — call it directly.
            L1L2T_K_xv_xv = np.asarray(self._calc_L1L2())
            m = L1L2T_K_xv_xv.shape[0]
            self.v1 = np.matrix(np.zeros((0, m)))
            self.A1 = np.matrix(np.zeros((m, 0)))
            # B1 = Cov(L_ineq u(X_v)) + eps_v I. Stacking multiple correlated
            # derivative blocks (e.g. d/dx AND d/dy on one virtual grid) makes
            # this severely ill-conditioned; the R truncated-MVN routines need a
            # strictly PD sigma, so we adaptively raise eps_v to the smallest
            # value that Cholesky-factorises.
            self.B1 = np.matrix(
                _make_positive_definite(L1L2T_K_xv_xv, self.constr_likelihood)
            )
            self._p2 = True

    gpm = _EmptyEqualityGPmodel(
        kernel=adapter,
        likelihood=0.0,  # equality noise is already in k_E
        mean=0.0,  # m_E folded into bounds via transform_bounds_for_zero_mean
        constr_likelihood=constr_likelihood,
        verbatim=False,
    )
    # Property setters reset cached state, so we call them before building
    # any prep state. Empty (0, d) and (0,) survive _check_XY_training.
    gpm.X_training = np.empty((0, d))
    gpm.Y_training = np.empty((0,))
    return gpm


# -----------------------------------------------------------------------------
# ConstrainedLinPDEPosterior
# -----------------------------------------------------------------------------


class ConstrainedLinPDEPosterior:
    """Posterior of u | (E, C): linpde-gp equality posterior + gp_constr inequality.

    Parameters
    ----------
    posterior_gp
        A linpde-gp ``ConditionalGaussianProcess`` representing u | E.
        Hyperparameters of the underlying prior covariance must be frozen
        before this wrapper is built; the underlying kernel adapter raises
        :class:`NoHyperparamOptimization` if any caller attempts to retune.
    inequality_constraints
        A sequence of :class:`InequalityConstraint`. An empty sequence is
        permitted; with no constraints the wrapper recovers the
        unconstrained equality posterior on its outputs.
    constr_likelihood
        Slack ε_v added to the inequality-side gram matrix at virtual
        points. Both for numerical conditioning and for the modeling
        interpretation that the inequality is enforced up to a small
        Gaussian error (Agrell, JMLR 2019, §3). Default 1e-6.
    adapter
        Optional pre-built kernel adapter exposing ``posterior_gp`` under
        gp_constr's kernel API. When ``None`` (default) a
        :class:`LinPDEKernelAdapter` is built from ``posterior_gp`` — the
        standard equality-conditioned path, unchanged. Supplying an adapter is
        only needed for the *inequality-only* ablation arm, where
        ``posterior_gp`` is a bare prior GP (no equality side) and the adapter
        is a :class:`PriorKernelAdapter`; the adapter must expose the same
        ``input_shape`` / ``dim`` as ``posterior_gp`` and the mean evaluated by
        ``posterior_gp.mean`` is still folded into the inequality bounds.

    Notes
    -----
    The wrapper is read-only with respect to ``posterior_gp``: it does not
    mutate the linpde-gp object. It does build internal cached structures
    (the kernel adapter and a populated ``gp_constr.GPmodel``).
    """

    def __init__(
        self,
        posterior_gp: Any,
        inequality_constraints: Sequence[InequalityConstraint],
        constr_likelihood: float = 1e-6,
        *,
        adapter: Any | None = None,
    ) -> None:
        self._posterior_gp = posterior_gp
        self._inequality_constraints = list(inequality_constraints)
        self._constr_likelihood = float(constr_likelihood)

        self._adapter = (
            adapter if adapter is not None else LinPDEKernelAdapter(posterior_gp)
        )
        d = self._adapter.dim

        self._gp_constr_model = _empty_safe_gpmodel(
            self._adapter, self._constr_likelihood
        )

        if not self._inequality_constraints:
            # No inequality information; the constrained posterior is the
            # equality posterior itself. We leave the gp_constr model
            # without any virtual points, and posterior_mean / samples
            # short-circuit accordingly.
            return

        # Bucket: identity constraints all merge into gp_constr's single
        # constr_bounded slot; partial-derivative constraints merge per axis
        # into constr_deriv[i].
        bounded_xv: list[np.ndarray] = []
        bounded_lb: list[np.ndarray] = []
        bounded_ub: list[np.ndarray] = []
        deriv_xv: list[list[np.ndarray]] = [[] for _ in range(d)]
        deriv_lb: list[list[np.ndarray]] = [[] for _ in range(d)]
        deriv_ub: list[list[np.ndarray]] = [[] for _ in range(d)]

        for c in self._inequality_constraints:
            kind, axis = c.kind_and_axis(d)
            Xv = np.atleast_2d(np.asarray(c.Xv, dtype=float))
            if Xv.ndim != 2 or Xv.shape[1] != d:
                raise ValueError(
                    f"InequalityConstraint.Xv must have shape (m, {d}); "
                    f"got {Xv.shape}."
                )
            Lm = self._evaluate_L_mean(kind, axis, Xv)
            a, b = c.bounds
            a_t, b_t = transform_bounds_for_zero_mean(Lm, a, b)
            if kind == "identity":
                bounded_xv.append(Xv)
                bounded_lb.append(a_t)
                bounded_ub.append(b_t)
            else:  # 'partial'
                deriv_xv[axis].append(Xv)
                deriv_lb[axis].append(a_t)
                deriv_ub[axis].append(b_t)

        # Lazy import for gp_constr's Constraint class.
        from GPConstr.model import Constraint

        if bounded_xv:
            Xv_b = np.concatenate(bounded_xv, axis=0)
            LB_b = np.concatenate(bounded_lb)
            UB_b = np.concatenate(bounded_ub)
            cb = Constraint(
                LB=lambda X, _v=LB_b: _v,
                UB=lambda X, _v=UB_b: _v,
            )
            cb.Xv = Xv_b
            self._gp_constr_model.constr_bounded = cb

        # Always populate constr_deriv as a length-d list (one slot per
        # axis); slots without constraints have Xv=None. This matches
        # gp_constr's iteration convention in _calc_FiL2T / _calc_L1L2.
        if any(len(deriv_xv[i]) > 0 for i in range(d)):
            deriv_constraints: list = []
            for i in range(d):
                if deriv_xv[i]:
                    Xv_i = np.concatenate(deriv_xv[i], axis=0)
                    LB_i = np.concatenate(deriv_lb[i])
                    UB_i = np.concatenate(deriv_ub[i])
                    ci = Constraint(
                        LB=lambda X, _v=LB_i: _v,
                        UB=lambda X, _v=UB_i: _v,
                    )
                    ci.Xv = Xv_i
                else:
                    # Placeholder: no virtual points along this axis.
                    ci = Constraint(
                        LB=lambda X: -np.inf * np.ones(X.shape[0]),
                        UB=lambda X: np.inf * np.ones(X.shape[0]),
                    )
                    ci.Xv = None
                deriv_constraints.append(ci)
            self._gp_constr_model.constr_deriv = deriv_constraints

    # -------------------------------------------------------------------------
    # Internal: evaluate L_ineq m_E at virtual points
    # -------------------------------------------------------------------------
    # We deliberately do *not* dispatch ``operator(self._posterior_gp).mean(X)``
    # for derivative operators. That path applies ``operator`` to the prior's
    # mean (often ``pn.functions.Zero``) and to the cross-covariances kLas,
    # then re-assembles them via ``Mean._evaluate``. For ``operator =
    # PartialDerivative`` this falls back to ``_jax_fallback`` which JAX-traces
    # the prior mean. ``pn.functions.Zero.__call__`` does ``np.asarray(x)``,
    # which raises ``TracerArrayConversionError`` because it has no JaxFunction
    # path — a structural upstream limitation distinct from the
    # ``_evaluate_jax`` typos collected in ``docs/upstream-issues.md``.
    #
    # Instead we evaluate ``L_ineq m_E`` by central finite differences on the
    # *posterior* mean ``m_E``. ``m_E.__call__`` uses the numpy ``_evaluate``
    # path, which works. With h=1e-4 the truncation error is O(h^2) ~ 1e-8,
    # well below the constr_likelihood slack added at the inequality gram. For
    # identity (boundedness) we evaluate ``m_E`` directly — exact, no FD.
    _FD_STEP = 1e-4

    def _evaluate_L_mean(self, kind: str, axis: int, Xv: np.ndarray) -> np.ndarray:
        """Compute (L_ineq m_E)(X_v) for the supported operator classes.

        Parameters
        ----------
        kind
            Either ``'identity'`` (boundedness) or ``'partial'`` (single-axis
            first-order partial derivative).
        axis
            For ``kind == 'partial'``, the axis along which to differentiate.
            Ignored when ``kind == 'identity'``.
        Xv
            Virtual points, shape ``(m, d)``.

        Returns
        -------
        Lm
            Shape ``(m,)`` array.
        """
        if kind == "identity":
            return self._eval_m_E(Xv)
        if kind == "partial":
            h = self._FD_STEP
            Xv_p = np.array(Xv, copy=True)
            Xv_m = np.array(Xv, copy=True)
            Xv_p[:, axis] += h
            Xv_m[:, axis] -= h
            return (self._eval_m_E(Xv_p) - self._eval_m_E(Xv_m)) / (2.0 * h)
        raise NotImplementedError(f"Unsupported operator kind: {kind!r}.")

    def _reshape_for_gp(self, X: np.ndarray) -> np.ndarray:
        """Reshape gp_constr-style (n, d) input to whatever linpde-gp expects.

        For scalar-input GPs (``input_shape == ()``), linpde-gp wants a
        1-D ``(n,)`` array; for vector-input GPs (``input_shape == (d,)``)
        it wants ``(n, d)`` directly.
        """
        X = np.asarray(X)
        if self._adapter._scalar_input:  # pylint: disable=protected-access
            if X.ndim == 2 and X.shape[1] == 1:
                return X[:, 0]
            return np.asarray(X).reshape(-1)
        return X

    def _eval_m_E(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the equality-posterior mean m_E at X. Returns shape (n,)."""
        return np.asarray(self._posterior_gp.mean(self._reshape_for_gp(X))).reshape(-1)

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def posterior_mean(self, X: np.ndarray) -> np.ndarray:
        """Mean of u | (E, C) at X. Shape (n,).

        If no inequality constraints are present, returns the equality
        posterior mean m_E(X) directly. Otherwise calls gp_constr's
        moment-based computation and adds m_E back.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or X.shape[1] != self._adapter.dim:
            raise ValueError(
                f"X must have shape (n, {self._adapter.dim}); got {X.shape}."
            )

        m_E = self._eval_m_E(X)
        if not self._has_active_constraints():
            return m_E

        # gp_constr returns mean as an (n_xs, 1) np.matrix containing the
        # *centered* posterior mean. Add m_E back at output time.
        _seed_r_rng()
        mean_centered, _ = self._gp_constr_model.calc_posterior_constrained_moments(X)
        mean_centered = np.asarray(mean_centered).reshape(-1)
        return m_E + mean_centered

    def posterior_moments(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and marginal standard deviation of u | (E, C) at X.

        Both are read from truncated-Gaussian posterior moments (gp_constr's
        ``calc_posterior_constrained_moments``). The seeded randomized
        quasi-Monte Carlo call makes the result reproducible but leaves
        numerical-integration error. A 1.96-sigma band built from the returned
        standard deviation is a Gaussian-equivalent dispersion summary, not a
        95% credible interval for the non-Gaussian constrained posterior. This
        route avoids the second (truncated-MVN sampling) R call.

        Returns
        -------
        mean, std : np.ndarray
            Posterior mean and marginal standard deviation, each shape (n,).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or X.shape[1] != self._adapter.dim:
            raise ValueError(
                f"X must have shape (n, {self._adapter.dim}); got {X.shape}."
            )

        m_E = self._eval_m_E(X)
        if not self._has_active_constraints():
            # No constraints: fall back to the equality-posterior covariance.
            cov = np.asarray(self._adapter.K(X, X))
            std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            return m_E, std

        _seed_r_rng()
        mean_centered, cov = self._gp_constr_model.calc_posterior_constrained_moments(X)
        mean = m_E + np.asarray(mean_centered).reshape(-1)
        std = np.sqrt(np.clip(np.diag(np.asarray(cov)), 0.0, None))
        return mean, std

    def posterior_std(self, X: np.ndarray) -> np.ndarray:
        """Marginal standard deviation of u | (E, C) at X (reproducible).

        Convenience wrapper around :meth:`posterior_moments`; see there for the
        reproducibility rationale.
        """
        return self.posterior_moments(X)[1]

    def posterior_samples(
        self,
        X: np.ndarray,
        n_samples: int,
        algorithm: str = "minimax_tilting",
    ) -> np.ndarray:
        """Samples from u | (E, C) at X. Shape (n_samples, n).

        Parameters
        ----------
        X
            Test points, shape (n, d).
        n_samples
            Number of samples to draw.
        algorithm
            Truncated-multivariate-normal sampler used by gp_constr for the
            constraint distribution. One of ``'rejection'``, ``'gibbs'``,
            ``'minimax_tilting'`` (default; see Agrell, §3.2).

        Notes
        -----
        With ``inequality_constraints=[]`` this falls back to drawing from
        the unconstrained equality posterior via the underlying linpde-gp
        ``ConditionalGaussianProcess``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or X.shape[1] != self._adapter.dim:
            raise ValueError(
                f"X must have shape (n, {self._adapter.dim}); got {X.shape}."
            )

        m_E = self._eval_m_E(X)

        if not self._has_active_constraints():
            # Fallback: sample from the linpde-gp posterior directly. We
            # form the mean and full covariance via the kernel adapter and
            # draw from the resulting multivariate normal.
            cov = self._adapter.K(X, X)
            cov = 0.5 * (cov + cov.T)
            rng = np.random.default_rng()
            samples = rng.multivariate_normal(
                mean=m_E,
                cov=cov,
                size=n_samples,
                check_valid="ignore",
            )
            return samples

        # gp_constr: save_samples must be < num_samples but we want all
        # samples. We oversample by 1 and request all.
        save = max(1, n_samples)
        num = save + 1
        _seed_r_rng()
        with _matrix_safe_numpy_percentile():
            _, _, _, _, samples_mat, _ = (
                self._gp_constr_model.calc_posterior_constrained(
                    X,
                    num_samples=num,
                    save_samples=save,
                    algorithm=algorithm,
                )
            )
        # samples_mat is (n_xs, save) np.matrix. Take the first n_samples
        # columns and transpose to (n_samples, n_xs).
        samples = np.asarray(samples_mat)[:, :n_samples].T
        return samples + m_E[None, :]

    @property
    def constraint_satisfaction_probability(self) -> float:
        """P(C | E) — probability the unconstrained posterior u | E satisfies C.

        Diagnostic only: low values
        flag either (a) the prior fights the inequality, or (b) the
        equality information itself is incompatible with C. The wrapper
        cannot distinguish the two.

        With no constraints, the satisfaction probability is 1 by convention.

        Notes
        -----
        Computed by gp_constr's ``constrprob_Xv(posterior=True)``, which
        delegates to R's ``pmvnorm``. Requires rpy2 + R packages
        ``mvtnorm``, ``tmvtnorm``, ``TruncatedNormal``, ``truncnorm``.
        """
        if not self._has_active_constraints():
            return 1.0
        _seed_r_rng()
        prob = self._gp_constr_model.constrprob_Xv(posterior=True)
        return float(np.asarray(prob).reshape(()))

    @property
    def gp_constr_model(self) -> Any:
        """Underlying gp_constr.GPmodel — direct access for advanced use.

        WARNING: do NOT call ``.optimize()`` on this. Calling optimize()
        invokes the kernel's set_params/get_params/K_gradients, all of
        which raise :class:`~linpde_gp_constrained.NoHyperparamOptimization`
        in the kernel adapter. The two-stage integration freezes
        equality-side hyperparameters before this wrapper is built;
        re-tuning them here would silently break the framework.
        """
        return self._gp_constr_model

    # -------------------------------------------------------------------------
    # Internal: are there any virtual points to condition on?
    # -------------------------------------------------------------------------
    def _has_active_constraints(self) -> bool:
        gpm = self._gp_constr_model
        if gpm.constr_bounded is not None and gpm.constr_bounded.Xv is not None:
            return True
        if gpm.constr_deriv is not None:
            for c in gpm.constr_deriv:
                if c.Xv is not None:
                    return True
        return False
