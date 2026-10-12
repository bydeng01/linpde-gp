"""
Build a linpde-gp equality posterior whose equality side carries a *weak-form*
PDE residual (PDE projected onto a finite test basis) together with Dirichlet
boundary conditions at points.

Why this lives in the wrapper
-----------------------------
linpde-gp's ``condition_on_observations`` accepts a ``LinearFunctional`` ``L``
and assembles the gram via ``Lf = L(prior); gram = Lf.cov``. For
``L = l2_proj @ Laplacian`` this requires applying the Laplacian to a
``CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis`` cross-
covariance — a dispatch that does not exist upstream. (The standard pointwise
PDE conditioning ``L = bvp.pde.diffop`` followed by ``X=X_pde`` works because
it composes Laplacian with a *point* evaluation, for which the
``LinearFunctionOperator.__call__.register(CovarianceFunction_Identity_Evaluation)``
handler is provided.) Adding the missing handler upstream would be the
"proper" fix; the wrapper instead assembles the equality posterior directly,
using only dispatch paths that already exist:

  * ``Laplacian × Laplacian × ExpQuad`` is a registered handler chain that
    yields ``ExpQuad_WeightedLaplacian_WeightedLaplacian``.
  * The L2-projection cross-covariance handler (registered for any
    ``CovarianceFunction``) does the single-quad and double-quad against the
    hat basis.
  * ``_EvaluationFunctional`` cross-cov handlers do the BC blocks.

We then construct a ``ConditionalGaussianProcess`` directly via its
``__init__``, supplying the assembled gram, ``PriorPredictiveCrossCovariance``,
and pre-computed representer weights. This bypasses the missing handler
without modifying upstream linpde-gp.

Mathematical content
--------------------
Given a prior ``u ~ GP(0, k)`` on ``[0, 1]``, a univariate hat basis
``{φ_i}_{i=1}^m`` with grid endpoints at the domain boundary, and a Dirichlet
BVP

    -u''(x) = f(x),   x in (0, 1),    u(0) = a,  u(1) = b,

the equality information is

    L_pde u = ( ∫ φ_i(x) (-u''(x)) dx )_{i=1}^m  =  Y_pde + ε_pde,
    L_bc  u = ( u(0), u(1) )                     =  (a, b) + ε_bc,

where ``Y_pde[i] = ∫ φ_i(x) f(x) dx`` is the L2-projection of the source onto
the hat basis (computed once via numerical quadrature), and the noise terms
``ε_pde, ε_bc`` model both the discretisation error and the small jitter we
add for numerical conditioning.

By linearity, ``L_pde = -P_h ∘ Δ`` where ``P_h`` is the L2-projection onto
``{φ_i}``. We therefore implement ``L_pde`` as the linpde-gp composite
``-1.0 * (l2_proj @ WeightedLaplacian(1.0))`` (univariate Laplacian = -d²/dx²
sign flipped, see :func:`make_pde_equality_posterior`).

Conditional posterior identity
------------------------------
``u | (Y_pde, Y_bc) ~ GP(m_E, k_E)``, with the standard Pförtner-Theorem-1
expression for ``m_E`` and ``k_E`` in terms of the assembled
``kLa(x) = (L_pde(k(x, ·)), L_bc(k(x, ·)))`` and the block gram. The
:class:`ConditionalGaussianProcess` we construct has the same surface API as
one returned by the standard ``condition_on_observations`` chain — the
:class:`LinPDEKernelAdapter` and :class:`ConstrainedLinPDEPosterior`
downstream consume it without code changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import probnum as pn
import scipy.integrate


def _l2_project_function(f, basis, normalized: bool = True) -> np.ndarray:
    """L2-project a callable f onto a univariate hat basis.

    Returns the vector ``(M_h^{-1} g)`` where ``g[i] = ∫ φ_i(x) f(x) dx`` and
    ``M_h`` is the basis mass matrix (omitted if ``normalized=False``).
    Mirrors ``L2Projection_UnivariateLinearInterpolationBasis.__call__`` for a
    plain ``Function``.

    Parameters
    ----------
    f
        Callable ``x -> f(x)`` returning a scalar.
    basis
        ``UnivariateLinearInterpolationBasis``.
    normalized
        If True (default), apply the inverse mass matrix at the end so the
        result is the L2-orthogonal projection coefficients. If False, return
        the raw integrals ``g[i]`` (useful when ``L_pde`` is the unnormalised
        weak form).
    """
    g = np.array(
        [
            scipy.integrate.quad(
                lambda x, _idx=idx: basis.eval_elem(_idx, x) * f(x),
                *basis.support_bounds(idx),
            )[0]
            for idx in range(len(basis))
        ]
    )
    if not normalized:
        return g

    proj_normalizer = basis.l2_projection(normalized=True).normalizer
    return np.asarray(proj_normalizer(g, axis=-1))


def make_pde_equality_posterior(
    prior: Any,
    test_basis: Any,
    f: Any,
    *,
    pde_op: Any | None = None,
    pde_sign: float = -1.0,
    sigma_pde: float = 0.0,
    X_bc: np.ndarray | None = None,
    Y_bc: np.ndarray | None = None,
    sigma_bc: float = 0.0,
    pde_jitter: float = 1e-10,
    bc_jitter: float = 1e-12,
):
    """Build the equality posterior u | (weak-form PDE residual, BCs).

    Parameters
    ----------
    prior
        ``pn.randprocs.GaussianProcess`` with scalar input shape ``()``,
        zero (or ``Constant``) mean, and a covariance for which the dispatch
        chain ``WeightedLaplacian(WeightedLaplacian(prior.cov, argnum=1),
        argnum=0)`` resolves to a closed-form covariance object (ExpQuad and
        half-integer Matern with sufficient smoothness do).
    test_basis
        ``UnivariateLinearInterpolationBasis`` covering ``[0, 1]``. Used both
        for the L2-projection of the residual and for the test functions
        ``φ_i`` that define ``L_pde``.
    f
        Callable ``x -> f(x)`` for the right-hand side of ``-u'' = f``.
    pde_op
        The differential operator. Defaults to ``WeightedLaplacian([1.0])``
        (univariate Laplacian = ``d²/dx²``).
    pde_sign
        Sign convention multiplier on the operator. Default ``-1.0``
        corresponds to ``-u''`` on the left-hand side; combined with
        ``Y_pde = +∫ φ_i f dx`` this enforces ``-u'' = f`` weakly.
    sigma_pde
        Standard deviation of additive noise on each PDE residual mode (the
        L2-projection of ``r_i = (f - (-u'')) onto φ_i``). Use ``0.0`` for
        noiseless weak conditioning; any small positive value relaxes the
        residual to "approximately satisfied". A small jitter is added on top
        for numerical conditioning regardless.
    X_bc, Y_bc, sigma_bc
        Boundary points, observed values, and observation noise std.
        ``X_bc = None`` skips the BC block (then the equality side is
        weak-form PDE only).
    pde_jitter, bc_jitter
        Diagonal jitter added inside the gram for each block. Independent of
        ``sigma_pde``/``sigma_bc`` so noiseless conditioning still
        Cholesky-factorises.

    Returns
    -------
    posterior
        A ``linpde_gp.randprocs.ConditionalGaussianProcess`` whose ``cov``,
        ``mean``, ``gram``, ``_kLas``, and ``_Ls`` are populated to be
        consumed by :class:`LinPDEKernelAdapter` /
        :class:`ConstrainedLinPDEPosterior` without modification.

    Notes
    -----
    The returned object's ``_Ls`` are
        (L_pde, L_bc)        if X_bc is provided,
        (L_pde,)             otherwise,
    where ``L_pde`` is a ``CompositeLinearFunctional`` (l2_proj @ pde_op,
    optionally scaled by ``pde_sign``) and ``L_bc`` is an
    ``_EvaluationFunctional``. Downstream code that only inspects ``_Ls``
    (such as the kernel adapter's ``_eval_only`` branch) will correctly
    classify ``L_bc`` as eval but will fall back to the per-L block path
    because ``L_pde`` is non-evaluation.
    """
    # Lazy imports — keep the wrapper module importable without linpde-gp.
    from linpde_gp.linfuncops.diffops import WeightedLaplacian
    from linpde_gp.linfunctls._evaluation import _EvaluationFunctional
    from linpde_gp.linfunctls.projections.l2 import (
        L2Projection_UnivariateLinearInterpolationBasis,
    )
    from linpde_gp.randprocs._gaussian_process._conditional import (
        ConditionalGaussianProcess,
    )

    if prior.input_shape != ():
        raise NotImplementedError(
            "make_pde_equality_posterior currently supports scalar-input GPs "
            "(input_shape == ()) only."
        )

    if pde_op is None:
        pde_op = WeightedLaplacian(np.array(1.0))

    # ---------------------------------------------------------------------
    # 1. The functionals.
    # ---------------------------------------------------------------------
    l2_proj = L2Projection_UnivariateLinearInterpolationBasis(
        test_basis, normalized=True
    )
    L_pde_unsigned = l2_proj @ pde_op  # CompositeLinearFunctional
    L_pde = pde_sign * L_pde_unsigned  # ScaledLinearFunctional

    has_bc = X_bc is not None
    if has_bc:
        X_bc = np.asarray(X_bc, dtype=float)
        Y_bc = np.asarray(Y_bc, dtype=float)
        if X_bc.shape != Y_bc.shape:
            raise ValueError(f"X_bc {X_bc.shape} and Y_bc {Y_bc.shape} must match.")
        L_bc = _EvaluationFunctional(
            input_domain_shape=prior.input_shape,
            input_codomain_shape=prior.output_shape,
            X=X_bc,
        )

    # ---------------------------------------------------------------------
    # 2. Per-L cross-covariances on the prior. These go through dispatched
    #    handlers only:
    #      WeightedLaplacian(ExpQuad, argnum=k)  registered;
    #      L2Projection(any CovarianceFunction, argnum=k)  registered (generic).
    # ---------------------------------------------------------------------
    # k_t-Laplacian: Δ_t k(x, t)  (scalar-input → reverse=False from argnum=1)
    k_op_t = pde_op(prior.cov, argnum=1)
    # kLa_pde(x) = ∫ φ_i(t) [pde_sign * Δ_t k(x, t)] dt
    kLa_pde_unsigned = l2_proj(k_op_t, argnum=1)
    if pde_sign != 1.0:
        # Scale the cross-cov object by pde_sign. The scalar-by-PVCrossCov
        # multiplication uses the registered ScaledProcessVectorCrossCovariance.
        kLa_pde = pde_sign * kLa_pde_unsigned
    else:
        kLa_pde = kLa_pde_unsigned

    if has_bc:
        kLa_bc = L_bc(prior.cov, argnum=1)

    # ---------------------------------------------------------------------
    # 3. Block gram. We assemble each block from existing dispatch:
    #    * G_pde_pde[i,j] = ∫∫ φ_i(s) [pde_sign² Δ_s Δ_t k(s,t)] φ_j(t) ds dt
    #      = pde_sign² * l2_proj(l2_proj(Δ_s Δ_t k, argnum=1)).
    #      Internally projections.py:75 does dblquad over the inner covfunc.
    #    * G_bc_bc = k(X_bc, X_bc).
    #    * G_pde_bc[i,j] = ∫ φ_i(s) [pde_sign * Δ_s k(s, X_bc[j])] ds
    #      via l2_proj(pde_op(prior.cov, argnum=0), argnum=0)(X_bc).
    # ---------------------------------------------------------------------
    # PDE-PDE
    k_op_s_op_t = pde_op(k_op_t, argnum=0)  # Δ_s Δ_t k(s, t)
    pdex_pdex_crosscov = l2_proj(k_op_s_op_t, argnum=1)
    G_pde_pde_cov = l2_proj(pdex_pdex_crosscov)
    G_pde_pde = (pde_sign**2) * np.asarray(G_pde_pde_cov.array)
    m = G_pde_pde.shape[0]

    if has_bc:
        # PDE-BC
        k_op_s = pde_op(prior.cov, argnum=0)  # Δ_s k(s, t)
        # Project on first arg (reverse=True). Output shape (m, batch).
        crosscov_pdex = l2_proj(k_op_s, argnum=0)
        # Evaluate at X_bc — gives (m, n_bc).
        G_pde_bc_unsigned = np.asarray(crosscov_pdex(X_bc))
        G_pde_bc = pde_sign * G_pde_bc_unsigned

        # BC-BC
        G_bc_bc = np.asarray(prior.cov(X_bc[:, None], X_bc[None, :]))
        n_bc = G_bc_bc.shape[0]
    else:
        G_pde_bc = np.zeros((m, 0))
        G_bc_bc = np.zeros((0, 0))
        n_bc = 0

    G = np.empty((m + n_bc, m + n_bc))
    G[:m, :m] = G_pde_pde + (sigma_pde**2 + pde_jitter) * np.eye(m)
    if has_bc:
        G[:m, m:] = G_pde_bc
        G[m:, :m] = G_pde_bc.T
        G[m:, m:] = G_bc_bc + (sigma_bc**2 + bc_jitter) * np.eye(n_bc)
    G = 0.5 * (G + G.T)  # symmetrise to absorb roundoff drift

    # ---------------------------------------------------------------------
    # 4. RHS of the equality system.
    # ---------------------------------------------------------------------
    Y_pde = _l2_project_function(f, test_basis, normalized=True)
    if Y_pde.shape != (m,):
        raise RuntimeError(
            f"L2 projection of f returned shape {Y_pde.shape}; expected ({m},)."
        )

    if has_bc:
        y_full = np.concatenate([Y_pde, Y_bc], axis=0)
    else:
        y_full = Y_pde

    # ---------------------------------------------------------------------
    # 5. Pre-compute representer weights bypassing L(prior.mean) dispatch.
    #
    # The standard ConditionalGaussianProcess cached path computes
    # ``L(prior.mean)`` for each L. For a composite L containing a
    # WeightedLaplacian, this hits the JAX-tracing crash on
    # ``pn.functions.Zero`` documented in docs/upstream-issues.md. We
    # therefore restrict to zero-mean priors for now and supply
    # ``representer_weights`` directly.
    # ---------------------------------------------------------------------
    if not isinstance(prior.mean, pn.functions.Zero):
        # We could compute L(prior.mean) numerically here for the BC slot
        # (eval is cheap) but L(prior.mean) for the composite hits the JAX
        # path. Until we add a numeric workaround, restrict to zero-mean.
        raise NotImplementedError(
            "make_pde_equality_posterior currently requires a zero-mean prior. "
            "Generalising to non-zero means requires a numeric L(prior.mean) "
            "evaluator that bypasses the pn.functions.Zero JAX-tracer issue."
        )

    G_linop = pn.linops.aslinop(G)
    G_linop.is_symmetric = True
    G_linop.is_positive_definite = True
    representer_weights = np.linalg.solve(G, y_full)

    # ---------------------------------------------------------------------
    # 6. Assemble the ConditionalGaussianProcess.
    # ---------------------------------------------------------------------
    if has_bc:
        Ys = (Y_pde, Y_bc)
        Ls = (L_pde, L_bc)
        bs = (
            (
                pn.randvars.Normal(np.zeros(m), (sigma_pde**2 + pde_jitter) * np.eye(m))
                if (sigma_pde > 0.0 or pde_jitter > 0.0)
                else None
            ),
            (
                pn.randvars.Normal(
                    np.zeros(n_bc), (sigma_bc**2 + bc_jitter) * np.eye(n_bc)
                )
                if (sigma_bc > 0.0 or bc_jitter > 0.0)
                else None
            ),
        )
        kLas = ConditionalGaussianProcess.PriorPredictiveCrossCovariance(
            (kLa_pde, kLa_bc)
        )
    else:
        Ys = (Y_pde,)
        Ls = (L_pde,)
        bs = (
            (
                pn.randvars.Normal(np.zeros(m), (sigma_pde**2 + pde_jitter) * np.eye(m))
                if (sigma_pde > 0.0 or pde_jitter > 0.0)
                else None
            ),
        )
        kLas = ConditionalGaussianProcess.PriorPredictiveCrossCovariance((kLa_pde,))

    posterior = ConditionalGaussianProcess(
        prior=prior,
        Ys=Ys,
        Ls=Ls,
        bs=bs,
        kLas=kLas,
        gram_matrix=G_linop,
        representer_weights=representer_weights,
    )
    return posterior


# =============================================================================
# Collocation (pointwise PDE residual) equality posterior — dimension-generic.
#
# Used for the 2D Poisson demo (see
# experiments/linpde_gp_constrained/poisson_2d_weak_form). The weak-form route
# above relies on the L2-projection cross-cov handler, which is registered only
# on the *univariate* interpolation basis. Rather than add a tensor-product hat
# basis + a new 2D cross-cov handler (which would need a 4-D quadrature for the
# PDE-PDE gram block), we enforce the PDE residual *pointwise* at collocation
# points. Every kernel-side quantity then reduces to point evaluations of the
# SAME closed-form bivariate operators used in 1D (Δ_sΔ_t k, Δ_s k, k) —
# verified dimension-generic for ExpQuad — so no new handler and no
# multidimensional quadrature are required, and the construction works for any
# input dimension.
# =============================================================================


def _format_pair(cov, A: np.ndarray, B: np.ndarray, scalar_input: bool):
    """Evaluate a covariance function at all pairs (A[a], B[b]) -> (nA, nB).

    Mirrors the broadcasting convention used by ``LinPDEKernelAdapter``:
    probnum's ``CovarianceFunction.__call__`` broadcasts over batch dims, so to
    obtain a full Gram we broadcast the batch axes outward.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if scalar_input:
        # A, B are (n,) point arrays.
        return np.asarray(cov(A[:, None], B[None, :]))
    # A, B are (n, d).
    return np.asarray(cov(A[:, None, :], B[None, :, :]))


def make_pde_collocation_posterior(
    prior: Any,
    X_pde: np.ndarray,
    f: Any,
    *,
    pde_op: Any | None = None,
    pde_sign: float = -1.0,
    sigma_pde: float = 0.0,
    X_bc: np.ndarray | None = None,
    Y_bc: np.ndarray | None = None,
    sigma_bc: float = 0.0,
    pde_jitter: float = 1e-10,
    bc_jitter: float = 1e-12,
):
    r"""Build the equality posterior ``u | (pointwise PDE residual, BCs)``.

    Strong-form (collocation) analogue of :func:`make_pde_equality_posterior`.
    The PDE is enforced at a finite set of collocation points

        L_pde u = ( (pde_sign · pde_op) u )(X_pde)  =  Y_pde + ε_pde,
        Y_pde   = f(X_pde),

    together with optional Dirichlet boundary conditions

        L_bc u = u(X_bc) = Y_bc + ε_bc.

    With ``pde_op = Laplacian`` and ``pde_sign = -1`` this enforces ``-Δu = f``
    at the collocation points. Dimension-generic: works for scalar input
    (``input_shape == ()``) and vector input (``input_shape == (d,)``).

    Parameters
    ----------
    prior
        Zero-mean ``pn.randprocs.GaussianProcess`` whose covariance admits the
        closed-form dispatch chain
        ``WeightedLaplacian(WeightedLaplacian(cov, argnum=1), argnum=0)``
        (ExpQuad in any dimension).
    X_pde
        Collocation points. Shape ``(n_pde,)`` for scalar input or
        ``(n_pde, d)`` for vector input.
    f
        Callable for the right-hand side of ``-Δu = f``. Receives points in the
        same layout as ``X_pde`` and returns a ``(n_pde,)`` array.
    pde_op
        Differential operator; defaults to the isotropic Laplacian
        ``WeightedLaplacian(ones(input_shape))``.
    pde_sign
        Sign multiplier; ``-1.0`` for ``-Δu = f``.
    sigma_pde, sigma_bc
        Additive noise std on the PDE-residual / BC blocks (``0.0`` = noiseless,
        relying on the small jitter for conditioning).
    X_bc, Y_bc
        Boundary points and values. ``X_bc=None`` drops the BC block.
    pde_jitter, bc_jitter
        Diagonal jitter for numerical conditioning, independent of the noise.

    Returns
    -------
    posterior
        A ``ConditionalGaussianProcess`` with ``_Ls = (L_pde, L_bc)`` (or
        ``(L_pde,)``), ``_kLas``, ``gram``, and ``representer_weights`` set, so
        that :class:`LinPDEKernelAdapter` / :class:`ConstrainedLinPDEPosterior`
        consume it without modification. ``L_pde`` is a (scaled)
        ``CompositeLinearFunctional`` ``δ_{X_pde} ∘ pde_op``, classified as
        non-evaluation by the adapter so the generic per-L block path is used.
    """
    from linpde_gp.linfuncops.diffops import WeightedLaplacian
    from linpde_gp.linfunctls._evaluation import _EvaluationFunctional
    from linpde_gp.randprocs._gaussian_process._conditional import (
        ConditionalGaussianProcess,
    )

    in_shape = prior.input_shape
    scalar_input = in_shape == ()
    if not (scalar_input or len(in_shape) == 1):
        raise NotImplementedError(
            "make_pde_collocation_posterior supports input_shape == () or "
            f"(d,); got {in_shape}."
        )

    if not isinstance(prior.mean, pn.functions.Zero):
        raise NotImplementedError(
            "make_pde_collocation_posterior currently requires a zero-mean "
            "prior (representer weights are assembled directly, bypassing the "
            "L(prior.mean) dispatch that crashes on pn.functions.Zero under a "
            "differential operator)."
        )

    if pde_op is None:
        if scalar_input:
            pde_op = WeightedLaplacian(np.array(1.0))
        else:
            pde_op = WeightedLaplacian(np.ones(in_shape))

    X_pde = np.asarray(X_pde, dtype=float)
    n_pde = X_pde.shape[0]

    # ---------------------------------------------------------------------
    # 1. Functionals.
    # ---------------------------------------------------------------------
    L_pde_eval = _EvaluationFunctional(
        input_domain_shape=prior.input_shape,
        input_codomain_shape=prior.output_shape,
        X=X_pde,
    )
    L_pde = pde_sign * (L_pde_eval @ pde_op)  # scaled CompositeLinearFunctional

    has_bc = X_bc is not None
    if has_bc:
        X_bc = np.asarray(X_bc, dtype=float)
        Y_bc = np.asarray(Y_bc, dtype=float)
        L_bc = _EvaluationFunctional(
            input_domain_shape=prior.input_shape,
            input_codomain_shape=prior.output_shape,
            X=X_bc,
        )

    # ---------------------------------------------------------------------
    # 2. Per-L cross-covariances on the prior (dispatched closed forms).
    #    k_op_t = Δ_t k(x, t);  kLa_pde(x) = pde_sign · (Δ_t k)(x, X_pde).
    # ---------------------------------------------------------------------
    k_op_t = pde_op(prior.cov, argnum=1)
    kLa_pde = pde_sign * L_pde_eval(k_op_t, argnum=1)
    if has_bc:
        kLa_bc = L_bc(prior.cov, argnum=1)

    # ---------------------------------------------------------------------
    # 3. Block gram by point evaluation of the closed-form operators.
    #    G_pde_pde = pde_sign² · (Δ_sΔ_t k)(X_pde, X_pde)
    #    G_pde_bc  = pde_sign  · (Δ_s k)(X_pde, X_bc)
    #    G_bc_bc   = k(X_bc, X_bc)
    # ---------------------------------------------------------------------
    k_op_s_op_t = pde_op(k_op_t, argnum=0)  # Δ_s Δ_t k  (closed form)
    G_pde_pde = (pde_sign**2) * _format_pair(k_op_s_op_t, X_pde, X_pde, scalar_input)
    m = G_pde_pde.shape[0]

    if has_bc:
        k_op_s = pde_op(prior.cov, argnum=0)  # Δ_s k
        G_pde_bc = pde_sign * _format_pair(k_op_s, X_pde, X_bc, scalar_input)
        G_bc_bc = _format_pair(prior.cov, X_bc, X_bc, scalar_input)
        n_bc = G_bc_bc.shape[0]
    else:
        G_pde_bc = np.zeros((m, 0))
        G_bc_bc = np.zeros((0, 0))
        n_bc = 0

    G = np.empty((m + n_bc, m + n_bc))
    G[:m, :m] = G_pde_pde + (sigma_pde**2 + pde_jitter) * np.eye(m)
    if has_bc:
        G[:m, m:] = G_pde_bc
        G[m:, :m] = G_pde_bc.T
        G[m:, m:] = G_bc_bc + (sigma_bc**2 + bc_jitter) * np.eye(n_bc)
    G = 0.5 * (G + G.T)

    # ---------------------------------------------------------------------
    # 4. RHS: Y_pde = f(X_pde) (collocation values of the source).
    # ---------------------------------------------------------------------
    Y_pde = np.asarray(f(X_pde), dtype=float).reshape(-1)
    if Y_pde.shape != (m,):
        raise RuntimeError(f"f(X_pde) returned shape {Y_pde.shape}; expected ({m},).")
    y_full = np.concatenate([Y_pde, Y_bc], axis=0) if has_bc else Y_pde

    # ---------------------------------------------------------------------
    # 5. Representer weights, bypassing L(prior.mean) (zero-mean only).
    # ---------------------------------------------------------------------
    G_linop = pn.linops.aslinop(G)
    G_linop.is_symmetric = True
    G_linop.is_positive_definite = True
    representer_weights = np.linalg.solve(G, y_full)

    # ---------------------------------------------------------------------
    # 6. Assemble the ConditionalGaussianProcess.
    # ---------------------------------------------------------------------
    b_pde = (
        pn.randvars.Normal(np.zeros(m), (sigma_pde**2 + pde_jitter) * np.eye(m))
        if (sigma_pde > 0.0 or pde_jitter > 0.0)
        else None
    )
    if has_bc:
        b_bc = (
            pn.randvars.Normal(np.zeros(n_bc), (sigma_bc**2 + bc_jitter) * np.eye(n_bc))
            if (sigma_bc > 0.0 or bc_jitter > 0.0)
            else None
        )
        Ys, Ls, bs = (Y_pde, Y_bc), (L_pde, L_bc), (b_pde, b_bc)
        kLas = ConditionalGaussianProcess.PriorPredictiveCrossCovariance(
            (kLa_pde, kLa_bc)
        )
    else:
        Ys, Ls, bs = (Y_pde,), (L_pde,), (b_pde,)
        kLas = ConditionalGaussianProcess.PriorPredictiveCrossCovariance((kLa_pde,))

    return ConditionalGaussianProcess(
        prior=prior,
        Ys=Ys,
        Ls=Ls,
        bs=bs,
        kLas=kLas,
        gram_matrix=G_linop,
        representer_weights=representer_weights,
    )
