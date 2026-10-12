"""
Mean-subtraction utility for the LinPDE-GP × gp_constr integration.

Mathematical identity
---------------------
Let u ~ GP(m_E, k_E) be the linpde-gp posterior after equality conditioning.
Define the centered process

    tilde u(x) := u(x) - m_E(x).

Then tilde u ~ GP(0, k_E). For any linear operator L_ineq, an inequality
constraint

    a(x) <= L_ineq u(x) <= b(x)

is equivalent (event-wise, sample-by-sample, by linearity of L_ineq) to

    a(x) - L_ineq m_E(x) <= L_ineq tilde u(x) <= b(x) - L_ineq m_E(x).

This lets us pass mean=0 to gp_constr's GPmodel (whose 'mean' parameter is a
*constant scalar*, not a callable) while folding the spatially-varying
posterior mean of u into spatially-varying inequality bounds at the virtual
points X_v.

Why this is the cleanest fix
----------------------------
gp_constr's GPmodel only supports a constant prior mean. Modifying it to
accept a callable mean would be an upstream edit to a paper-backed library;
the wrapper-level fix below is mathematically equivalent and requires zero
upstream change.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def transform_bounds_for_zero_mean(
    L_m_E_at_Xv: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform inequality bounds (a, b) on L_ineq u to bounds on L_ineq (u - m_E).

    Parameters
    ----------
    L_m_E_at_Xv
        Values of (L_ineq m_E) at the virtual points X_v. Shape (m,).
    a, b
        Original lower / upper bounds on (L_ineq u)(X_v). Each shape (m,) or
        broadcastable to that shape; ``np.inf`` and ``-np.inf`` are accepted
        for one-sided constraints.

    Returns
    -------
    a_tilde, b_tilde
        Transformed bounds: ``a - L_ineq m_E`` and ``b - L_ineq m_E``. These
        are the bounds to pass to gp_constr's Constraint object together with
        a zero-mean GP. ``np.inf`` propagates through addition/subtraction
        unchanged, so one-sided constraints survive the transformation.

    Notes
    -----
    The transformation is purely linear: by linearity of L_ineq,
    ``L_ineq (u - m_E) = L_ineq u - L_ineq m_E``, so the original event
    ``{a <= L_ineq u <= b}`` is the same as
    ``{a - L_ineq m_E <= L_ineq (u - m_E) <= b - L_ineq m_E}``. This holds
    pathwise and therefore preserves the truncated-Gaussian posterior exactly.

    The function performs no shape inference about L_ineq (e.g., whether it
    is identity, a directional derivative, or a Laplacian); the caller is
    responsible for evaluating L_ineq m_E at the virtual points using the
    same operator they will pass to gp_constr.
    """
    L_m_E_at_Xv = np.asarray(L_m_E_at_Xv, dtype=float)
    a = np.broadcast_to(np.asarray(a, dtype=float), L_m_E_at_Xv.shape).copy()
    b = np.broadcast_to(np.asarray(b, dtype=float), L_m_E_at_Xv.shape).copy()

    if np.any(a > b):
        raise ValueError("Lower bound a exceeds upper bound b at some virtual point.")

    a_tilde = a - L_m_E_at_Xv
    b_tilde = b - L_m_E_at_Xv

    # ±inf propagation guard (sanity check: subtracting a finite number from
    # ±inf must give ±inf; numpy already does this correctly but we assert).
    finite_Lm = np.isfinite(L_m_E_at_Xv)
    assert np.all(np.isfinite(a_tilde) | ~finite_Lm | ~np.isfinite(a))
    assert np.all(np.isfinite(b_tilde) | ~finite_Lm | ~np.isfinite(b))

    return a_tilde, b_tilde


def evaluate_operator_on_mean(
    posterior_mean_callable: Callable[[np.ndarray], np.ndarray],
    Xv: np.ndarray,
    operator_apply: Callable[[Callable, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Convenience helper: evaluate (L_ineq m_E) at virtual points X_v.

    Parameters
    ----------
    posterior_mean_callable
        A callable representing m_E (e.g., ``posterior_gp.mean`` from
        linpde-gp). Must accept the appropriately shaped X_v.
    Xv
        Virtual points, shape (m, d) by gp_constr convention.
    operator_apply
        A function ``(callable_f, Xv) -> np.ndarray of shape (m,)`` that
        evaluates (L_ineq f) at X_v. If None (default), assumes
        ``L_ineq = Identity`` and just evaluates m_E directly. Supply your own
        for derivative / Laplacian inequalities.

    Returns
    -------
    L_m_E_at_Xv
        Shape-(m,) array.
    """
    if operator_apply is None:
        return np.asarray(posterior_mean_callable(Xv)).reshape(-1)
    return np.asarray(operator_apply(posterior_mean_callable, Xv)).reshape(-1)
