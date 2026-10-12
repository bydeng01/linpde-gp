r"""Constant-coefficient steady advection-diffusion operator for linpde-gp.

Operator
--------
    L = -D Delta + v . grad + k I
      = -D (d_xx + d_yy) + vx d_x + vy d_y + k

acting on a scalar field c(x, y). All coefficients are constants (D > 0
diffusivity, v = (vx, vy) wind, k >= 0 linear decay).

Assembly choice
---------------
L is built as a ``SumLinearFunctionOperator`` of three *named* operators ---
``WeightedLaplacian(-D)``, ``DirectionalDerivative(v)`` and
``IdentityOperator(k)`` --- exactly the Helmholtz-operator pattern. This is NOT
cosmetic: ``ExpQuad`` registers closed-form diffop handlers keyed on these named
operator classes (``ExpQuad_WeightedLaplacian_*``, ``ExpQuad_DirectionalDerivative_*``,
``ExpQuad_Identity_*``), but NOT on bare ``PartialDerivative``. A single merged
``LinearDifferentialOperator`` decomposes through ``to_sum()`` into
``PartialDerivative`` summands, for which the ExpQuad chain falls through to a
``NotImplemented`` first-order partial and raises. Keeping the advection term as
``DirectionalDerivative`` routes it to its registered handler, so every term of
the double chain ``L_s L_t k`` stays closed form. No upstream handler is added;
nothing is reimplemented locally beyond this assembly helper.

The builder returns a ``SumLinearFunctionOperator`` so it drops into
``make_pde_collocation_posterior(..., pde_op=L, pde_sign=1.0)``: the sign is
already folded into the coefficients (``-D`` on the Laplacian), and the residual
RHS is ``Y_pde = S``.
"""

from __future__ import annotations

import numpy as np

from linpde_gp.linfuncops._arithmetic import SumLinearFunctionOperator
from linpde_gp.linfuncops.diffops import DirectionalDerivative, WeightedLaplacian
from linpde_gp.linfuncops.diffops._helmholtz_operator import IdentityOperator


def make_adv_diff_operator(
    D: float, v: tuple[float, float], k: float, dim: int = 2
) -> SumLinearFunctionOperator:
    """Assemble L = -D Delta + v . grad + k I as a sum of named operators.

    Parameters
    ----------
    D
        Diffusivity (> 0). Enters as ``-D`` on each second derivative.
    v
        Wind vector ``(vx, vy)``; advection term ``v . grad``.
    k
        Linear decay rate (>= 0); zeroth-order term ``k I``.
    dim
        Spatial dimension (2 here). The domain shape is ``(dim,)``.

    Returns
    -------
    L
        ``SumLinearFunctionOperator`` of ``WeightedLaplacian(-D)``,
        ``DirectionalDerivative(v)`` and ``IdentityOperator(k)``;
        ``input_shapes == ((dim,), ())``. The named summands route through the
        closed-form ExpQuad diffop handlers (see check_dispatch.py).
    """
    v = np.asarray(v, dtype=float)
    if v.shape != (dim,):
        raise ValueError(f"v must have shape ({dim},); got {v.shape}.")

    laplacian = WeightedLaplacian(-float(D) * np.ones(dim))
    advection = DirectionalDerivative(v)
    decay = IdentityOperator((dim,), scalar=float(k))
    return SumLinearFunctionOperator(laplacian, advection, decay)
