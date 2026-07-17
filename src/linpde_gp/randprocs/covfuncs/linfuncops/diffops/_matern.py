from __future__ import annotations

import functools
from typing import Optional

from jax import numpy as jnp
import numpy as np
from probnum.randprocs import covfuncs
from pykeops.numpy import LazyTensor, Pm, Vi, Vj

from linpde_gp.functions import Monomial, RationalPolynomial
from linpde_gp.linfuncops import diffops

from ..._jax import JaxCovarianceFunction, JaxIsotropicMixin


class HalfIntegerMatern_Identity_DirectionalDerivative(JaxCovarianceFunction):
    def __init__(
        self,
        matern: covfuncs.Matern,
        *,
        direction: np.ndarray,
        reverse: bool = False,
    ):
        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._direction = direction
        self._reverse = reverse

        self._matern_scale_factors = self._matern._scale_factors

        self._poly = half_integer_matern_derivative_polynomial(
            self.matern.p, 1
        ) // Monomial(1)

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _scaled_direction(self) -> np.ndarray:
        scaled_direction = self._matern_scale_factors
        scaled_direction *= self.direction

        if not self._reverse:
            scaled_direction *= -1

        return scaled_direction

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern_scale_factors

        proj_scaled_diffs = self._batched_sum(self._scaled_direction * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm(scaled_diffs)

        # Polynomial part
        res = self._poly(scaled_dists)

        # Exponential part
        res *= np.exp(-scaled_dists)

        # Chain Rule
        res *= proj_scaled_diffs

        return res

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern_scale_factors

        proj_scaled_diffs = self._batched_sum_jax(self._scaled_direction * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm_jax(scaled_diffs)

        # Polynomial part
        res = self._poly.jax(scaled_dists)

        # Exponential part
        res *= jnp.exp(-scaled_dists)

        # Chain Rule
        res *= proj_scaled_diffs

        return res

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        if x1 is None:
            x1 = x0
        if len(x0.shape) < 2:
            x0 = x0.reshape(-1, 1)
        if len(x1.shape) < 2:
            x1 = x1.reshape(-1, 1)

        scaled_diffs = Vi(x0) - Vj(x1)
        scaled_diffs *= Pm(self._matern_scale_factors)

        proj_scaled_diffs = (Pm(self._scaled_direction) * scaled_diffs).sum()
        scaled_dists = scaled_diffs * scaled_diffs
        scaled_dists = scaled_dists.sum().sqrt()

        res = self._poly._evaluate_keops(  # pylint: disable=protected-access
            scaled_dists
        )
        res *= (-scaled_dists).exp()
        res *= proj_scaled_diffs

        return res


class HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative(
    JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._matern_scale_factors = self._matern._scale_factors

        self._direction0 = direction0
        self._direction1 = direction1

        self._neg_poly_deriv = -half_integer_matern_derivative_polynomial(
            self._matern.p, 1
        ) // Monomial(1)

        self._poly_diff = (
            half_integer_matern_derivative_polynomial(self._matern.p, 2)
            + self._neg_poly_deriv
        ) // Monomial(2)

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @functools.cached_property
    def _scaled_direction0(self) -> np.ndarray:
        return self._matern_scale_factors * self._direction0

    @functools.cached_property
    def _scaled_direction1(self) -> np.ndarray:
        return self._matern_scale_factors * self._direction1

    @functools.cached_property
    def _directions_inprod(self) -> np.floating:
        return np.sum(self._scaled_direction0 * self._scaled_direction1)

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._directions_inprod * self._neg_poly_deriv.coefficients[0],
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern_scale_factors

        proj_scaled_diffs0 = self._batched_sum(self._scaled_direction0 * scaled_diffs)
        proj_scaled_diffs1 = self._batched_sum(self._scaled_direction1 * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm(scaled_diffs)

        res = self._directions_inprod * self._neg_poly_deriv(scaled_dists)
        res -= proj_scaled_diffs0 * proj_scaled_diffs1 * self._poly_diff(scaled_dists)

        return res * np.exp(-scaled_dists)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._directions_inprod * self._neg_poly_deriv.coefficients[0],
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern_scale_factors

        proj_scaled_diffs0 = self._batched_sum_jax(
            self._scaled_direction0 * scaled_diffs
        )
        proj_scaled_diffs1 = self._batched_sum_jax(
            self._scaled_direction1 * scaled_diffs
        )
        scaled_dists = self._batched_euclidean_norm_jax(scaled_diffs)

        res = self._directions_inprod * self._neg_poly_deriv.jax(scaled_dists)
        res -= (
            proj_scaled_diffs0 * proj_scaled_diffs1 * self._poly_diff.jax(scaled_dists)
        )

        return res * jnp.exp(-scaled_dists)

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        if x1 is None:
            x1 = x0
        if len(x0.shape) < 2:
            x0 = x0.reshape(-1, 1)
        if len(x1.shape) < 2:
            x1 = x1.reshape(-1, 1)

        scaled_diffs = Vi(x0) - Vj(x1)
        scaled_diffs *= Pm(self._matern_scale_factors)

        proj_scaled_diffs0 = (Pm(self._scaled_direction0) * scaled_diffs).sum()
        proj_scaled_diffs1 = (Pm(self._scaled_direction1) * scaled_diffs).sum()

        scaled_dists = scaled_diffs * scaled_diffs
        scaled_dists = scaled_dists.sum().sqrt()

        res = Pm(
            self._directions_inprod
        ) * self._neg_poly_deriv._evaluate_keops(  # pylint: disable=protected-access
            scaled_dists
        )
        res -= (
            proj_scaled_diffs0
            * proj_scaled_diffs1
            * self._poly_diff._evaluate_keops(  # pylint: disable=protected-access
                scaled_dists
            )
        )
        res *= (-scaled_dists).exp()

        return res


class UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative(
    covfuncs.IsotropicMixin, JaxIsotropicMixin, JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        if matern.input_size != 1:
            raise ValueError("`matern` must be univariate.")

        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._matern_scale_factors = self._matern._scale_factors

        self._direction0 = direction0
        self._direction1 = direction1

        self._poly = -half_integer_matern_derivative_polynomial(self._matern.p, 2)

    @functools.cached_property
    def _scaled_directions_prod(self) -> np.floating:
        return np.squeeze(
            self._direction0 * self._direction1 * self._matern_scale_factors**2
        )[()]

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._scaled_directions_prod * self._poly.coefficients[0],
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_dists = self._euclidean_distances(
            x0,
            x1,
            scale_factors=self._matern_scale_factors,
        )

        return (
            self._scaled_directions_prod
            * self._poly(scaled_dists)
            * np.exp(-scaled_dists)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._scaled_directions_prod * self._poly.coefficients[0],
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_dists = self._euclidean_distances_jax(
            x0,
            x1,
            scale_factors=self._matern_scale_factors,
        )

        return (
            self._scaled_directions_prod
            * self._poly.jax(scaled_dists)
            * jnp.exp(-scaled_dists)  # pylint: disable=invalid-unary-operand-type
        )

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        scaled_dists = self._euclidean_distances_keops(
            x0,
            x1,
            scale_factors=self._matern_scale_factors,
        )

        return (
            Pm(self._scaled_directions_prod)
            * self._poly._evaluate_keops(  # pylint: disable=protected-access
                scaled_dists
            )
            * (-scaled_dists).exp()
        )


class HalfIntegerMatern_Identity_WeightedLaplacian(JaxCovarianceFunction):
    r"""Closed-form :math:`\Delta_a k` for a multi-D half-integer Matern kernel.

    Mirrors :class:`UnivariateHalfIntegerMatern_Identity_WeightedLaplacian` but
    works for ``matern.input_size > 1``. The closed form requires the radial
    function

    .. math::
        \tilde\gamma(r) := \frac{\phi''(r) - \phi'(r)/r}{r^2}

    to be smooth at :math:`r=0`, which holds for half-integer Matern with
    smoothness :math:`\nu = p + 1/2` and :math:`p \ge 2` (Matern 5/2, 7/2, ...).
    Univariate ``p = 1`` (Matern 3/2) is still served by the existing
    univariate class because in 1D the formula collapses to
    :math:`w \cdot s^2 \phi''(r)`.

    The evaluation uses the identity (in scaled coordinates
    :math:`u_i = s_i (x_{0,i} - x_{1,i})`, :math:`r = \|u\|`):

    .. math::
        \Delta_a k(x_0, x_1)
        = e^{-r} \, \big[ \, \tilde p_0(r) \, Q_a(u) + W_a \, q_1(r) \, \big]

    where :math:`s_i` is the per-dimension Matern scale factor
    (``= sqrt(2*nu)/lengthscale_i``), :math:`\bar w_{a,i} = w_{a,i} s_i^2`,
    :math:`W_a = \sum_i \bar w_{a,i}`, :math:`Q_a(u) = \sum_i \bar w_{a,i} u_i^2`,
    :math:`q_1(r) = p_1(r)/r` and :math:`\tilde p_0(r) = (p_2(r) - q_1(r))/r^2`
    are the Matern derivative polynomials.
    """

    def __init__(
        self,
        matern: covfuncs.Matern,
        L: diffops.WeightedLaplacian,
        reverse: bool = True,
    ):
        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )
        if matern.p < 2:
            raise ValueError(
                "Multi-D Matern Laplacian closed form requires p >= 2 "
                "(smoothness nu >= 5/2)."
            )
        if matern.input_size <= 1:
            raise ValueError(
                "Use `UnivariateHalfIntegerMatern_Identity_WeightedLaplacian` "
                "for univariate Matérn kernels."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._L = L
        self._reverse = bool(reverse)

        self._matern_scale_factors = (
            np.broadcast_to(matern._scale_factors, matern.input_shape)
            .astype(np.float64)
            .copy()
        )

        # phi'(r)/r = q_1(r) e^{-r}
        self._q1 = half_integer_matern_derivative_polynomial(matern.p, 1) // Monomial(1)
        # tilde_gamma(r) = (phi''(r) - phi'(r)/r) / r^2 = tilde_p_0(r) e^{-r}
        p2 = half_integer_matern_derivative_polynomial(matern.p, 2)
        self._tp0 = (p2 - self._q1) // Monomial(2)

        weights = (
            np.broadcast_to(L.weights, matern.input_shape).astype(np.float64).copy()
        )
        s = self._matern_scale_factors
        # bar_w_i = w_i * s_i^2
        self._bar_w = weights * s * s
        self._W = float(np.sum(self._bar_w))

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @property
    def L(self) -> diffops.WeightedLaplacian:
        return self._L

    @property
    def reverse(self) -> bool:
        return self._reverse

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            # u = 0, so Q_a(u) = 0; only W_a * q_1(0) survives.
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._W * float(self._q1.coefficients[0]),
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1
        scaled_diffs = diffs * self._matern_scale_factors  # u_i = s_i * dz_i
        scaled_dists = self._batched_euclidean_norm(scaled_diffs)  # r = ||u||
        Q_u = self._batched_sum(self._bar_w * scaled_diffs * scaled_diffs)

        return np.exp(-scaled_dists) * (
            self._tp0(scaled_dists) * Q_u + self._W * self._q1(scaled_dists)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._W * float(self._q1.coefficients[0]),
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1
        scaled_diffs = diffs * self._matern_scale_factors
        scaled_dists = self._batched_euclidean_norm_jax(scaled_diffs)
        Q_u = self._batched_sum_jax(self._bar_w * scaled_diffs * scaled_diffs)

        return jnp.exp(-scaled_dists) * (
            self._tp0.jax(scaled_dists) * Q_u + self._W * self._q1.jax(scaled_dists)
        )


class UnivariateHalfIntegerMatern_Identity_WeightedLaplacian(
    covfuncs.IsotropicMixin, JaxIsotropicMixin, JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        L: diffops.WeightedLaplacian,
        reverse: bool = True,
    ):
        if matern.input_size != 1:
            raise ValueError("`matern` must be univariate.")

        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._L = L
        self._reverse = bool(reverse)

        self._matern_scale_factors = self._matern._scale_factors

        self._poly = half_integer_matern_derivative_polynomial(matern.p, 2)

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @property
    def L(self) -> diffops.WeightedLaplacian:
        return self._L

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _output_scale_factor(self) -> np.floating:
        return np.squeeze(
            self._L.weights * self._matern_scale_factors * self._matern_scale_factors
        )[()]

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        scaled_dists = self._euclidean_distances(
            x0, x1, scale_factors=self._matern_scale_factors
        )

        return (
            self._output_scale_factor * np.exp(-scaled_dists) * self._poly(scaled_dists)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        scaled_dists = self._euclidean_distances_jax(
            x0, x1, scale_factors=self._matern_scale_factors
        )

        return (
            self._output_scale_factor
            * jnp.exp(-scaled_dists)  # pylint: disable=invalid-unary-operand-type
            * self._poly.jax(scaled_dists)
        )

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        scaled_dists = self._euclidean_distances_keops(
            x0, x1, scale_factors=self._matern_scale_factors
        )

        return (
            Pm(self._output_scale_factor)
            * (-scaled_dists).exp()
            * self._poly._evaluate_keops(  # pylint: disable=protected-access
                scaled_dists
            )
        )


class HalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(JaxCovarianceFunction):
    r"""Closed-form :math:`\Delta_0 \Delta_1 k` for a multi-D half-integer
    Matern kernel.

    Multi-D analogue of
    :class:`UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian`.
    Restricted to half-integer Matern with smoothness :math:`\nu = p + 1/2` and
    :math:`p \ge 2` (Matern 5/2 and smoother), since :math:`\Delta_0 \Delta_1 k`
    is finite at the diagonal only for :math:`\nu \ge 5/2` in :math:`d > 1`.

    Derivation. Write the kernel as :math:`k(x_0, x_1) = \phi(r)` with
    :math:`r = \|S(x_0 - x_1)\|` and :math:`S = \mathrm{diag}(s)`, and define
    :math:`\alpha(r) := \phi'(r)/r`,
    :math:`\tilde\gamma(r) := (\phi''(r) - \alpha(r))/r^2`. Then a direct
    radial-derivative computation yields the manifestly :math:`0 \leftrightarrow 1`
    symmetric form

    .. math::
        \Delta_0 \Delta_1 k = e^{-r} \, \Big[ \,
        & (2M + W_0 W_1) \, \tilde p_0(r) \\
        + & \big( C_1(u)/r - Q_0(u) Q_1(u)/r^3 \big) \, \tilde p_1(r) \\
        + & Q_0(u) Q_1(u) / r^2 \, \tilde p_2(r) \, \Big]

    in scaled coordinates :math:`u_i = s_i (x_{0,i} - x_{1,i})`, with
    :math:`\bar w_{a,i} = w_{a,i} s_i^2`,
    :math:`W_a = \sum_i \bar w_{a,i}`,
    :math:`Q_a(u) = \sum_i \bar w_{a,i} u_i^2`,
    :math:`M = \sum_i \bar w_{0,i} \bar w_{1,i}`,
    :math:`N(u) = \sum_i \bar w_{0,i} \bar w_{1,i} u_i^2`,
    :math:`C_1(u) = 4 N(u) + W_1 Q_0(u) + W_0 Q_1(u)`, and the polynomials
    :math:`\tilde p_n` recursively defined by
    :math:`\tilde p_0(r) = (p_2(r) - p_1(r)/r) / r^2`,
    :math:`\tilde p_{n+1}(r) = \tilde p_n'(r) - \tilde p_n(r)`.

    The :math:`1/r,\,1/r^3` factors are paired with quadratic forms
    (:math:`C_1, Q_0 Q_1`) that vanish to even order in :math:`u`, so the
    result is finite at the diagonal :math:`r = 0`. Numerically we use the
    standard ``where(r > 0, 1/r, 0)`` idiom to keep autodiff and JIT
    well-defined.
    """

    def __init__(
        self,
        matern: covfuncs.Matern,
        L0: diffops.WeightedLaplacian,
        L1: diffops.WeightedLaplacian,
    ):
        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )
        if matern.p < 2:
            raise ValueError(
                "Multi-D Matern double-Laplacian closed form requires p >= 2 "
                "(smoothness nu >= 5/2)."
            )
        if matern.input_size <= 1:
            raise ValueError(
                "Use `UnivariateHalfIntegerMatern_WeightedLaplacian_"
                "WeightedLaplacian` for univariate Matérn kernels."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._L0 = L0
        self._L1 = L1

        self._matern_scale_factors = (
            np.broadcast_to(matern._scale_factors, matern.input_shape)
            .astype(np.float64)
            .copy()
        )

        # Polynomials for tilde_gamma(r), tilde_gamma'(r), tilde_gamma''(r)
        p2 = half_integer_matern_derivative_polynomial(matern.p, 2)
        q1 = half_integer_matern_derivative_polynomial(matern.p, 1) // Monomial(1)
        self._tp0 = (p2 - q1) // Monomial(2)
        self._tp1 = self._tp0.differentiate() - self._tp0
        self._tp2 = self._tp1.differentiate() - self._tp1

        s = self._matern_scale_factors
        s2 = s * s
        w0 = np.broadcast_to(L0.weights, matern.input_shape).astype(np.float64).copy()
        w1 = np.broadcast_to(L1.weights, matern.input_shape).astype(np.float64).copy()

        self._bar_w0 = w0 * s2  # shape (d,)
        self._bar_w1 = w1 * s2
        self._N_w = self._bar_w0 * self._bar_w1  # for N(u)
        self._W0 = float(np.sum(self._bar_w0))
        self._W1 = float(np.sum(self._bar_w1))
        self._M = float(np.sum(self._N_w))

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    def _diagonal_value(self) -> float:
        return (2.0 * self._M + self._W0 * self._W1) * float(self._tp0.coefficients[0])

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._diagonal_value(),
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1
        u = diffs * self._matern_scale_factors
        u2 = u * u
        r = self._batched_euclidean_norm(u)
        Q0 = self._batched_sum(self._bar_w0 * u2)
        Q1 = self._batched_sum(self._bar_w1 * u2)
        N = self._batched_sum(self._N_w * u2)
        QQ = Q0 * Q1
        C1 = 4.0 * N + self._W1 * Q0 + self._W0 * Q1

        # Safe division at r = 0; numerators vanish faster than r so the
        # corrected value is the diagonal limit.
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_r = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
        inv_r2 = inv_r * inv_r
        inv_r3 = inv_r2 * inv_r

        tp0_r = self._tp0(r)
        tp1_r = self._tp1(r)
        tp2_r = self._tp2(r)

        bracket = (
            (2.0 * self._M + self._W0 * self._W1) * tp0_r
            + (C1 * inv_r - QQ * inv_r3) * tp1_r
            + QQ * inv_r2 * tp2_r
        )
        return np.exp(-r) * bracket

    # pylint: disable=too-many-locals
    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._diagonal_value(),
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1
        u = diffs * self._matern_scale_factors
        u2 = u * u
        r = self._batched_euclidean_norm_jax(u)
        Q0 = self._batched_sum_jax(self._bar_w0 * u2)
        Q1 = self._batched_sum_jax(self._bar_w1 * u2)
        N = self._batched_sum_jax(self._N_w * u2)
        QQ = Q0 * Q1
        C1 = 4.0 * N + self._W1 * Q0 + self._W0 * Q1

        # Standard JAX-safe division: keep the inner divide finite even where
        # the result is masked away, so autodiff doesn't see NaN.
        r_pos = r > 0
        r_safe = jnp.where(r_pos, r, 1.0)
        inv_r = jnp.where(r_pos, 1.0 / r_safe, 0.0)
        inv_r2 = inv_r * inv_r
        inv_r3 = inv_r2 * inv_r

        tp0_r = self._tp0.jax(r)
        tp1_r = self._tp1.jax(r)
        tp2_r = self._tp2.jax(r)

        bracket = (
            (2.0 * self._M + self._W0 * self._W1) * tp0_r
            + (C1 * inv_r - QQ * inv_r3) * tp1_r
            + QQ * inv_r2 * tp2_r
        )
        return jnp.exp(-r) * bracket


class UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
    covfuncs.IsotropicMixin, JaxIsotropicMixin, JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        L0: diffops.WeightedLaplacian,
        L1: diffops.WeightedLaplacian,
    ):
        if matern.input_size != 1:
            raise ValueError("`matern` must be univariate.")

        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._matern_scale_factors = self._matern._scale_factors

        self._L0 = L0
        self._L1 = L1

        self._poly = half_integer_matern_derivative_polynomial(matern.p, 4)

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @functools.cached_property
    def _output_scale_factor(self) -> float:
        return np.squeeze(
            self._L0.weights * self._L1.weights * self._matern_scale_factors**4
        )[()]

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        scaled_dists = self._euclidean_distances(
            x0, x1, scale_factors=self._matern_scale_factors
        )

        return (
            self._output_scale_factor * np.exp(-scaled_dists) * self._poly(scaled_dists)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        scaled_dists = self._euclidean_distances_jax(
            x0, x1, scale_factors=self._matern_scale_factors
        )

        return (
            self._output_scale_factor
            * jnp.exp(-scaled_dists)  # pylint: disable=invalid-unary-operand-type
            * self._poly.jax(scaled_dists)
        )

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        scaled_dists = self._euclidean_distances_keops(
            x0, x1, scale_factors=self._matern_scale_factors
        )

        return (
            Pm(self._output_scale_factor)
            * (-scaled_dists).exp()
            * self._poly._evaluate_keops(  # pylint: disable=protected-access
                scaled_dists
            )
        )


class UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian(
    JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        direction: np.ndarray,
        L1: diffops.WeightedLaplacian,
        reverse: bool = False,
    ):
        if matern.input_size != 1:
            raise ValueError("`matern` must be univariate.")

        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._matern_scale_factors = self._matern._scale_factors

        self._direction = direction
        self._L1 = L1

        self._reverse = bool(reverse)

        self._poly = half_integer_matern_derivative_polynomial(matern.p, 3) // Monomial(
            1
        )

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @functools.cached_property
    def _scaled_direction(self) -> np.ndarray:
        scaled_direction = self._L1.weights * self._matern_scale_factors**3
        scaled_direction *= self._direction

        if self._reverse:
            scaled_direction *= -1

        return scaled_direction

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern_scale_factors

        proj_scaled_diffs = self._batched_sum(self._scaled_direction * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm(scaled_diffs)

        return np.exp(-scaled_dists) * self._poly(scaled_dists) * proj_scaled_diffs

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern_scale_factors

        proj_scaled_diffs = self._batched_sum_jax(self._scaled_direction * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm_jax(scaled_diffs)

        return jnp.exp(-scaled_dists) * self._poly.jax(scaled_dists) * proj_scaled_diffs

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        if x1 is None:
            x1 = x0
        if len(x0.shape) < 2:
            x0 = x0.reshape(-1, 1)
        if len(x1.shape) < 2:
            x1 = x1.reshape(-1, 1)
        scaled_diffs = Vi(x0) - Vj(x1)
        scaled_diffs *= Pm(self._matern_scale_factors)

        proj_scaled_diffs = (Pm(self._scaled_direction) * scaled_diffs).sum()
        scaled_dists = scaled_diffs * scaled_diffs
        scaled_dists = scaled_dists.sum().sqrt()

        return (
            (-scaled_dists).exp()
            * self._poly._evaluate_keops(  # pylint: disable=protected-access
                scaled_dists
            )
            * proj_scaled_diffs
        )


@functools.lru_cache(maxsize=None)
def half_integer_matern_polynomial(p: int) -> RationalPolynomial:
    return RationalPolynomial(covfuncs.Matern.half_integer_coefficients(p))


@functools.lru_cache(maxsize=None)
def half_integer_matern_derivative_polynomial(p: int, n: int) -> RationalPolynomial:
    r"""Polynomial coefficients for `n`-th derivatives of the Matérn covariance function
    with :math:`\nu = p + \frac{1}{2}`.

    We can express the Matérn covariance function as

    .. math::
        k_{\nu}(x_0, x_1) = \kappa_{\nu}(\sqrt{2 \nu} \lVert x_0 - x_1 \rVert_2).

    If :math:`\nu = p + \frac{1}{2}` for some nonnegative integer :math:`p`, then
    :math:`\kappa_\nu(r)` and all its derivatives are products of an exponential
    and a polynomial of degree :math:`p`.
    This function computes the coefficients of the polynomial.
    """

    if n == 0:
        return half_integer_matern_polynomial(p)

    poly = half_integer_matern_derivative_polynomial(p, n - 1)

    return poly.differentiate() - poly
