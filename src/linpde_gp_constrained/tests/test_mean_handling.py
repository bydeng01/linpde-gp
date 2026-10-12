"""
Tests for transform_bounds_for_zero_mean.

Mathematical claim being tested
-------------------------------
For any linear operator L_ineq, posterior mean m_E, and inequality bounds
(a, b),

    a <= L_ineq u <= b
        is the same event as
    a - L_ineq m_E <= L_ineq (u - m_E) <= b - L_ineq m_E.

So the helper must compute exactly ``(a - L_ineq m_E, b - L_ineq m_E)`` with
correct broadcasting, ±inf propagation, and a guard against a > b inputs.
"""

from linpde_gp_constrained.mean_handling import transform_bounds_for_zero_mean
import numpy as np


def test_simple_subtraction():
    """Direct algebraic check: a_tilde = a - Lm, b_tilde = b - Lm."""
    Lm = np.array([0.5, -1.0, 2.0, 0.0])
    a = np.array([0.0, -2.0, 1.0, -np.inf])
    b = np.array([1.0, 0.0, 3.0, 5.0])

    a_t, b_t = transform_bounds_for_zero_mean(Lm, a, b)
    np.testing.assert_array_equal(a_t, a - Lm)
    np.testing.assert_array_equal(b_t, b - Lm)


def test_inf_propagation():
    """Infinite bounds (one-sided constraints) survive the transformation."""
    Lm = np.array([0.0, 1.5, -2.0])
    a = np.array([-np.inf, 0.0, -np.inf])
    b = np.array([np.inf, np.inf, 1.0])

    a_t, b_t = transform_bounds_for_zero_mean(Lm, a, b)
    assert a_t[0] == -np.inf
    assert b_t[0] == np.inf
    assert b_t[1] == np.inf
    assert a_t[2] == -np.inf
    assert np.isfinite(a_t[1]) and a_t[1] == 0.0 - 1.5
    assert np.isfinite(b_t[2]) and b_t[2] == 1.0 - (-2.0)


def test_event_equivalence_random():
    """
    Sample-based equivalence: for many random functions u and a fixed
    L_ineq m_E, the events
        {a <= L_ineq u <= b}      and      {a_t <= L_ineq (u - m_E) <= b_t}
    have the same indicator pattern.

    L_ineq is taken as identity here so the test does not require operator
    machinery; it covers the bound transformation specifically.
    """
    rng = np.random.default_rng(0)
    Lm = rng.standard_normal(7)  # m_E values at 7 virtual points
    a = rng.uniform(-2, 0, 7)
    b = rng.uniform(0, 2, 7)
    a, b = np.minimum(a, b), np.maximum(a, b)

    a_t, b_t = transform_bounds_for_zero_mean(Lm, a, b)

    for _ in range(2000):
        u = rng.standard_normal(7)
        original_event = np.all((a <= u) & (u <= b))
        # Equivalent event in centered coordinates: a_t <= u - Lm <= b_t
        centered = u - Lm
        centered_event = np.all((a_t <= centered) & (centered <= b_t))
        assert original_event == centered_event


def test_invalid_bounds_raise():
    """a > b on any coordinate must raise."""
    Lm = np.array([0.0, 0.0])
    a = np.array([1.0, 2.0])
    b = np.array([0.0, 5.0])  # a[0] > b[0]
    try:
        transform_bounds_for_zero_mean(Lm, a, b)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for a > b inputs")


def test_broadcasting_scalar_bounds():
    """Scalar bounds should broadcast against the m-dim Lm vector."""
    Lm = np.array([1.0, -1.0, 0.0])
    a_t, b_t = transform_bounds_for_zero_mean(Lm, 0.0, 1.0)
    np.testing.assert_array_equal(a_t, np.array([-1.0, 1.0, 0.0]))
    np.testing.assert_array_equal(b_t, np.array([0.0, 2.0, 1.0]))
