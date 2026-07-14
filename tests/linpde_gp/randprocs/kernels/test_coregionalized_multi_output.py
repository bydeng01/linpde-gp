"""Unit tests for the Phase 8a ICM covariance function.

Parallels ``test_independent_multi_output.py``. Covers: output shape, batched
input, joint-swap symmetry, the off-diagonal entries equalling ``B[c, c'] *
base``, ``linop`` agreeing with the dense evaluation, and the reduction
identity ``CoregionalizedMultiOutput(base, eye(P)) == IndependentMultiOutput(
base, ..., base)`` to machine precision (the backward-compatibility anchor, V5).
"""

import numpy as np

import pytest

from linpde_gp.randprocs.covfuncs import (
    CoregionalizedMultiOutputCovarianceFunction,
    IndependentMultiOutputCovarianceFunction,
    Matern,
    TensorProduct,
)


@pytest.fixture
def base_kernel():
    # Scalar-output, 2D-input base kernel (TensorProduct of two scalar Materns),
    # mirroring the input_shape == (2,) used in test_independent_multi_output.py.
    return TensorProduct(
        Matern((), nu=2.5, lengthscales=0.7),
        Matern((), nu=2.5, lengthscales=1.3),
    )


@pytest.fixture
def B2():
    # Symmetric positive-definite 2x2 coregionalization matrix (the real2 case).
    return np.array([[1.0, 0.5], [0.5, 2.0]])


@pytest.fixture
def inputs_unbatched():
    rng = np.random.default_rng(9238134)
    return rng.random(size=(2,)), rng.random(size=(2,))


@pytest.fixture
def inputs_batched():
    rng = np.random.default_rng(9238134)
    return rng.random(size=(10, 1, 2)), rng.random(size=(1, 15, 2))


def test_output_shape(base_kernel, B2, inputs_unbatched):
    mo = CoregionalizedMultiOutputCovarianceFunction(base_kernel, B2)
    assert mo.output_shape_0 == (2,)
    assert mo.output_shape_1 == (2,)
    x0, x1 = inputs_unbatched
    res = mo(x0, x1)
    assert res.shape == (2, 2)


def test_offdiagonal_equals_B_times_base(base_kernel, B2, inputs_unbatched):
    mo = CoregionalizedMultiOutputCovarianceFunction(base_kernel, B2)
    x0, x1 = inputs_unbatched
    base_val = base_kernel(x0, x1)
    res = mo(x0, x1)
    # Every entry (including off-diagonal) must equal B[c, c'] * base.
    for c, cp in np.ndindex(2, 2):
        np.testing.assert_allclose(res[c, cp], B2[c, cp] * base_val)


def test_batched_input(base_kernel, B2, inputs_batched):
    mo = CoregionalizedMultiOutputCovarianceFunction(base_kernel, B2)
    x0, x1 = inputs_batched
    res = mo(x0, x1)
    assert res.shape == (10, 15, 2, 2)
    base_vals = base_kernel(x0, x1)
    assert base_vals.shape == (10, 15)
    for c, cp in np.ndindex(2, 2):
        np.testing.assert_allclose(res[..., c, cp], B2[c, cp] * base_vals)


def test_joint_swap_symmetry(base_kernel, B2, inputs_unbatched):
    """k((x, c), (x', c')) == k((x', c'), (x, c))."""
    mo = CoregionalizedMultiOutputCovarianceFunction(base_kernel, B2)
    x0, x1 = inputs_unbatched
    res = mo(x0, x1)
    res_swapped = mo(x1, x0)
    for c, cp in np.ndindex(2, 2):
        np.testing.assert_allclose(res[c, cp], res_swapped[cp, c])


def _dense_from_evaluate(mo, x0, x1):
    """Assemble the channel-major dense Gram (P*N0, P*N1) from mo(x0, x1)."""
    res = mo(x0, x1)  # (N0, N1, P, P)
    n0, n1, p, _ = res.shape
    # (N0, N1, c, c') -> (c, N0, c', N1) -> (P*N0, P*N1)
    return np.transpose(res, (2, 0, 3, 1)).reshape(p * n0, p * n1)


def test_linop_matches_dense(base_kernel, B2, inputs_batched):
    mo = CoregionalizedMultiOutputCovarianceFunction(base_kernel, B2)
    x0, x1 = inputs_batched
    num_input_0 = int(np.prod(x0.shape[:-1]))
    num_input_1 = int(np.prod(x1.shape[:-1]))

    linop = mo.linop(x0, x1)
    assert linop.shape == (2 * num_input_0, 2 * num_input_1)
    linop_dense = linop @ np.eye(linop.shape[1])

    # Truth: B (x) K (Kronecker), channel-major ordering.
    kron_ref = np.kron(B2, base_kernel.matrix(x0, x1))
    np.testing.assert_allclose(linop_dense, kron_ref)

    # linop must also agree with the dense evaluation path.
    np.testing.assert_allclose(linop_dense, _dense_from_evaluate(mo, x0, x1))


@pytest.mark.parametrize("p", [2, 3])
def test_reduction_identity(base_kernel, inputs_batched, p):
    """ICM(base, eye(P)) == IndependentMultiOutput(base, ..., base) (V5)."""
    x0, x1 = inputs_batched
    icm = CoregionalizedMultiOutputCovarianceFunction(base_kernel, np.eye(p))
    iid = IndependentMultiOutputCovarianceFunction(*([base_kernel] * p))

    # Dense element-wise evaluation.
    np.testing.assert_allclose(icm(x0, x1), iid(x0, x1), atol=0.0, rtol=0.0)

    # Dense Gram via linop.
    icm_dense = icm.linop(x0, x1) @ np.eye(icm.linop(x0, x1).shape[1])
    iid_dense = iid.linop(x0, x1) @ np.eye(iid.linop(x0, x1).shape[1])
    np.testing.assert_allclose(icm_dense, iid_dense, atol=0.0, rtol=0.0)


def test_invalid_B_rejected(base_kernel):
    with pytest.raises(ValueError):  # not symmetric
        CoregionalizedMultiOutputCovarianceFunction(
            base_kernel, np.array([[1.0, 0.5], [0.2, 1.0]])
        )
    with pytest.raises(ValueError):  # not PSD
        CoregionalizedMultiOutputCovarianceFunction(
            base_kernel, np.array([[1.0, 2.0], [2.0, 1.0]])
        )
