import numpy as np
import probnum as pn

import linpde_gp


def test_real2_kernel_shape():
    scalar_kernel = linpde_gp.randprocs.covfuncs.ExpQuad(
        input_shape=(1,), lengthscales=1.0
    )
    wrapper = linpde_gp.randprocs.covfuncs.Real2FromScalarKernel(scalar_kernel)

    X = np.linspace(-1.0, 1.0, 5)[:, None]
    cov = wrapper(X, None)

    assert cov.shape == (5, 2, 2)
    assert np.allclose(cov[..., 0, 1], 0.0)
    assert np.allclose(cov[..., 1, 0], 0.0)


def test_real2_kernel_variance():
    scalar_kernel = linpde_gp.randprocs.covfuncs.ExpQuad(
        input_shape=(1,), lengthscales=0.5
    )
    wrapper = linpde_gp.randprocs.covfuncs.Real2FromScalarKernel(scalar_kernel)

    X = np.zeros((3, 1))
    cov_scalar = scalar_kernel(X, None)
    cov_wrapper = wrapper(X, None)

    assert np.allclose(cov_wrapper[..., 0, 0], cov_scalar / 2.0)
    assert np.allclose(cov_wrapper[..., 1, 1], cov_scalar / 2.0)
