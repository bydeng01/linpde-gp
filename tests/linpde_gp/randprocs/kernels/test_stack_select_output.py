import numpy as np
import probnum as pn

import linpde_gp


def test_stack_covariance_selectoutput_handles_diagonal_base_kernel():
    class DiagonalVectorKernel(pn.randprocs.covfuncs.CovarianceFunction):
        def __init__(self):
            super().__init__(
                input_shape=(1,),
                output_shape_0=(2,),
                output_shape_1=(2,),
            )

        def _evaluate(self, x0, x1):  # pylint: disable=unused-argument
            x0 = np.asarray(x0)
            n = x0.shape[0]
            # Return only diagonal entries (shape (n, 2, 2)) to mimic kernels that
            # skip the full pairwise expansion.
            cov = np.zeros((n,) + self.output_shape_0 + self.output_shape_1)
            cov[...] = np.eye(2)
            return cov

    base_kernel = DiagonalVectorKernel()
    summed_kernel = base_kernel + base_kernel

    selector = linpde_gp.linfuncops.SelectOutput(((1,), (2,)), 0)
    selected_kernel = selector(summed_kernel, argnum=0)

    stacked = linpde_gp.randprocs.covfuncs.StackCovarianceFunction(
        [selected_kernel],
        output_idx=0,
    )

    X = np.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    cov = stacked(X[:, None, ...], X[None, ...])

    # Expected shape: (batch_x0, batch_x1, stacked_output_dim, remaining_output_dim)
    assert cov.shape == (4, 4, 1, 2)

    diag_idx = np.arange(4)
    expected_diag = np.zeros((4, 2))
    expected_diag[:, 0] = 2.0
    np.testing.assert_allclose(cov[diag_idx, diag_idx, 0, :], expected_diag)
    np.testing.assert_allclose(cov[:, :, 0, 1], 0.0)
