import numpy as np
import probnum as pn

import linpde_gp


def test_condition_accepts_batch_last_observations():
    gp = pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(), output_shape=(2,)),
        cov=linpde_gp.randprocs.covfuncs.Real2FromScalarKernel(
            linpde_gp.randprocs.covfuncs.ExpQuad(input_shape=(), lengthscales=1.0)
        ),
    )

    X = np.linspace(0.0, 1.0, 4)
    Y = np.stack((np.sin(X), np.cos(X)), axis=-1)

    L = linpde_gp.linfuncops.SelectOutput(((), (2,)), idx=slice(None))

    conditioned = gp.condition_on_observations(Y, X=X, L=L)

    assert conditioned.mean(X).shape == Y.shape
