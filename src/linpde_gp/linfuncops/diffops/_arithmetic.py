import functools

import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ScalarType

from ._lindiffop import LinearDifferentialOperator


class ScaledLinearDifferentialOperator(LinearDifferentialOperator):
    def __init__(
        self, lindiffop: LinearDifferentialOperator, /, scalar: ScalarLike
    ) -> None:
        self._lindiffop = lindiffop

        if not np.ndim(scalar) == 0:
            raise ValueError()

        # Handle complex scalars
        if np.iscomplex(scalar) or np.iscomplexobj(scalar):
            self._scalar = np.asarray(scalar, dtype=np.complex128)
        else:
            self._scalar = np.asarray(scalar, dtype=np.double)

        # Convert scalar to appropriate type for coefficient multiplication
        scalar_for_coeffs = (
            complex(self._scalar)
            if np.iscomplexobj(self._scalar)
            else float(self._scalar)
        )

        super().__init__(
            coefficients=scalar_for_coeffs * self._lindiffop.coefficients,
            input_shapes=self._lindiffop.input_shapes,
        )

    @property
    def lindiffop(self) -> LinearDifferentialOperator:
        return self._lindiffop

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self._scalar * self._lindiffop(f, **kwargs)

    def _jax_fallback(self, f, /, **kwargs):
        raise NotImplementedError()

    # TODO: Only need until GPs can be scaled
    @__call__.register
    def _(
        self, gp: pn.randprocs.GaussianProcess, /, **kwargs
    ) -> pn.randprocs.GaussianProcess:
        return super().__call__(gp, **kwargs)

    def __rmul__(self, other) -> LinearDifferentialOperator:
        if np.ndim(other) == 0:
            return ScaledLinearDifferentialOperator(
                lindiffop=self._lindiffop,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)

    @functools.singledispatchmethod
    def weak_form(self, test_basis, /):
        return self._scalar * self._lindiffop.weak_form(test_basis)

    def __repr__(self) -> str:
        return f"{self._scalar} * {self._lindiffop}"
