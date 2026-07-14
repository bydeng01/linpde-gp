from __future__ import annotations

import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - JAX optional
    jnp = None


def _get_backend(values):
    """Select numpy or jax.numpy based on the input array type."""

    if jnp is not None and isinstance(values, jnp.ndarray):
        return jnp
    return np


def to_real2(values: np.ndarray) -> np.ndarray:
    """Stack real and imaginary parts along a new last axis (NumPy or JAX)."""

    xp = _get_backend(values)
    values = xp.asarray(values)

    # If the input already stores (Re, Im) in the last axis, return it unchanged
    # to avoid stacking an extra dimension.
    if values.shape and values.shape[-1] == 2 and not np.iscomplexobj(values):
        return values

    real = xp.real(values)
    imag = xp.imag(values)

    stacked = xp.stack((real, imag), axis=-1)

    return stacked


def from_real2(values: np.ndarray) -> np.ndarray:
    """Reconstruct complex array from stacked real-imag representation."""

    xp = _get_backend(values)
    values = xp.asarray(values)

    if values.shape[-1] != 2:
        raise ValueError("Expected last axis to have length 2 for (Re, Im).")

    real = values[..., 0]
    imag = values[..., 1]

    return real + 1j * imag
