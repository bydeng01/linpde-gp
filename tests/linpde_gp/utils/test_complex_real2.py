import numpy as np

from linpde_gp.utils import from_real2, to_real2


def test_to_real2_stacks_real_imag():
    values = np.array([1.0 + 2.0j, -3.0 + 0.5j])
    stacked = to_real2(values)
    assert stacked.shape == (2, 2)
    assert np.allclose(stacked[..., 0], np.real(values))
    assert np.allclose(stacked[..., 1], np.imag(values))


def test_from_real2_roundtrip():
    values = np.random.randn(4, 3, 2)
    reconstructed = from_real2(values)
    manual = values[..., 0] + 1j * values[..., 1]
    assert np.allclose(reconstructed, manual)


def test_roundtrip():
    data = np.random.randn(5) + 1j * np.random.randn(5)
    stacked = to_real2(data)
    recon = from_real2(stacked)
    assert np.allclose(recon, data)


def test_to_real2_handles_already_stacked_input():
    stacked = np.array([[1.0, -0.5], [0.0, 2.0]])
    result = to_real2(stacked)
    assert result.shape == stacked.shape
    assert np.allclose(result, stacked)
