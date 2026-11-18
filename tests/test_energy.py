import numpy as np
import pytest

from seam_carving.energy import sobel_energy, to_grayscale


def test_constant_image_has_zero_energy():
    image = np.full((5, 6, 3), 100, dtype=np.uint8)
    assert np.allclose(sobel_energy(image), 0.0)


def test_vertical_edge_has_high_energy():
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    image[:, 3:] = 255
    energy = sobel_energy(image)
    assert energy.shape == (5, 6)
    assert energy[2, 2] > 100
    assert energy[2, 3] > 100
    # Far away from the edge the energy is zero.
    assert np.isclose(energy[2, 0], 0.0)
    assert np.isclose(energy[2, 5], 0.0)


def test_grayscale_input_is_supported():
    gray = np.zeros((4, 4), dtype=np.uint8)
    gray[:, 2:] = 255
    energy = sobel_energy(gray)
    assert energy.shape == (4, 4)
    assert energy.max() > 100


def test_luminance_weights():
    red = np.zeros((1, 1, 3), dtype=np.uint8)
    red[0, 0, 0] = 255
    assert np.isclose(to_grayscale(red)[0, 0], 0.299 * 255)


def test_bad_shape_is_rejected():
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((2, 2, 2), dtype=np.uint8))
