import numpy as np
import pytest

from seam_carving.carve import (
    FINDERS,
    carve_height,
    carve_width,
    enlarge_height,
    enlarge_width,
    insert_horizontal_seam,
    insert_vertical_seam,
    remove_horizontal_seam,
    remove_vertical_seam,
    resize,
)
from seam_carving.dp import find_vertical_seam


def test_remove_vertical_seam_keeps_other_pixels():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    seam = [0, 1, 2, 3]
    out = remove_vertical_seam(image, seam)
    assert out.shape == (4, 4, 3)
    for i, j in enumerate(seam):
        assert np.array_equal(out[i, :j], image[i, :j])
        assert np.array_equal(out[i, j:], image[i, j + 1:])


def test_remove_horizontal_seam_shape():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    seam = [0, 0, 1, 1, 2]
    out = remove_horizontal_seam(image, seam)
    assert out.shape == (3, 5, 3)


def test_insert_vertical_seam_duplicates_the_pixel():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:, :, 0] = [10, 20, 30, 40]
    seam = [1, 1, 1]
    out = insert_vertical_seam(image, seam)
    assert out.shape == (3, 5, 3)
    assert np.array_equal(out[:, 1, 0], np.array([20, 20, 20]))
    assert np.array_equal(out[:, 2, 0], np.array([20, 20, 20]))
    assert np.array_equal(out[:, 3, 0], np.array([30, 30, 30]))


def test_insert_horizontal_seam_shape():
    image = np.zeros((4, 3, 3), dtype=np.uint8)
    out = insert_horizontal_seam(image, [0, 1, 2])
    assert out.shape == (5, 3, 3)


def test_carve_width_with_all_methods_on_small_image():
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[:, 3:] = 255
    for name, finder in FINDERS.items():
        result = carve_width(image, 5, finder)
        assert result.image.shape == (6, 5, 3)
        assert len(result.seams) == 1
        assert len(result.seams[0]) == 6
        assert len(result.energies) == 1


def test_carve_width_removes_the_expected_number_of_seams():
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    result = carve_width(image, 8)
    assert result.image.shape == (10, 8, 3)
    assert len(result.seams) == 4
    assert all(len(seam) == 10 for seam in result.seams)


def test_carve_height_removes_the_expected_number_of_seams():
    image = np.zeros((12, 10, 3), dtype=np.uint8)
    result = carve_height(image, 9)
    assert result.image.shape == (9, 10, 3)
    assert len(result.seams) == 3
    assert all(len(seam) == 10 for seam in result.seams)


def test_carve_width_noop_returns_no_seams():
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    result = carve_width(image, 6)
    assert result.image.shape == (6, 6, 3)
    assert result.seams == []


def test_invalid_targets_raise():
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        carve_width(image, 0)
    with pytest.raises(ValueError):
        carve_width(image, 7)
    with pytest.raises(ValueError):
        carve_height(image, 0)
    with pytest.raises(ValueError):
        carve_height(image, 7)


def test_enlarge_width_and_height():
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    wide = enlarge_width(image, 8)
    assert wide.image.shape == (6, 8, 3)
    assert len(wide.seams) == 2

    tall = enlarge_height(image, 8)
    assert tall.image.shape == (8, 6, 3)
    assert len(tall.seams) == 2


def test_resize_width_then_height():
    image = np.zeros((12, 14, 3), dtype=np.uint8)
    out, width_result, height_result = resize(image, 10, 11)
    assert out.shape == (11, 10, 3)
    assert width_result is not None and width_result.image.shape == (12, 10, 3)
    assert height_result is not None and height_result.image.shape == (11, 10, 3)


def test_resize_cannot_grow():
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize(image, 7, 6)
    with pytest.raises(ValueError):
        resize(image, 6, 7)


def test_free_functions_validate_seam_length():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        remove_vertical_seam(image, [0, 1])
    with pytest.raises(ValueError):
        insert_vertical_seam(image, [0, 1])


def test_dp_finder_is_the_default():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[:, 2] = 255
    result = carve_width(image, 4)
    manual_energy = np.zeros((5, 5))
    manual = find_vertical_seam(manual_energy)
    assert result.image.shape == (5, 4, 3)
    assert len(manual) == 5
