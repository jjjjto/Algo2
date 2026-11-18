from itertools import product

import numpy as np
import pytest

from seam_carving.dp import (
    cumulative_energy,
    find_horizontal_seam,
    find_vertical_seam,
    seam_energy,
)


def is_valid_vertical(seam, height, width):
    return (
        len(seam) == height
        and all(0 <= column < width for column in seam)
        and all(abs(seam[i] - seam[i - 1]) <= 1 for i in range(1, height))
    )


def brute_force_vertical(energy):
    height, width = energy.shape
    best = None
    for columns in product(range(width), repeat=height):
        if any(abs(columns[i] - columns[i - 1]) > 1 for i in range(1, height)):
            continue
        cost = sum(energy[i, columns[i]] for i in range(height))
        if best is None or cost < best[0]:
            best = (cost, list(columns))
    return best


def test_seam_is_valid():
    rng = np.random.default_rng(0)
    energy = rng.random((10, 8)) * 100
    seam = find_vertical_seam(energy)
    assert is_valid_vertical(seam, 10, 8)


def test_dp_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(5):
        energy = rng.random((4, 4)) * 100
        best_cost, _ = brute_force_vertical(energy)
        seam = find_vertical_seam(energy)
        assert np.isclose(seam_energy(energy, seam), best_cost)


def test_dp_follows_a_clear_diagonal_path():
    energy = np.full((6, 6), 10.0)
    for i in range(6):
        energy[i, i] = 0.0
    assert find_vertical_seam(energy) == list(range(6))


def test_dp_can_use_the_last_column():
    energy = np.full((6, 5), 10.0)
    energy[:, -1] = 0.0
    assert find_vertical_seam(energy) == [4] * 6


def test_dp_can_use_the_first_column():
    energy = np.full((6, 5), 10.0)
    energy[:, 0] = 0.0
    assert find_vertical_seam(energy) == [0] * 6


def test_horizontal_seam_follows_a_low_row():
    energy = np.full((5, 6), 10.0)
    energy[2, :] = 0.0
    seam = find_horizontal_seam(energy)
    assert seam == [2] * 6


def test_cumulative_energy_shape_and_first_row():
    energy = np.arange(12, dtype=float).reshape(3, 4)
    total = cumulative_energy(energy)
    assert total.shape == energy.shape
    assert np.array_equal(total[0], energy[0])
    assert total[-1].min() == seam_energy(energy, find_vertical_seam(energy))


def test_seam_energy_and_length_check():
    energy = np.arange(12, dtype=float).reshape(3, 4)
    assert seam_energy(energy, [0, 1, 2]) == 0 + 5 + 10
    with pytest.raises(ValueError):
        seam_energy(energy, [0, 1])


def test_empty_energy_map():
    assert find_vertical_seam(np.empty((0, 0))) == []
