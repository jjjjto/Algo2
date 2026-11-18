from itertools import product

import numpy as np
import pytest

from seam_carving.dp import find_vertical_seam, seam_energy
from seam_carving.graphcut import (
    find_horizontal_seam_graphcut,
    find_vertical_seam_graphcut,
)


def is_valid_vertical(seam, height, width):
    return (
        len(seam) == height
        and all(0 <= column < width for column in seam)
        and all(abs(seam[i] - seam[i - 1]) <= 1 for i in range(1, height))
    )


def brute_force_vertical(energy):
    height, width = energy.shape
    best = float("inf")
    for columns in product(range(width), repeat=height):
        if any(abs(columns[i] - columns[i - 1]) > 1 for i in range(1, height)):
            continue
        best = min(best, sum(energy[i, columns[i]] for i in range(height)))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_graphcut_matches_dp_cost(seed):
    rng = np.random.default_rng(seed)
    energy = rng.random((6, 5)) * 100
    dynamic = find_vertical_seam(energy)
    cut = find_vertical_seam_graphcut(energy)
    assert is_valid_vertical(cut, 6, 5)
    assert np.isclose(seam_energy(energy, cut), seam_energy(energy, dynamic))


def test_graphcut_matches_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(3):
        energy = rng.random((4, 4)) * 100
        cut = find_vertical_seam_graphcut(energy)
        assert np.isclose(seam_energy(energy, cut), brute_force_vertical(energy))


def test_graphcut_handles_last_column():
    energy = np.full((5, 4), 10.0)
    energy[:, -1] = 0.0
    assert find_vertical_seam_graphcut(energy) == [3] * 5


def test_graphcut_handles_first_column():
    energy = np.full((5, 4), 10.0)
    energy[:, 0] = 0.0
    assert find_vertical_seam_graphcut(energy) == [0] * 5


def test_graphcut_horizontal_seam():
    energy = np.full((5, 6), 10.0)
    energy[3, :] = 0.0
    seam = find_horizontal_seam_graphcut(energy)
    assert seam == [3] * 6


def test_graphcut_rejects_negative_energy():
    with pytest.raises(ValueError):
        find_vertical_seam_graphcut(np.array([[-1.0, 2.0], [3.0, 4.0]]))


def test_graphcut_single_column():
    energy = np.ones((4, 1))
    assert find_vertical_seam_graphcut(energy) == [0, 0, 0, 0]
