import numpy as np

from seam_carving.dp import find_horizontal_seam, find_vertical_seam, seam_energy
from seam_carving.greedy import (
    find_horizontal_seam_greedy,
    find_vertical_seam_greedy,
)


def is_valid_vertical(seam, height, width):
    return (
        len(seam) == height
        and all(0 <= column < width for column in seam)
        and all(abs(seam[i] - seam[i - 1]) <= 1 for i in range(1, height))
    )


def test_greedy_seam_is_valid():
    rng = np.random.default_rng(2)
    energy = rng.random((9, 7)) * 100
    seam = find_vertical_seam_greedy(energy)
    assert is_valid_vertical(seam, 9, 7)


def test_greedy_can_be_suboptimal():
    energy = np.array(
        [
            [0.0, 1.0, 100.0],
            [100.0, 50.0, 0.0],
            [100.0, 100.0, 0.0],
        ]
    )
    greedy = find_vertical_seam_greedy(energy)
    optimal = find_vertical_seam(energy)
    assert seam_energy(energy, greedy) > seam_energy(energy, optimal)
    assert seam_energy(energy, greedy) == 50.0
    assert seam_energy(energy, optimal) == 1.0


def test_horizontal_greedy_is_valid():
    rng = np.random.default_rng(3)
    energy = rng.random((7, 9)) * 100
    seam = find_horizontal_seam_greedy(energy)
    assert len(seam) == 9
    assert all(0 <= row < 7 for row in seam)
    assert all(abs(seam[j] - seam[j - 1]) <= 1 for j in range(1, 9))


def test_horizontal_dp_beats_greedy_on_a_trap():
    energy = np.array(
        [
            [0.0, 1.0, 100.0],
            [100.0, 50.0, 0.0],
            [100.0, 100.0, 0.0],
        ]
    ).T
    greedy = find_horizontal_seam_greedy(energy)
    optimal = find_horizontal_seam(energy)
    greedy_cost = sum(energy[greedy[j], j] for j in range(energy.shape[1]))
    optimal_cost = sum(energy[optimal[j], j] for j in range(energy.shape[1]))
    assert greedy_cost > optimal_cost
