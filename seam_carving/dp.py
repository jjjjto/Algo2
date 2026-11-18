"""Dynamic-programming seam finding.

For a vertical seam, let ``total[i, j]`` be the minimum energy of a seam that
starts in row 0 and ends at pixel ``(i, j)``::

    total[0, j] = energy[0, j]
    total[i, j] = energy[i, j]
                  + min(total[i-1, j-1], total[i-1, j], total[i-1, j+1])

The best seam ends at the minimum value in the last row.  Parent pointers
record which of the three predecessors was chosen, so the seam can be rebuilt
by walking upwards.

Time O(H*W), space O(H*W).
"""

from __future__ import annotations

import numpy as np

INF = 1e18


def _cumulative_energy(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the DP table and the parent table used to rebuild a seam."""
    h, w = energy.shape
    total = np.empty((h, w), dtype=np.float64)
    parent = np.zeros((h, w), dtype=np.int8)  # -1 = left, 0 = up, +1 = right
    if h == 0 or w == 0:
        return total, parent

    total[0] = energy[0]
    for i in range(1, h):
        previous = total[i - 1]
        left = np.full(w, INF)
        left[1:] = previous[:-1]
        right = np.full(w, INF)
        right[:-1] = previous[1:]

        choices = np.stack((left, previous, right))
        total[i] = energy[i] + choices.min(axis=0)
        parent[i] = choices.argmin(axis=0) - 1

    return total, parent


def cumulative_energy(energy: np.ndarray) -> np.ndarray:
    """Return the DP table (used by report figures)."""
    return _cumulative_energy(energy)[0]


def find_vertical_seam(energy: np.ndarray) -> list[int]:
    """Return the minimum-energy vertical seam as one column per row."""
    h, w = energy.shape
    if h == 0 or w == 0:
        return []

    total, parent = _cumulative_energy(energy)
    j = int(np.argmin(total[-1]))
    seam = [j]
    for i in range(h - 1, 0, -1):
        j += int(parent[i, j])
        seam.append(j)
    seam.reverse()
    return seam


def find_horizontal_seam(energy: np.ndarray) -> list[int]:
    """Return the minimum-energy horizontal seam as one row per column.

    Transposing the energy map turns a horizontal seam into a vertical one,
    so the vertical DP is reused.
    """
    return find_vertical_seam(energy.T)


def seam_energy(energy: np.ndarray, seam: list[int]) -> float:
    """Return the sum of the energy values under a vertical seam."""
    if len(seam) != energy.shape[0]:
        raise ValueError("a vertical seam needs one column index per row")
    return float(sum(energy[i, j] for i, j in enumerate(seam)))
