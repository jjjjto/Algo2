"""Greedy seam finding, implemented for comparison with dynamic programming.

A greedy seam is built row by row.  At each row it moves to the lowest-energy
pixel among the three reachable neighbours.  It never looks at future rows, so
a cheap choice now can force an expensive choice later.
"""

from __future__ import annotations

import numpy as np


def find_vertical_seam_greedy(energy: np.ndarray) -> list[int]:
    """Return a greedy vertical seam as one column per row.

    Time O(H*W) in the worst case, space O(H).  The result is a valid seam but
    it is not guaranteed to have minimum energy.
    """
    h, w = energy.shape
    if h == 0 or w == 0:
        return []

    seam = [int(np.argmin(energy[0]))]
    for i in range(1, h):
        j = seam[-1]
        candidates = [column for column in (j - 1, j, j + 1) if 0 <= column < w]
        seam.append(min(candidates, key=lambda column: energy[i, column]))
    return seam


def find_horizontal_seam_greedy(energy: np.ndarray) -> list[int]:
    """Return a greedy horizontal seam as one row per column."""
    return find_vertical_seam_greedy(energy.T)
