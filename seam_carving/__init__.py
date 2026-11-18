"""Seam carving: content-aware image resizing with dynamic programming.

The package exposes one seam finder per algorithm:

* ``find_vertical_seam`` / ``find_horizontal_seam`` - dynamic programming (optimal)
* ``find_vertical_seam_greedy`` / ``find_horizontal_seam_greedy`` - greedy
* ``find_vertical_seam_graphcut`` / ``find_horizontal_seam_graphcut`` - min-cut / max-flow

All vertical finders take an energy map of shape ``(H, W)`` and return a list
of ``H`` column indices.  A horizontal seam is a list of ``W`` row indices.
"""

from .carve import (
    FINDERS,
    CarveResult,
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
from .dp import cumulative_energy, find_horizontal_seam, find_vertical_seam, seam_energy
from .energy import sobel_energy, to_grayscale
from .graphcut import find_horizontal_seam_graphcut, find_vertical_seam_graphcut
from .greedy import find_horizontal_seam_greedy, find_vertical_seam_greedy

__all__ = [
    "FINDERS",
    "CarveResult",
    "carve_height",
    "carve_width",
    "cumulative_energy",
    "enlarge_height",
    "enlarge_width",
    "find_horizontal_seam",
    "find_horizontal_seam_graphcut",
    "find_horizontal_seam_greedy",
    "find_vertical_seam",
    "find_vertical_seam_graphcut",
    "find_vertical_seam_greedy",
    "insert_horizontal_seam",
    "insert_vertical_seam",
    "remove_horizontal_seam",
    "remove_vertical_seam",
    "resize",
    "seam_energy",
    "sobel_energy",
    "to_grayscale",
]
