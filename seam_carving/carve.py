"""Remove, insert, and iterate seams until the requested size is reached."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dp import find_vertical_seam, seam_energy
from .energy import sobel_energy
from .graphcut import find_vertical_seam_graphcut
from .greedy import find_vertical_seam_greedy

Finder = Callable[[np.ndarray], list[int]]

FINDERS: dict[str, Finder] = {
    "dp": find_vertical_seam,
    "greedy": find_vertical_seam_greedy,
    "graphcut": find_vertical_seam_graphcut,
}


@dataclass
class CarveResult:
    """The resized image, the seams that were used, and their energies."""

    image: np.ndarray
    seams: list[list[int]]
    energies: list[float]


def _transpose(image: np.ndarray) -> np.ndarray:
    """Swap the first two axes, keeping any color channels in place."""
    axes = (1, 0) + tuple(range(2, image.ndim))
    return image.transpose(axes)


def remove_vertical_seam(image: np.ndarray, seam: list[int]) -> np.ndarray:
    """Remove one pixel per row, following a vertical seam."""
    h, w = image.shape[:2]
    if len(seam) != h:
        raise ValueError("a vertical seam needs one column index per row")
    out = np.empty((h, w - 1) + image.shape[2:], dtype=image.dtype)
    for i, j in enumerate(seam):
        if not 0 <= j < w:
            raise ValueError(f"seam column {j} is outside the image")
        out[i, :j] = image[i, :j]
        out[i, j:] = image[i, j + 1:]
    return out


def remove_horizontal_seam(image: np.ndarray, seam: list[int]) -> np.ndarray:
    """Remove one pixel per column, following a horizontal seam."""
    return _transpose(remove_vertical_seam(_transpose(image), seam))


def insert_vertical_seam(image: np.ndarray, seam: list[int]) -> np.ndarray:
    """Insert one pixel per row next to a vertical seam (duplicate the pixel)."""
    h, w = image.shape[:2]
    if len(seam) != h:
        raise ValueError("a vertical seam needs one column index per row")
    out = np.empty((h, w + 1) + image.shape[2:], dtype=image.dtype)
    for i, j in enumerate(seam):
        if not 0 <= j < w:
            raise ValueError(f"seam column {j} is outside the image")
        out[i, : j + 1] = image[i, : j + 1]
        out[i, j + 1] = image[i, j]
        out[i, j + 2 :] = image[i, j + 1 :]
    return out


def insert_horizontal_seam(image: np.ndarray, seam: list[int]) -> np.ndarray:
    """Insert one pixel per column next to a horizontal seam."""
    return _transpose(insert_vertical_seam(_transpose(image), seam))


def carve_width(
    image: np.ndarray,
    target_width: int,
    finder: Finder = find_vertical_seam,
) -> CarveResult:
    """Remove vertical seams until the image is ``target_width`` wide.

    The energy map is recomputed after every removal, because removing one
    seam changes the neighbourhood of the remaining pixels.
    """
    if not 1 <= target_width <= image.shape[1]:
        raise ValueError("target_width must be between 1 and the current width")

    seams: list[list[int]] = []
    energies: list[float] = []
    while image.shape[1] > target_width:
        energy = sobel_energy(image)
        seam = finder(energy)
        seams.append(seam)
        energies.append(seam_energy(energy, seam))
        image = remove_vertical_seam(image, seam)
    return CarveResult(image=image, seams=seams, energies=energies)


def carve_height(
    image: np.ndarray,
    target_height: int,
    finder: Finder = find_vertical_seam,
) -> CarveResult:
    """Remove horizontal seams until the image is ``target_height`` tall.

    The image is transposed and the vertical pipeline is reused.  The returned
    seams are horizontal seams of the original image: one row per column.
    """
    if not 1 <= target_height <= image.shape[0]:
        raise ValueError("target_height must be between 1 and the current height")

    result = carve_width(_transpose(image), target_height, finder)
    return CarveResult(
        image=_transpose(result.image),
        seams=result.seams,
        energies=result.energies,
    )


def enlarge_width(
    image: np.ndarray,
    target_width: int,
    finder: Finder = find_vertical_seam,
) -> CarveResult:
    """Insert vertical seams until the image is ``target_width`` wide."""
    if target_width < image.shape[1]:
        raise ValueError("target_width must be at least the current width")

    seams: list[list[int]] = []
    energies: list[float] = []
    while image.shape[1] < target_width:
        energy = sobel_energy(image)
        seam = finder(energy)
        seams.append(seam)
        energies.append(seam_energy(energy, seam))
        image = insert_vertical_seam(image, seam)
    return CarveResult(image=image, seams=seams, energies=energies)


def enlarge_height(
    image: np.ndarray,
    target_height: int,
    finder: Finder = find_vertical_seam,
) -> CarveResult:
    """Insert horizontal seams until the image is ``target_height`` tall."""
    if target_height < image.shape[0]:
        raise ValueError("target_height must be at least the current height")

    result = enlarge_width(_transpose(image), target_height, finder)
    return CarveResult(
        image=_transpose(result.image),
        seams=result.seams,
        energies=result.energies,
    )


def resize(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    finder: Finder = find_vertical_seam,
) -> tuple[np.ndarray, CarveResult | None, CarveResult | None]:
    """Remove seams to reach a target size, width first and then height."""
    if target_width > image.shape[1]:
        raise ValueError("resize only removes seams; use enlarge_width to grow")
    if target_height > image.shape[0]:
        raise ValueError("resize only removes seams; use enlarge_height to grow")

    width_result = None
    if target_width < image.shape[1]:
        width_result = carve_width(image, target_width, finder)
        image = width_result.image

    height_result = None
    if target_height < image.shape[0]:
        height_result = carve_height(image, target_height, finder)
        image = height_result.image

    return image, width_result, height_result
