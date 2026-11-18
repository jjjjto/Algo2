"""Energy map computation.

The energy of a pixel measures how much visual disruption removing it would
cause.  Edges have high energy, flat regions have low energy.  Seam carving
removes pixels with low energy first.
"""

from __future__ import annotations

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to float64 grayscale.

    RGB images use the standard luminance weights (0.299 R + 0.587 G + 0.114 B).
    A 2-D input is returned as float64 without copying its channels.
    """
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("expected a grayscale (H, W) or RGB (H, W, 3) image")
    rgb = image[..., :3].astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def sobel_energy(image: np.ndarray) -> np.ndarray:
    """Return the Sobel gradient magnitude of every pixel.

    The Sobel kernels are::

        Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
              [-2, 0, 2],           [ 0,  0,  0],
              [-1, 0, 1]]           [ 1,  2,  1]]

    ``sqrt(Gx**2 + Gy**2)`` is the energy.  The border is padded by edge
    replication so the output has the same shape as the input.

    Time O(H*W), space O(H*W).
    """
    gray = to_grayscale(image)
    padded = np.pad(gray, 1, mode="edge")

    gx = (
        padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    )
    return np.hypot(gx, gy)
