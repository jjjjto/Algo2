"""Visualization helpers: seam overlays, energy maps, step images, GIFs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from PIL import Image

from .carve import remove_horizontal_seam, remove_vertical_seam


def _paint(image: np.ndarray, row: int, column: int, color: tuple[int, int, int]) -> None:
    if image.ndim == 2:
        image[row, column] = color[0]
    else:
        image[row, column] = color


def overlay_vertical_seam(
    image: np.ndarray,
    seam: list[int],
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Return a copy of the image with a vertical seam painted red."""
    out = image.copy()
    for row, column in enumerate(seam):
        _paint(out, row, column, color)
    return out


def overlay_horizontal_seam(
    image: np.ndarray,
    seam: list[int],
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Return a copy of the image with a horizontal seam painted red."""
    out = image.copy()
    for column, row in enumerate(seam):
        _paint(out, row, column, color)
    return out


def normalize_energy(energy: np.ndarray) -> np.ndarray:
    """Scale an energy map into [0, 1] for display."""
    low = float(energy.min())
    high = float(energy.max())
    if high - low < 1e-12:
        return np.zeros_like(energy, dtype=np.float64)
    return (energy - low) / (high - low)


def colorize_energy(energy: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """Turn an energy map into an RGB uint8 image."""
    rgba = colormaps[cmap](normalize_energy(energy))
    return (rgba[..., :3] * 255).astype(np.uint8)


def save_energy_image(energy: np.ndarray, path: Path) -> None:
    """Save a grayscale-normalized energy map."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((normalize_energy(energy) * 255).astype(np.uint8)).save(path)


def save_energy_figure(energy: np.ndarray, path: Path, title: str = "Energy map") -> None:
    """Save a colored energy map with a colorbar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    shown = ax.imshow(energy, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_steps(
    image: np.ndarray,
    seams: list[list[int]],
    directory: Path | None = None,
    horizontal: bool = False,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Save one overlay per seam and return the frames for a GIF.

    The seams must have been produced on consecutive versions of ``image``.
    Replaying them in order therefore reproduces the carving process.  By
    default every seam is saved; pass ``max_frames`` to limit the output.
    Pass ``directory=None`` to build the frames without writing step images.
    """
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    current = image.copy()
    frames = [current.copy()]
    overlay = overlay_horizontal_seam if horizontal else overlay_vertical_seam
    remove = remove_horizontal_seam if horizontal else remove_vertical_seam
    selected = seams if max_frames is None else seams[:max_frames]

    for index, seam in enumerate(selected, start=1):
        marked = overlay(current, seam)
        if directory is not None:
            Image.fromarray(marked).save(directory / f"seam_{index:03d}.png")
        frames.append(marked)
        current = remove(current, seam)

    if directory is not None:
        Image.fromarray(current).save(directory / "final.png")
    return frames


def save_gif(frames: list[np.ndarray], path: Path, duration: int = 250) -> None:
    """Save frames as an animated GIF, padding smaller frames on a white canvas."""
    if not frames:
        raise ValueError("cannot save a GIF without frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    padded: list[Image.Image] = []
    for frame in frames:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[: frame.shape[0], : frame.shape[1]] = frame[..., :3]
        padded.append(Image.fromarray(canvas))

    padded[0].save(
        path,
        save_all=True,
        append_images=padded[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def save_comparison_figure(
    images: list[np.ndarray],
    titles: list[str],
    path: Path,
    suptitle: str | None = None,
    figsize_per_image: float = 3.2,
) -> None:
    """Save a row of images with titles."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(figsize_per_image * len(images), 3.6),
    )
    axes = np.atleast_1d(axes)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_runtime_plot(
    sizes: list[int],
    seconds: list[float],
    path: Path,
    title: str = "Dynamic-programming carving time",
) -> None:
    """Save a simple runtime-versus-image-size plot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(sizes, seconds, marker="o")
    ax.set_xlabel("Image width (pixels, 4:3 aspect ratio)")
    ax.set_ylabel("Time for 20 seams (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
