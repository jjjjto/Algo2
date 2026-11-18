import numpy as np
from PIL import Image

from seam_carving.dp import find_vertical_seam
from seam_carving.energy import sobel_energy
from seam_carving.visualize import (
    colorize_energy,
    normalize_energy,
    overlay_vertical_seam,
    save_comparison_figure,
    save_energy_figure,
    save_energy_image,
    save_gif,
    save_runtime_plot,
    save_steps,
)


def test_overlay_paints_only_the_seam():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    seam = [0, 1, 2, 3]
    marked = overlay_vertical_seam(image, seam)
    assert marked.shape == image.shape
    for row, column in enumerate(seam):
        assert tuple(marked[row, column]) == (255, 0, 0)
    assert np.count_nonzero(marked.sum(axis=2)) == 4


def test_normalize_constant_energy():
    energy = np.full((3, 4), 7.0)
    assert np.allclose(normalize_energy(energy), 0.0)
    assert colorize_energy(energy).shape == (3, 4, 3)


def test_save_energy_outputs(tmp_path):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[:, 4:] = 255
    energy = sobel_energy(image)
    grey = tmp_path / "energy.png"
    figure = tmp_path / "energy_figure.png"
    save_energy_image(energy, grey)
    save_energy_figure(energy, figure)
    assert grey.exists() and grey.stat().st_size > 0
    assert figure.exists() and figure.stat().st_size > 0
    with Image.open(grey) as opened:
        assert opened.size == (8, 6)


def test_save_steps_and_gif(tmp_path):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, 5:] = 255
    seam = find_vertical_seam(sobel_energy(image))
    frames = save_steps(image, [seam], tmp_path / "steps")
    assert len(frames) == 2
    assert (tmp_path / "steps" / "seam_001.png").exists()
    assert (tmp_path / "steps" / "final.png").exists()

    gif = tmp_path / "carving.gif"
    save_gif(frames, gif, duration=100)
    assert gif.exists() and gif.stat().st_size > 0
    with Image.open(gif) as opened:
        assert opened.n_frames == 2


def test_save_comparison_and_runtime_plots(tmp_path):
    first = np.zeros((6, 8, 3), dtype=np.uint8)
    second = np.full((6, 8, 3), 128, dtype=np.uint8)
    comparison = tmp_path / "comparison.png"
    save_comparison_figure([first, second], ["first", "second"], comparison)
    assert comparison.exists() and comparison.stat().st_size > 0

    plot = tmp_path / "runtime.png"
    save_runtime_plot([100, 200, 300], [0.1, 0.2, 0.3], plot)
    assert plot.exists() and plot.stat().st_size > 0
