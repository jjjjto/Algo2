import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def make_image(path: Path) -> None:
    height, width = 20, 30
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    image[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    Image.fromarray(image).save(path)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "seam_carving.cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_cli_resizes_width_and_height(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    make_image(source)

    result = run_cli(str(source), str(output), "--width", "25", "--height", "18")
    assert result.returncode == 0, result.stderr
    assert "Output: 25x18" in result.stdout
    with Image.open(output) as image:
        assert image.size == (25, 18)


def test_cli_writes_seam_steps_and_gif(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    steps = tmp_path / "steps"
    gif = tmp_path / "steps.gif"
    make_image(source)

    result = run_cli(
        str(source),
        str(output),
        "--width",
        "27",
        "--seams-dir",
        str(steps),
        "--gif",
        str(gif),
    )
    assert result.returncode == 0, result.stderr
    assert (steps / "vertical" / "seam_001.png").exists()
    assert (steps / "vertical" / "seam_002.png").exists()
    assert (steps / "vertical" / "final.png").exists()
    assert gif.exists() and gif.stat().st_size > 0


def test_cli_can_make_a_gif_without_saving_steps(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    gif = tmp_path / "only.gif"
    make_image(source)

    result = run_cli(
        str(source),
        str(output),
        "--width",
        "27",
        "--gif",
        str(gif),
    )
    assert result.returncode == 0, result.stderr
    assert gif.exists() and gif.stat().st_size > 0
    assert not (tmp_path / "steps").exists()


def test_cli_rejects_growing(tmp_path):
    source = tmp_path / "input.png"
    make_image(source)
    result = run_cli(str(source), str(tmp_path / "out.png"), "--width", "40")
    assert result.returncode != 0
    assert "larger than the input" in result.stderr
