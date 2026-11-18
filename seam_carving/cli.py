"""Command-line interface.

Examples::

    python -m seam_carving.cli assets/scene.png out/scene_small.png --width 260
    python -m seam_carving.cli assets/scene.png out/demo.png --width 260 --height 180 --method greedy
    python -m seam_carving.cli assets/scene.png out/demo.png --width 260 --seams-dir out/steps --gif out/steps.gif
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .carve import FINDERS, carve_height, carve_width
from .energy import sobel_energy
from .visualize import save_energy_image, save_gif, save_steps


def load_image(path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 array."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def save_image(image: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Content-aware image resizing with seam carving."
    )
    parser.add_argument("input", type=Path, help="input image")
    parser.add_argument("output", type=Path, help="output image")
    parser.add_argument("--width", type=int, help="target width (must be smaller)")
    parser.add_argument("--height", type=int, help="target height (must be smaller)")
    parser.add_argument(
        "--method",
        choices=sorted(FINDERS),
        default="dp",
        help="seam finder: dp (optimal), greedy, or graphcut",
    )
    parser.add_argument(
        "--seams-dir",
        type=Path,
        help="save one image per removed seam in this directory",
    )
    parser.add_argument("--gif", type=Path, help="save an animated GIF of the removal")
    parser.add_argument("--energy", type=Path, help="save the initial energy map")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original = load_image(args.input)
    target_width = args.width if args.width is not None else original.shape[1]
    target_height = args.height if args.height is not None else original.shape[0]

    if target_width > original.shape[1]:
        raise SystemExit("this tool removes seams; target width is larger than the input")
    if target_height > original.shape[0]:
        raise SystemExit("this tool removes seams; target height is larger than the input")
    if args.energy is not None:
        save_energy_image(sobel_energy(original), args.energy)

    finder = FINDERS[args.method]
    start = time.perf_counter()

    image = original
    vertical = None
    horizontal = None
    if target_width < image.shape[1]:
        vertical = carve_width(image, target_width, finder)
        image = vertical.image
    if target_height < image.shape[0]:
        horizontal = carve_height(image, target_height, finder)
        image = horizontal.image

    elapsed = time.perf_counter() - start
    save_image(image, args.output)

    frames: list[np.ndarray] = []
    want_steps = args.seams_dir is not None
    want_gif = args.gif is not None
    if want_steps or want_gif:
        base = original
        if vertical is not None:
            vertical_dir = args.seams_dir / "vertical" if want_steps else None
            frames.extend(save_steps(base, vertical.seams, vertical_dir))
            base = vertical.image
        if horizontal is not None:
            horizontal_dir = args.seams_dir / "horizontal" if want_steps else None
            horizontal_frames = save_steps(
                base,
                horizontal.seams,
                horizontal_dir,
                horizontal=True,
            )
            frames.extend(horizontal_frames if not frames else horizontal_frames[1:])
    if want_gif and frames:
        save_gif(frames, args.gif, duration=120)

    removed_vertical = 0 if vertical is None else original.shape[1] - vertical.image.shape[1]
    height_before_horizontal = (
        vertical.image.shape[0] if vertical is not None else original.shape[0]
    )
    removed_horizontal = (
        0
        if horizontal is None
        else height_before_horizontal - horizontal.image.shape[0]
    )

    print(f"Input:  {original.shape[1]}x{original.shape[0]}")
    print(f"Method: {args.method}")
    print(
        f"Removed {removed_vertical} vertical and "
        f"{removed_horizontal} horizontal seam(s)"
    )
    print(f"Output: {image.shape[1]}x{image.shape[0]}  ({args.output})")
    print(f"Time:   {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
