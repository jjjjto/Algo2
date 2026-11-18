"""Generate deterministic sample images for the demo and the report.

Run from the repository root::

    python scripts/generate_assets.py

No network access is needed, so CI and the report figures are reproducible.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def _lerp(start: tuple[int, int, int], end: tuple[int, int, int], t: float):
    return tuple(int(a + (b - a) * t) for a, b in zip(start, end))


def make_scene(width: int = 320, height: int = 240) -> Image.Image:
    """A lighthouse on a coast: strong edges, a large sky, and a flat sea."""
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    horizon = int(height * 0.62)
    shore = int(height * 0.84)

    for y in range(height):
        if y < horizon:
            color = _lerp((110, 170, 235), (250, 225, 185), y / horizon)
        elif y < shore:
            color = _lerp((28, 92, 150), (70, 150, 190), (y - horizon) / (shore - horizon))
        else:
            color = _lerp((232, 205, 150), (245, 225, 180), (y - shore) / (height - shore))
        draw.line([(0, y), (width, y)], fill=color)

    sun_x, sun_y, sun_r = int(width * 0.18), int(height * 0.18), int(height * 0.075)
    draw.ellipse(
        [sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r],
        fill=(255, 226, 120),
    )

    for cx, cy, rx, ry in (
        (0.38, 0.14, 0.10, 0.035),
        (0.54, 0.10, 0.07, 0.028),
        (0.42, 0.24, 0.12, 0.030),
    ):
        draw.ellipse(
            [
                int(width * (cx - rx)),
                int(height * (cy - ry)),
                int(width * (cx + rx)),
                int(height * (cy + ry)),
            ],
            fill=(245, 248, 255),
        )

    draw.polygon(
        [
            (0, horizon),
            (int(width * 0.16), int(height * 0.40)),
            (int(width * 0.30), horizon),
        ],
        fill=(78, 92, 112),
    )
    draw.polygon(
        [
            (int(width * 0.22), horizon),
            (int(width * 0.42), int(height * 0.34)),
            (int(width * 0.60), horizon),
        ],
        fill=(96, 110, 130),
    )
    draw.polygon(
        [
            (int(width * 0.48), horizon),
            (int(width * 0.66), int(height * 0.43)),
            (int(width * 0.82), horizon),
        ],
        fill=(70, 84, 104),
    )

    for i in range(6):
        y = horizon + int((shore - horizon) * (i + 1) / 7)
        draw.line([(0, y), (width, y)], fill=(90, 170, 205), width=1)

    tower_x = int(width * 0.80)
    tower_base = shore + int(height * 0.06)
    tower_top = int(height * 0.34)
    half_base = int(width * 0.045)
    half_top = int(width * 0.026)

    draw.polygon(
        [
            (tower_x - half_base, tower_base),
            (tower_x + half_base, tower_base),
            (tower_x + half_top, tower_top),
            (tower_x - half_top, tower_top),
        ],
        fill=(245, 242, 232),
        outline=(40, 40, 45),
    )
    for start in (0.22, 0.52, 0.80):
        y0 = int(tower_top + (tower_base - tower_top) * start)
        y1 = int(tower_top + (tower_base - tower_top) * (start + 0.12))
        half0 = int(half_top + (half_base - half_top) * start)
        half1 = int(half_top + (half_base - half_top) * (start + 0.12))
        draw.polygon(
            [
                (tower_x - half0, y0),
                (tower_x + half0, y0),
                (tower_x + half1, y1),
                (tower_x - half1, y1),
            ],
            fill=(190, 55, 55),
        )

    lantern_w = int(width * 0.038)
    lantern_h = int(height * 0.055)
    draw.rectangle(
        [
            tower_x - lantern_w,
            tower_top - lantern_h,
            tower_x + lantern_w,
            tower_top,
        ],
        fill=(255, 235, 150),
        outline=(40, 40, 45),
    )
    draw.line(
        [(tower_x, tower_top - lantern_h), (tower_x, tower_top)],
        fill=(40, 40, 45),
        width=2,
    )
    draw.polygon(
        [
            (tower_x - lantern_w - 5, tower_top - lantern_h),
            (tower_x + lantern_w + 5, tower_top - lantern_h),
            (tower_x, tower_top - lantern_h - int(height * 0.045)),
        ],
        fill=(150, 45, 45),
        outline=(40, 40, 45),
    )
    draw.line(
        [(tower_x - lantern_w, tower_top - lantern_h - 2), (tower_x - width, tower_top - 2)],
        fill=(255, 245, 190),
        width=3,
    )
    draw.line(
        [(tower_x + lantern_w, tower_top - lantern_h - 2), (tower_x + width, tower_top - 2)],
        fill=(255, 245, 190),
        width=3,
    )

    return image


def make_shapes(width: int = 320, height: int = 200) -> Image.Image:
    """A gradient with simple geometric objects."""
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        color = _lerp((35, 45, 95), (120, 40, 120), y / max(1, height - 1))
        draw.line([(0, y), (width, y)], fill=color)

    draw.ellipse(
        [int(width * 0.08), int(height * 0.12), int(width * 0.30), int(height * 0.58)],
        fill=(245, 200, 60),
    )
    draw.rectangle(
        [int(width * 0.36), int(height * 0.48), int(width * 0.52), int(height * 0.80)],
        fill=(60, 190, 120),
    )
    draw.polygon(
        [
            (int(width * 0.58), int(height * 0.22)),
            (int(width * 0.76), int(height * 0.78)),
            (int(width * 0.42), int(height * 0.78)),
        ],
        fill=(220, 70, 90),
    )
    draw.ellipse(
        [int(width * 0.80), int(height * 0.30), int(width * 0.93), int(height * 0.52)],
        outline=(240, 240, 255),
        width=6,
    )
    for y in range(int(height * 0.10), int(height * 0.90), 12):
        draw.line(
            [(int(width * 0.66), y), (int(width * 0.69), y + 5)],
            fill=(250, 250, 250),
            width=3,
        )
    return image


def make_portrait(width: int = 240, height: int = 300) -> Image.Image:
    """A stylized face: useful for checking that important content survives."""
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        color = _lerp((235, 220, 205), (190, 200, 220), y / max(1, height - 1))
        draw.line([(0, y), (width, y)], fill=color)

    draw.ellipse(
        [int(width * 0.16), int(height * 0.18), int(width * 0.84), int(height * 0.88)],
        fill=(224, 176, 140),
        outline=(90, 60, 45),
        width=3,
    )
    draw.polygon(
        [
            (int(width * 0.16), int(height * 0.36)),
            (int(width * 0.18), int(height * 0.17)),
            (int(width * 0.50), int(height * 0.10)),
            (int(width * 0.82), int(height * 0.17)),
            (int(width * 0.84), int(height * 0.40)),
            (int(width * 0.72), int(height * 0.28)),
            (int(width * 0.50), int(height * 0.24)),
            (int(width * 0.28), int(height * 0.28)),
        ],
        fill=(55, 38, 32),
    )
    for eye_x in (0.36, 0.64):
        draw.ellipse(
            [
                int(width * (eye_x - 0.075)),
                int(height * 0.47),
                int(width * (eye_x + 0.075)),
                int(height * 0.54),
            ],
            fill=(250, 250, 250),
            outline=(50, 40, 40),
            width=2,
        )
        draw.ellipse(
            [
                int(width * (eye_x - 0.022)),
                int(height * 0.485),
                int(width * (eye_x + 0.022)),
                int(height * 0.525),
            ],
            fill=(40, 35, 35),
        )
    draw.arc(
        [int(width * 0.38), int(height * 0.62), int(width * 0.62), int(height * 0.76)],
        start=10,
        end=170,
        fill=(120, 45, 45),
        width=4,
    )
    return image


def make_gradient(width: int = 320, height: int = 200) -> Image.Image:
    """A smooth gradient with one bright vertical bar."""
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for x in range(width):
        color = _lerp((20, 20, 40), (220, 220, 240), x / max(1, width - 1))
        draw.line([(x, 0), (x, height)], fill=color)
    bar_x = int(width * 0.62)
    draw.rectangle([bar_x, 0, bar_x + int(width * 0.05), height], fill=(255, 80, 60))
    return image


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "scene.png": make_scene(320, 240),
        "scene_small.png": make_scene(96, 72),
        "shapes.png": make_shapes(320, 200),
        "portrait.png": make_portrait(240, 300),
        "gradient.png": make_gradient(320, 200),
    }
    for name, image in outputs.items():
        path = ASSETS / name
        image.save(path)
        print(f"wrote {path.relative_to(ROOT)} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
