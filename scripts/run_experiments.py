"""Run the experiments used in the report and regenerate all figures.

Run from the repository root::

    python scripts/run_experiments.py

The script writes ``report/figures/*`` and ``report/results.md``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seam_carving import (
    FINDERS,
    carve_height,
    carve_width,
    cumulative_energy,
    sobel_energy,
)
from seam_carving.visualize import (
    overlay_horizontal_seam,
    save_comparison_figure,
    save_energy_figure,
    save_gif,
    save_runtime_plot,
    save_steps,
)

ASSETS = ROOT / "assets"
FIGURES = ROOT / "report" / "figures"
RESULTS = ROOT / "report" / "results.md"


def load(name: str) -> np.ndarray:
    with Image.open(ASSETS / name) as image:
        return np.array(image.convert("RGB"))


def timed_carve(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start


def row(method: str, result, elapsed: float) -> str:
    return (
        f"| {method} | {result.image.shape[1]}x{result.image.shape[0]} "
        f"| {len(result.seams)} "
        f"| {sum(result.energies):.1f} "
        f"| {result.energies[0]:.1f} "
        f"| {elapsed:.3f} |"
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    original = load("scene.png")
    small = load("scene_small.png")

    # 1. Energy map and DP table.
    energy = sobel_energy(original)
    save_energy_figure(energy, FIGURES / "energy.png", "Sobel energy map")
    save_energy_figure(
        cumulative_energy(energy),
        FIGURES / "cumulative.png",
        "Cumulative energy (DP table)",
    )

    # 2. Large comparison: DP vs greedy, crop, and scale.
    target_width = 240
    dp_result, dp_time = timed_carve(carve_width, original, target_width, FINDERS["dp"])
    greedy_result, greedy_time = timed_carve(
        carve_width, original, target_width, FINDERS["greedy"]
    )
    cropped = original[:, :target_width]
    scaled = np.array(
        Image.fromarray(original).resize(
            (target_width, original.shape[0]), Image.Resampling.LANCZOS
        )
    )
    Image.fromarray(original).save(FIGURES / "original_scene.png")
    Image.fromarray(dp_result.image).save(FIGURES / "result_dp_240x240.png")
    Image.fromarray(greedy_result.image).save(FIGURES / "result_greedy_240x240.png")
    save_comparison_figure(
        [
            original,
            dp_result.image,
            greedy_result.image,
            cropped,
            scaled,
        ],
        [
            "Original 320x240",
            "Dynamic programming",
            "Greedy",
            "Crop (left 240 px)",
            "Scale to 240x240",
        ],
        FIGURES / "comparison_large.png",
        suptitle="Content-aware resizing vs cropping and scaling",
        figsize_per_image=3.1,
    )

    # 3. Step-by-step seam visualization and animated GIF.
    frames = save_steps(
        original,
        dp_result.seams,
        FIGURES / "steps",
        max_frames=40,
    )
    save_comparison_figure(
        frames[1:7],
        [f"Removed seam {index}" for index in range(1, 7)],
        FIGURES / "seams_first_6.png",
        suptitle="Step-by-step seam removal (first six seams)",
        figsize_per_image=2.6,
    )
    save_gif(frames, FIGURES / "carving.gif", duration=150)

    # 4. Three-way comparison on the small image so graph cut is practical.
    small_target = 80
    small_dp, small_dp_time = timed_carve(
        carve_width, small, small_target, FINDERS["dp"]
    )
    small_greedy, small_greedy_time = timed_carve(
        carve_width, small, small_target, FINDERS["greedy"]
    )
    small_cut, small_cut_time = timed_carve(
        carve_width, small, small_target, FINDERS["graphcut"]
    )
    Image.fromarray(small_cut.image).save(FIGURES / "result_graphcut_80x72.png")
    save_comparison_figure(
        [small, small_dp.image, small_greedy.image, small_cut.image],
        [
            "Original 96x72",
            "Dynamic programming",
            "Greedy",
            "Graph cut",
        ],
        FIGURES / "comparison_methods_small.png",
        suptitle="Same seams, three algorithms (remove 16 columns)",
        figsize_per_image=3.0,
    )
    graphcut_matches_dp = abs(sum(small_dp.energies) - sum(small_cut.energies)) < 1e-6

    # 5. Horizontal seams.
    horizontal_result, horizontal_time = timed_carve(
        carve_height, original, 200, FINDERS["dp"]
    )
    first_horizontal = horizontal_result.seams[0]
    Image.fromarray(horizontal_result.image).save(
        FIGURES / "result_horizontal_320x200.png"
    )
    save_comparison_figure(
        [
            original,
            overlay_horizontal_seam(original, first_horizontal),
            horizontal_result.image,
        ],
        [
            "Original",
            "First horizontal seam",
            "DP horizontal result (200 px tall)",
        ],
        FIGURES / "horizontal.png",
        suptitle="Horizontal seam removal",
        figsize_per_image=3.4,
    )

    # 6. Runtime scaling for the DP implementation.
    sizes: list[int] = []
    seconds: list[float] = []
    scalability_rows: list[str] = []
    for width in (160, 240, 320, 400, 480):
        height = int(width * 3 / 4)
        resized = np.array(
            Image.fromarray(original).resize((width, height), Image.Resampling.LANCZOS)
        )
        _, elapsed = timed_carve(carve_width, resized, width - 20, FINDERS["dp"])
        sizes.append(width)
        seconds.append(elapsed)
        scalability_rows.append(
            f"| {width}x{height} | {width * height} | 20 | {elapsed:.3f} |"
        )
    save_runtime_plot(sizes, seconds, FIGURES / "runtime.png")

    # 7. Results tables.
    lines = [
        "# Experimental results",
        "",
        "Generated by `scripts/run_experiments.py`.",
        "Times are single-run wall-clock measurements on Windows 11, Python 3.12,",
        "with NumPy and Pillow installed.",
        "",
        "## Large image: dynamic programming vs greedy",
        "",
        f"Original: 320x240. Target width: {target_width}. Removed"
        f" {len(dp_result.seams)} seams.",
        "",
        "| Method | Output | Seams | Total seam energy | First seam energy | Time (s) |",
        "|---|---|---|---|---|---|",
        row("DP", dp_result, dp_time),
        row("Greedy", greedy_result, greedy_time),
        "",
        "## Three-way comparison on a small image",
        "",
        f"Original: 96x72. Target width: {small_target}. Removed"
        f" {len(small_dp.seams)} seams.",
        "",
        "| Method | Output | Seams | Total seam energy | First seam energy | Time (s) |",
        "|---|---|---|---|---|---|",
        row("DP", small_dp, small_dp_time),
        row("Greedy", small_greedy, small_greedy_time),
        row("Graph cut", small_cut, small_cut_time),
        "",
        "Graph-cut total energy equals DP total energy: "
        f"{'yes' if graphcut_matches_dp else 'NO'}.",
        "This is the expected result because both find a minimum-energy seam",
        "on every iteration.",
        "",
        "## Horizontal carving",
        "",
        f"Original: 320x240. Target height: 200. Removed"
        f" {len(horizontal_result.seams)} horizontal seams in"
        f" {horizontal_time:.3f}s.",
        "",
        "## DP runtime scaling (20 seams)",
        "",
        "| Size | Pixels | Seams | Time (s) |",
        "|---|---|---|---|",
        *scalability_rows,
        "",
        "## Reproducing",
        "",
        "```bash",
        "python scripts/generate_assets.py",
        "python scripts/run_experiments.py",
        "python -m pytest -q",
        "```",
        "",
        "## Verification",
        "",
        "- `tests/test_dp.py` compares DP against brute-force enumeration on 4x4 maps.",
        "- `tests/test_graphcut.py` compares graph cut against DP and brute force,",
        "  including seams on the first and last columns.",
        "- `tests/test_cli.py` runs the command-line tool end-to-end.",
        "- `tests/test_carve.py` checks seam removal, insertion, and target sizes.",
        "",
    ]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {RESULTS.relative_to(ROOT)}")
    print(f"wrote figures to {FIGURES.relative_to(ROOT)}")
    print(
        "DP vs greedy: "
        f"{dp_time:.3f}s vs {greedy_time:.3f}s; "
        f"energy {sum(dp_result.energies):.1f} vs {sum(greedy_result.energies):.1f}"
    )
    print(
        "Small image: "
        f"DP {small_dp_time:.3f}s, greedy {small_greedy_time:.3f}s, "
        f"graph cut {small_cut_time:.3f}s"
    )
    print(f"Graph cut matches DP energy: {graphcut_matches_dp}")


if __name__ == "__main__":
    main()
