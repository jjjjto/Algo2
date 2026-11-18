# Seam Carving

Content-aware image resizing with dynamic programming, a greedy baseline, and
a graph-cut (min-cut / max-flow) solver. This is the implementation for
Assignment 2 (Image Compression by Seam Carving).

## Features

- Sobel energy map on grayscale luminance.
- Optimal vertical and horizontal seams with dynamic programming.
- Greedy seam finding for comparison.
- Graph-cut seam finding with Dinic max-flow.
- Seam removal, seam insertion, and target-size carving.
- Command-line tool, step-by-step visualisation, animated GIF, and a Tkinter
  graphical interface.
- 52 pytest tests, including brute-force optimality checks and DP vs graph-cut
  cross-checks.
- GitHub Actions CI on Python 3.10, 3.11, and 3.12.
- Reproducible report figures and result tables.

No content-aware resizing function from an image library is used. NumPy
handles arrays, Pillow reads and writes images, and Matplotlib draws the
report figures.

## Quick start

```bash
python -m pip install -r requirements.txt

# Shrink to width 240 with dynamic programming
python -m seam_carving.cli assets/scene.png out/scene_240.png --width 240

# Greedy comparison
python -m seam_carving.cli assets/scene.png out/scene_greedy.png --width 240 --method greedy

# Save the energy map, one image per seam, and an animated GIF
python -m seam_carving.cli assets/scene.png out/demo.png --width 260 \
    --energy out/energy.png --seams-dir out/steps --gif out/carving.gif

# Interactive application
python -m seam_carving.gui
```

The CLI accepts:

| Option | Meaning |
|---|---|
| `--width N` | Target width (must be smaller than the input) |
| `--height N` | Target height (must be smaller than the input) |
| `--method dp\|greedy\|graphcut` | Seam finder (default `dp`) |
| `--seams-dir DIR` | Save one overlay per removed seam |
| `--gif PATH` | Save an animated GIF of the removal |
| `--energy PATH` | Save the initial Sobel energy map |

The GUI supports shrinking, inserting one seam at a time, method selection,
reset, save, and a checkbox that shows the first removed seam.

## Tests

```bash
python -m pytest -q
```

The suite covers:

- Sobel energy on constant images, edges, and grayscale input.
- DP against brute-force enumeration on small maps.
- Graph cut against DP and brute force, including first-column and
  last-column seams.
- The greedy optimality counterexample.
- Seam removal, insertion, target sizes, and validation errors.
- The CLI end-to-end, including step images and the GIF.
- Visualisation outputs and a GUI smoke test.

## Reproduce the report

```bash
python scripts/generate_assets.py
python scripts/run_experiments.py
```

The second command writes `report/results.md` and the figures under
`report/figures/`.

## Documentation

- [Report (Markdown)](report/report.md)
- [Report (DOCX)](report/report.docx)
- [Slides (PPTX)](slides/seam_carving_slides.pptx)
- [Experimental results](report/results.md)
- [Quiz study guide](QUIZ_GUIDE.md)
- [Presentation outline](slides/outline.md)
- [Presentation script](slides/presentation_script.md)

## Build the submission documents

```bash
python -m pip install -r requirements-docs.txt
python scripts/build_documents.py
```

This regenerates `report/report.docx` from the Markdown report and
`slides/seam_carving_slides.pptx` from the slide content. The DOCX contains
the figures and tables; the PPTX contains 11 slides with speaker notes.

## Project layout

```
seam_carving/     implementation
tests/            pytest suite
scripts/          asset generator and experiment runner
assets/           deterministic sample images
report/           report, results, and figures
slides/           presentation outline and script
.github/          CI workflow
```

## Complexity

For `k` seams removed from an `H x W` image:

| Method | Time | Space |
|---|---|---|
| DP | `O(k*H*W)` | `O(H*W)` |
| Greedy | `O(k*H*W)` | `O(H*W)` |
| Graph cut | `O(k * max-flow)` | `O(H*W)` |

The energy map is recomputed after every removal, which is why the `k` factor
is outside the per-seam cost.

## Links

- **GitHub:** https://github.com/jjjjto/Algo2
- **Presentation video:** TODO: add the recorded video URL before submission.
