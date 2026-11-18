# Presentation Script

Each member has about two minutes. The wording below is a guide, not a
strict script. Adapt it to your voice and keep an eye on the clock.

---

## Member 1: Problem and energy (slides 1-3, 2 minutes)

### Slide 1: Title (15 seconds)

"Hi, we are Team T23. Our project is content-aware image resizing by seam
carving. We implemented the seam finder three ways: dynamic programming,
greedy, and graph cut. We also built a CLI, an interactive app, a test suite,
and report figures."

### Slide 2: Why seam carving? (45 seconds)

"Cropping is fast, but it can delete important content. Scaling keeps
everything, but it squashes object shapes. Seam carving is content-aware: it
finds connected paths of low-energy pixels and removes them, so edges and
objects stay where they are.

The loop is simple: compute an energy map, find the minimum-energy seam,
remove it, and repeat.

On the comparison image, the crop loses the lighthouse on the right, the
scaled image is squashed, and the seam-carved image keeps the lighthouse and
the horizon."

### Slide 3: Energy map (1 minute)

"We compute energy with the Sobel operator. First we convert to grayscale.
Then we convolve with the two Sobel kernels, one for horizontal changes and
one for vertical changes. The energy is the gradient magnitude: square root of
Gx squared plus Gy squared.

High energy means a strong edge. Low energy means a flat region such as sky or
water. The seam finder avoids the high-energy pixels.

The energy map takes O(H times W) time and space. We pad the border by edge
replication, so the output has the same size as the input."

---

## Member 2: DP and greedy (slides 4-5, 2 minutes)

### Slide 4: Dynamic programming (1 minute)

"A vertical seam has one pixel per row, and adjacent seam pixels are in the
same or an adjacent column.

We define total of i,j as the cheapest seam that starts in the top row and
ends at pixel i,j. The base case is the top row. For every other row, the seam
must come from the up-left, up, or up-right pixel, so we add the pixel energy
to the minimum of those three values.

The best seam ends at the smallest value in the last row. We store which
predecessor was chosen in a parent table, then walk upwards to rebuild the
seam.

This works because of optimal substructure and overlapping subproblems. The
cost is O(H times W) time and space per seam. For k seams the total is
O(k times H times W), because the energy map is recomputed after every
removal."

### Slide 5: Greedy and the comparison (1 minute)

"The greedy version starts at the cheapest pixel in the top row. Each row it
picks the cheapest of the three reachable columns. It never looks ahead.

That is the problem: a cheap pixel now can force an expensive pixel later. We
have a 3x3 counterexample in the tests: greedy costs 50, DP costs 1.

On the 320 by 240 scene, removing 80 columns:

- DP took 0.680 seconds and its total seam energy was 392,678.
- Greedy took 0.217 seconds and its total seam energy was 469,628.

Greedy is about 3.1 times faster, but it removes more disruptive pixels. DP is
the better default."

---

## Member 3: Graph cut and demo (slides 6-7, 2 minutes)

### Slide 6: Graph cut (1 minute)

"We can also write seam finding as a minimum cut. We create one node per
pixel, plus a source and a sink. The source connects to the first column with
infinite capacity. Each pixel connects to its right neighbour with capacity
equal to its energy. The last column connects to the sink with the energy of
the last pixel.

Infinite backward edges force the source side of the cut to be a prefix in
every row. Infinite diagonal edges force the prefix length to change by at
most one between adjacent rows. So the boundary is a valid seam, and the
rightmost source-side pixel in each row is the seam pixel.

The minimum cut equals the maximum flow. We use Dinic's algorithm: BFS builds
levels, and DFS sends a blocking flow.

One correction to the assignment sketch: the last column to the sink uses the
pixel's energy, not infinity. If it were infinite, a seam could never use the
last column.

On the 96 by 72 image, removing 16 columns: DP takes 0.028 seconds and graph
cut takes 17.207 seconds, but both have the same total energy, 51,472.5.
Same quality, much slower."

### Slide 7: Demo (1 minute)

"Let me show the app. I open the scene, choose DP, set the width to 240, and
apply. The lighthouse stays, and the sky and sea are compressed.

Now I reset and choose greedy. The seams are different and the total energy is
higher.

Here are the first six removed seams and the animated GIF. Each seam is one
connected path from the top to the bottom."

Run the CLI if time permits:

```bash
python -m seam_carving.cli assets/scene.png out/demo.png --width 240
```

---

## Member 4: Horizontal, results, testing, conclusion (slides 8-11, 2 minutes)

### Slide 8: Horizontal seams and insertion (30 seconds)

"Horizontal seams use the same code. We transpose the image, run the vertical
pipeline, and transpose back. For insertion, we find a low-energy seam and
duplicate its pixels. The GUI inserts one seam at a time."

### Slide 9: Complexity and analysis (45 seconds)

"Per seam, DP and greedy are both O(H times W). The difference is quality, not
Big-O. For k seams the total is O(k times H times W), because the energy map
is recomputed.

Graph cut is optimal, but max-flow is much more expensive. Our measurements
show the gap.

Do we need the full DP table? Not strictly. Two rows plus parent pointers, or
a Hirschberg-style split, reduce the space.

Does multithreading help? The outer loop is sequential. The energy map and the
columns of a DP row can be parallelised, but Python's GIL limits pure-Python
threads, so we rely on NumPy vectorisation."

### Slide 10: Testing, CI, and bonus (30 seconds)

"We have 52 tests. They compare DP with brute force, graph cut with DP, and
cover border seams, horizontal seams, removal, insertion, the CLI, the
visualisation, and the GUI. The graph-cut experiment reproduces DP's total
energy exactly.

CI runs the tests and a real CLI resize on Python 3.10, 3.11, and 3.12. The
bonus features are step-by-step visualisation, the interactive GUI, and
CI/CD."

### Slide 11: Conclusion (15 seconds)

"Conclusion: DP is the practical choice for seam carving. Graph cut proves the
same optimum but is much slower. Greedy is a fast baseline, but it is not
optimal. Thank you. We are happy to answer questions."

---

## Backup answers for Q&A

- **Why not use the full DP table?** It is simple and fast. The low-memory
  alternatives are two rows plus parent pointers, or Hirschberg.
- **Why does graph cut match DP?** Both minimise the same seam energy. The
  max-flow min-cut theorem guarantees the cut is optimal.
- **Why is greedy suboptimal?** No lookahead; a cheap pixel can force an
  expensive future pixel.
- **Why recompute the energy map?** Removing a seam changes the neighbourhood,
  so the old gradient map is stale.
- **Why is the last-column sink edge finite?** To allow a seam on the last
  column. An infinite edge would forbid it.
- **Why transpose for horizontal seams?** It reuses the vertical code and
  keeps one implementation of each algorithm.
- **Why does divide and conquer not help Sobel?** A 3x3 convolution is already
  linear; D&C adds halo overhead. Separability and vectorisation are the real
  wins.
- **Does multithreading help?** Not the outer loop. The energy and per-row
  column work can be parallelised; the GIL limits pure-Python threads.
