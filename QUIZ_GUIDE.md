# Seam Carving Quiz Guide

This guide covers the exact code and report in this repository. Read it once,
then answer the self-test at the end without looking.

The quiz is closed-book, team-based, and based on your submitted code and
report. You do not need to memorise every line. You need to be able to explain
what each part does, why it is there, and what its complexity is.

---

## 1. The five-minute cheat sheet

| Question | Short answer |
|---|---|
| What is a vertical seam? | One pixel per row, connected vertically or diagonally. Adjacent rows differ by at most one column. |
| What is a horizontal seam? | One pixel per column, same adjacency rule. |
| What is the energy of a pixel? | Sobel gradient magnitude: `sqrt(Gx^2 + Gy^2)` on grayscale. High energy = edge. |
| What is the energy of a seam? | Sum of the energies of its pixels. |
| How many seams are there? | At least `2^m` for `m` rows and `n > 1` columns. |
| DP recurrence? | `total[i,j] = e[i,j] + min(total[i-1,j-1], total[i-1,j], total[i-1,j+1])`. |
| DP base case? | `total[0,j] = e[0,j]`. |
| Where does the optimal seam end? | At the minimum value in the last row. |
| How is the seam rebuilt? | Follow the parent pointers from the bottom row upwards. |
| DP time per seam? | `O(H*W)`. |
| DP space? | `O(H*W)` for the full table and parent pointers. |
| Total for `k` seams? | `O(k*H*W)` because the energy map is recomputed every time. |
| Greedy time? | `O(H*W)` per seam, `O(H)` space. |
| Is greedy optimal? | No. It has no lookahead. |
| Graph cut idea? | One node per pixel; minimum s-t cut; max-flow min-cut theorem. |
| What do the infinite edges do? | Force the source side to be a valid seam: a prefix in each row, changing by at most one. |
| How is the seam read from the cut? | Rightmost source-side pixel in each row. |
| Graph cut vs DP quality? | Same optimal seam cost. |
| Graph cut vs DP speed? | Graph cut is much slower in this project. |
| Horizontal seams? | Transpose the image, run the vertical code, transpose back. |
| Why recompute the energy map? | Removing a seam changes the neighbourhood, so the old map is stale. |
| Does divide and conquer speed up Sobel? | No asymptotic improvement for a 3x3 kernel. Direct/vectorised convolution is `O(H*W)`. |
| Can we avoid the full DP table? | Two rows plus parent pointers, or Hirschberg's divide and conquer. |
| Does multithreading help? | The outer seam loop is sequential; energy and per-row columns can be parallelised. Python's GIL limits pure-Python threads. |

---

## 2. What each file does

| File | What to remember |
|---|---|
| `seam_carving/energy.py` | Grayscale conversion and Sobel. Pads the border by edge replication. `np.hypot(gx, gy)`. |
| `seam_carving/dp.py` | `total` table, `parent` table, recurrence, backtracking, `cumulative_energy`, `seam_energy`. |
| `seam_carving/greedy.py` | Starts at the top-row minimum, then chooses the cheapest of `j-1, j, j+1`. |
| `seam_carving/graphcut.py` | `_Dinic` max-flow, graph construction, source-side BFS, seam extraction. |
| `seam_carving/carve.py` | `remove_vertical_seam`, `remove_horizontal_seam`, `insert_*`, `carve_width`, `carve_height`, `enlarge_*`, `resize`, `FINDERS`, `CarveResult`. |
| `seam_carving/visualize.py` | Seam overlays, energy figures, step images, GIF, comparison and runtime plots. |
| `seam_carving/cli.py` | `--width`, `--height`, `--method`, `--seams-dir`, `--gif`, `--energy`. |
| `seam_carving/gui.py` | Tkinter window; carving runs in a background thread. |
| `tests/` | Brute force, DP vs graph cut, greedy counterexample, removal, insertion, CLI, visualisation. |
| `scripts/run_experiments.py` | Produces `report/results.md` and `report/figures/`. |

---

## 3. The theory answers in plain words

### 3.1 How many seams?

Take the first two columns. In every row you may pick column 0 or column 1.
Any sequence of picks is a valid seam, because adjacent picks are either the
same column (vertical move) or the other column (diagonal move). There are
`2^m` sequences for `m` rows. So the number of seams is at least `2^m`,
which is exponential. The condition `n > 1` gives us two columns to choose
from.

### 3.2 The DP subproblem

Let `total[i,j]` be the cheapest seam that ends at pixel `(i,j)`.

- First row: `total[0,j] = e[0,j]`.
- Other rows: add the pixel energy to the cheapest of the three pixels above
  it.
- The answer is the smallest `total[H-1,j]`.

Why is this correct? If the best seam to `(i,j)` used a non-best prefix, you
could swap in the better prefix and get a cheaper seam. That is optimal
substructure. The same `total[i-1,j]` is used by three pixels in row `i`, so
the subproblems overlap and storing them saves work.

### 3.3 Divide and conquer for Sobel

Convolution with a fixed 3x3 kernel is already `O(H*W)`. Splitting the image
into blocks needs one-pixel halos, and merging the blocks costs work at every
level. The naive recurrence `T(N) = 4T(N/4) + O(N)` gives `O(N log N)`, which
is worse. Divide and conquer helps with parallel execution or very large
kernels, not with the asymptotic cost of a 3x3 Sobel filter. Separable 1-D
passes (horizontal then vertical) and vectorisation are the useful
optimisations.

---

## 4. Code snippets you may be asked about

### 4.1 The DP update

```python
left = np.full(w, INF)
left[1:] = previous[:-1]
right = np.full(w, INF)
right[:-1] = previous[1:]

choices = np.stack((left, previous, right))
total[i] = energy[i] + choices.min(axis=0)
parent[i] = choices.argmin(axis=0) - 1
```

- `left[1:] = previous[:-1]` shifts the previous row right, so `left[j]` is
  `total[i-1, j-1]` (the up-left predecessor).
- `right[:-1] = previous[1:]` shifts it left, so `right[j]` is
  `total[i-1, j+1]` (the up-right predecessor).
- `choices.min(axis=0)` takes the cheapest predecessor for every column.
- `argmin` returns 0, 1, or 2 for left, up, right. Subtracting 1 stores
  `-1`, `0`, or `+1`. That is the parent pointer.
- The border entries are `INF`, so a seam cannot step outside the image.

### 4.2 Backtracking

```python
j = int(np.argmin(total[-1]))
seam = [j]
for i in range(h - 1, 0, -1):
    j += int(parent[i, j])
    seam.append(j)
seam.reverse()
```

- Start at the cheapest pixel in the last row.
- Move up one row at a time using the parent pointer.
- Reverse the list so it reads top to bottom.
- The result has exactly `H` entries.

### 4.3 The greedy loop

```python
seam = [int(np.argmin(energy[0]))]
for i in range(1, h):
    j = seam[-1]
    candidates = [c for c in (j - 1, j, j + 1) if 0 <= c < w]
    seam.append(min(candidates, key=lambda c: energy[i, c]))
```

- Starts at the cheapest pixel in the top row.
- Each step only sees the current row.
- It never checks whether a cheap pixel now forces an expensive pixel later.
- That is why it can lose to DP.

The 3x3 counterexample:

```
[[  0,  1, 100],
 [100, 50,   0],
 [100, 100,  0]]
```

Greedy: 0 (col 0), 50 (col 1), 0 (col 2) = 50.
DP: 1 (col 1), 0 (col 2), 0 (col 2) = 1.

### 4.4 The graph-cut construction

```python
dinic.add_edge(source, _pixel(i, 0, w), INF)
dinic.add_edge(_pixel(i, w - 1, w), sink, float(energy[i, w - 1]))
dinic.add_edge(u, v, float(energy[i, j]))   # u=(i,j), v=(i,j+1)
dinic.add_edge(v, u, INF)
```

- `source -> column 0` with `INF`: column 0 is always on the source side.
- `column W-1 -> sink` with `energy[i, W-1]`: cutting here costs the last
  pixel's energy. Using `INF` here would make the last column impossible to
  choose. The report explains this correction.
- `(i,j) -> (i,j+1)` with `energy[i,j]`: this is the cost of cutting after
  pixel `(i,j)`.
- `(i,j+1) -> (i,j)` with `INF`: the source side must be a prefix. You cannot
  have a hole in a row.
- The diagonal `INF` edges make the prefix length change by at most one
  between adjacent rows.

### 4.5 Reading the seam from the cut

```python
dinic.max_flow(source, sink)
on_source_side = dinic.source_side(source)

for i in range(h):
    row = [j for j in range(w) if on_source_side[_pixel(i, j, w)]]
    seam.append(max(row))
```

- After max-flow, BFS from the source in the residual graph gives the source
  side of the minimum cut.
- In each row, the source side is a prefix `0..k`.
- The rightmost source-side pixel `k` is the seam pixel.
- `max(row)` finds it.

### 4.6 The carving loop

```python
while image.shape[1] > target_width:
    energy = sobel_energy(image)
    seam = finder(energy)
    seams.append(seam)
    energies.append(seam_energy(energy, seam))
    image = remove_vertical_seam(image, seam)
```

- The energy map is inside the loop. It is recomputed after every removal.
- Removing a seam changes the neighbours of the pixels next to it.
- `seams` and `energies` make the experiment comparison possible.
- The loop stops exactly at the target width.

### 4.7 Horizontal seams

```python
result = carve_width(_transpose(image), target_height, finder)
return CarveResult(
    image=_transpose(result.image),
    seams=result.seams,
    energies=result.energies,
)
```

- Transposing swaps rows and columns.
- A horizontal seam in the original image is a vertical seam in the
  transposed image.
- The same DP, greedy, and graph-cut code is reused.
- The returned seam is a list of row indices, one per original column.

### 4.8 Insertion

```python
out[i, : j + 1] = image[i, : j + 1]
out[i, j + 1] = image[i, j]
out[i, j + 2 :] = image[i, j + 1 :]
```

- The seam pixel is duplicated, so the image becomes one column wider.
- This is simple but can create visible streaks.
- Averaging the pixel with its neighbour is a smoother alternative.

---

## 5. Complexity table

| Operation | Time | Space |
|---|---|---|
| Grayscale conversion | `O(H*W)` | `O(H*W)` |
| Sobel energy map | `O(H*W)` | `O(H*W)` |
| DP seam | `O(H*W)` | `O(H*W)` |
| Greedy seam | `O(H*W)` | `O(H)` |
| Graph-cut graph build | `O(H*W)` | `O(H*W)` |
| Dinic max-flow | `O(V^2 E)` worst case, `V = H*W + 2` | `O(V + E)` |
| Remove one seam | `O(H*W)` | `O(H*W)` |
| Carve `k` seams (DP/greedy) | `O(k*H*W)` | `O(H*W)` |
| Insert `k` seams | `O(k*H*W)` | `O(H*W)` |

The important answer to "what is the complexity of seam carving?" is:
**`O(k*H*W)` time and `O(H*W)` space** for `k` seams with the DP or greedy
finder, because the energy map is recomputed after each removal.

---

## 6. Compare the three algorithms

| | Greedy | DP | Graph cut |
|---|---|---|---|
| Optimal for the energy? | No | Yes | Yes |
| Time per seam | `O(H*W)` | `O(H*W)` | Max-flow (much larger) |
| Space | `O(H)` | `O(H*W)` | `O(H*W)` |
| Our measured time (96x72, 16 seams) | 0.005 s | 0.028 s | 17.207 s |
| Our measured total seam energy | 61,221.8 | 51,472.5 | 51,472.5 |
| Quality | Worse | Best | Best |
| Implementation effort | Low | Medium | High |

Key observations from the experiment:

- DP and graph cut have exactly the same total seam energy. Both find a
  minimum seam on every iteration.
- Greedy has higher energy because it does not look ahead.
- Graph cut is much slower than DP for this problem. DP exploits the grid's
  DAG structure; graph cut solves a more general max-flow problem.
- Greedy is faster than DP, but the quality loss is visible in the energy
  numbers and in the comparison figure.

---

## 7. Common wrong answers

| Wrong statement | Correct statement |
|---|---|
| "Greedy finds the minimum seam." | Greedy is local and can be suboptimal. The 3x3 example proves it. |
| "DP is `O(k*H*W)` per seam." | DP is `O(H*W)` per seam. The total for `k` seams is `O(k*H*W)`. |
| "Graph cut is faster than DP." | In this project, graph cut is much slower, although both are optimal. |
| "The last column is connected to the sink with infinite capacity in our code." | It uses `energy[i, W-1]`, so a seam on the last column is possible. |
| "We can reuse the energy map." | The map must be recomputed after each removal because the neighbourhood changes. |
| "Horizontal seams use a different algorithm." | They use the same algorithm on the transposed image. |
| "Divide and conquer makes Sobel faster asymptotically." | For a 3x3 kernel, direct convolution is already `O(H*W)`. D&C adds halo overhead. |
| "Multithreading removes seams in parallel." | The outer loop is sequential; only the energy and per-row computations can be parallelised. |
| "The energy is computed on colour channels." | This implementation uses grayscale luminance. |
| "The DP table cannot be reduced." | Two rows plus parent pointers, or Hirschberg's method, reduce the space. |

---

## 8. Self-test (answers below)

1. What is the minimum number of seams in a 5-row image with 2 columns?
2. Write the DP recurrence.
3. What does `parent[i,j] = -1` mean?
4. What is the time complexity of one DP seam? Of 10 seams?
5. Why is greedy not optimal?
6. What is the capacity of the edge `(i,j) -> (i,j+1)` in the graph cut?
7. Why are the backward and diagonal edges infinite?
8. How do you find the seam after max-flow?
9. Why does the graph cut use a finite capacity on the last column to the sink?
10. How are horizontal seams implemented?
11. Why is the energy map recomputed inside the carving loop?
12. Does divide and conquer improve the asymptotic cost of Sobel? Why?
13. What is the total time complexity for `k` seams with DP?
14. What is the space complexity of greedy?
15. How would you reduce the DP space?
16. Does multithreading help the outer seam loop?
17. What is the energy of a seam?
18. What is the difference between cropping, scaling, and seam carving?
19. What does `FINDERS` do?
20. What does the test `assert np.isclose(seam_energy(energy, cut), seam_energy(energy, dynamic))` prove?

### Answers

1. `2^5 = 32` seams (choose column 0 or 1 in every row).
2. `total[i,j] = e[i,j] + min(total[i-1,j-1], total[i-1,j], total[i-1,j+1])`.
3. The best predecessor is the up-left pixel, so the seam moves one column
   left when going down.
4. `O(H*W)` for one seam; `O(10*H*W)` for ten.
5. It chooses the cheapest neighbour row by row and never checks future rows.
   A cheap pixel can force an expensive one later.
6. `energy[i,j]`, the cost of cutting after pixel `(i,j)`.
7. They force the source side to be a prefix in each row and force the prefix
   lengths of adjacent rows to differ by at most one, so the boundary is a
   connected seam.
8. BFS/DFS from the source in the residual graph; in each row, take the
   rightmost source-side pixel.
9. To allow the seam to use the last column. An infinite capacity would make
   the cut impossible to cross there.
10. Transpose the image, run the vertical pipeline, transpose the result back.
11. Removing a seam changes the neighbours of nearby pixels, so the old
    gradient map is stale.
12. No. A 3x3 convolution is already `O(H*W)`. D&C needs halos and merging,
    which adds `O(N log N)` overhead in the naive version.
13. `O(k*H*W)` time.
14. `O(H)` space.
15. Store only two rows of costs and keep parent pointers, or use a
    Hirschberg-style divide-and-conquer reconstruction.
16. Not for the outer loop, because each seam depends on the previous removal.
    Energy and per-row column work can be parallelised.
17. The sum of the energies of the pixels on the seam.
18. Cropping removes a fixed rectangle and loses content; scaling distorts
    object shapes; seam carving removes low-energy paths and keeps content and
    object shapes.
19. It maps a method name (`dp`, `greedy`, `graphcut`) to the corresponding
    vertical seam-finder function, so the CLI and GUI do not need `if/elif`
    chains.
20. It proves that the graph-cut seam has the same energy as the DP seam, so
    the graph construction and max-flow implementation are correct (at least
    for that map).

---

## 9. Demo checklist for the presentation

1. Open `assets/scene.png` in the GUI.
2. Set width to 240, choose DP, apply. Show the lighthouse is preserved.
3. Reset, choose greedy, apply. Point out the higher energy / different seams.
4. Show the step images in `report/figures/steps/` and the GIF.
5. Run the CLI:
   ```bash
   python -m seam_carving.cli assets/scene.png out/demo.png --width 240 \
       --seams-dir out/steps --gif out/carving.gif
   ```
6. Run the tests:
   ```bash
   python -m pytest -q
   ```
7. Open `report/results.md` and point to the graph-cut energy matching DP.
8. Mention the CI workflow and the three bonus features.

Keep the demo under two minutes per person. Practise the commands once before
the presentation.
