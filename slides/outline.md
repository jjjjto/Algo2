# Presentation Outline

**Hard limits:** each person speaks for at most 2 minutes; the team's total
time is at most 12 minutes. The plan below assumes four members: 8 minutes of
talks and demo, then 4 minutes of Q&A. If the team has five members, use
10 minutes of talks and 2 minutes of Q&A. If it has six, use all 12 minutes
for talks and take questions afterwards.

## Slide 1: Title (15 seconds)

- Assignment 2: Image Compression by Seam Carving
- Team ID and member names
- One-line summary: content-aware resizing with DP, greedy, and graph cut

## Slide 2: Why seam carving? (45 seconds)

- Cropping: fast but removes content.
- Scaling: keeps content but distorts object shapes.
- Seam carving: removes low-energy connected paths and keeps the content.
- Show `report/figures/comparison_large.png`.
- The loop: energy map, minimum seam, remove, repeat.

## Slide 3: Energy map (1 minute)

- Grayscale conversion.
- Sobel kernels Gx and Gy.
- Energy = `sqrt(Gx^2 + Gy^2)`.
- High energy = edge; low energy = flat region.
- Show `report/figures/energy.png`.
- Complexity: `O(H*W)`.

## Slide 4: Dynamic programming (1 minute)

- A vertical seam: one pixel per row; adjacent rows differ by at most one
  column.
- `total[i,j] = e[i,j] + min(total[i-1,j-1], total[i-1,j], total[i-1,j+1])`.
- The best seam ends at the minimum of the last row.
- Parent pointers rebuild the seam.
- Complexity: `O(H*W)` per seam.

## Slide 5: Greedy and the comparison (1 minute)

- Greedy starts at the top-row minimum and takes the cheapest neighbour.
- No lookahead, so it can be suboptimal.
- Show the 3x3 counterexample: greedy 50, DP 1.
- Measured on 320x240, 80 seams:
  - DP: 0.680 s, total energy 392,678.5.
  - Greedy: 0.217 s, total energy 469,627.8.
- Greedy is faster; DP has lower energy and better quality.

## Slide 6: Graph cut (1 minute)

- One node per pixel; source on the left, sink on the right.
- Right edge capacity = `energy[i,j]`.
- Infinite backward and diagonal edges force a valid seam.
- Minimum cut = optimal seam; max-flow min-cut theorem.
- Dinic: BFS level graph + DFS blocking flow.
- Seam: rightmost source-side pixel in each row.
- Last-column sink capacity is `energy[i,W-1]`, not infinity.
- Measured on 96x72: DP 0.028 s, graph cut 17.207 s, same energy 51,472.5.

## Slide 7: Live demo (1 minute)

- Open `assets/scene.png` in the GUI.
- DP, width 240: the lighthouse is preserved.
- Reset, greedy: show the different seams.
- Show `report/figures/seams_first_6.png` and `report/figures/carving.gif`.
- Run the CLI if time permits.

## Slide 8: Horizontal seams and insertion (30 seconds)

- Horizontal seam: transpose, reuse the vertical code, transpose back.
- Show `report/figures/horizontal.png`.
- Insertion: duplicate a low-energy seam.
- The GUI inserts one seam at a time.

## Slide 9: Complexity and analysis (45 seconds)

- Per seam: DP `O(H*W)`, greedy `O(H*W)`, graph cut max-flow.
- Total for `k` seams: `O(k*H*W)` because the energy map is recomputed.
- Space: DP `O(H*W)`, greedy `O(H)`.
- Full DP table is not required: two rows plus parent pointers, or Hirschberg.
- Multithreading: the outer loop is sequential; energy and per-row columns can
  be parallelised.

## Slide 10: Testing, CI, and bonus (30 seconds)

- 52 tests: brute force, DP vs graph cut, edge cases, CLI, GUI.
- Graph-cut total energy equals DP total energy.
- CI runs tests and a real CLI resize on Python 3.10-3.12.
- Bonus: step visualisation, interactive GUI, CI/CD.

## Slide 11: Conclusion and Q&A (15 seconds plus Q&A)

- DP is the practical choice: optimal and fast.
- Graph cut proves the same optimum but is much slower.
- Greedy is a fast, lower-quality baseline.
- Thank the audience and invite questions.

---

## Timing chart (four members)

| Speaker | Slides | Time |
|---|---|---|
| Member 1 | 1-3 | 2 minutes |
| Member 2 | 4-5 | 2 minutes |
| Member 3 | 6-7 | 2 minutes |
| Member 4 | 8-11 | 2 minutes |
| Q&A | | 4 minutes |
| **Total** | | **12 minutes** |

For five members, give slides 6 and 7 to different speakers and move the
results discussion to the fifth speaker; keep each block at 2 minutes and
leave 2 minutes for Q&A.

## Recording checklist

1. Practise the demo commands once.
2. Record at 1080p with a clear microphone.
3. Show the terminal and the GUI.
4. Keep the video under 12 minutes.
5. Upload to YouTube (unlisted is fine).
6. Put the link in `report/report.md` and `README.md`.
