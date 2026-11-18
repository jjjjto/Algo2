"""Minimum s-t cut formulation of seam finding.

One node is created per pixel.  The source connects to column 0 and column
``W-1`` connects to the sink.  The finite edge from pixel ``(i, j)`` to
``(i, j+1)`` has capacity ``energy[i, j]``.  Infinite-capacity backward and
diagonal edges force the source side of the cut to be a prefix of every row
whose length changes by at most one between adjacent rows.  The boundary of
that prefix is a valid vertical seam, and the cut capacity is the seam energy.

The maximum flow is computed with Dinic's algorithm.  By the max-flow min-cut
theorem the seam is optimal for the given energy map.
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np

INF = 1e18
EPS = 1e-9


class _Edge:
    """A residual edge: destination, index of the reverse edge, capacity."""

    __slots__ = ("to", "rev", "cap")

    def __init__(self, to: int, rev: int, cap: float) -> None:
        self.to = to
        self.rev = rev
        self.cap = cap


class _Dinic:
    """Dinic's maximum-flow algorithm on a directed graph with float capacities."""

    def __init__(self, node_count: int) -> None:
        self.node_count = node_count
        self.graph: list[list[_Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, u: int, v: int, capacity: float) -> None:
        forward = _Edge(v, len(self.graph[v]), capacity)
        backward = _Edge(u, len(self.graph[u]), 0.0)
        self.graph[u].append(forward)
        self.graph[v].append(backward)

    def _bfs(self, source: int, sink: int) -> list[int] | None:
        level = [-1] * self.node_count
        level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge.cap > EPS and level[edge.to] < 0:
                    level[edge.to] = level[u] + 1
                    queue.append(edge.to)
        return level if level[sink] >= 0 else None

    def _dfs(
        self,
        u: int,
        sink: int,
        pushed: float,
        level: list[int],
        next_edge: list[int],
    ) -> float:
        if u == sink:
            return pushed
        while next_edge[u] < len(self.graph[u]):
            edge = self.graph[u][next_edge[u]]
            if edge.cap > EPS and level[edge.to] == level[u] + 1:
                sent = self._dfs(edge.to, sink, min(pushed, edge.cap), level, next_edge)
                if sent > EPS:
                    edge.cap -= sent
                    self.graph[edge.to][edge.rev].cap += sent
                    return sent
            next_edge[u] += 1
        return 0.0

    def max_flow(self, source: int, sink: int) -> float:
        """Return the value of a maximum flow from source to sink."""
        flow = 0.0
        while True:
            level = self._bfs(source, sink)
            if level is None:
                return flow
            next_edge = [0] * self.node_count
            while True:
                sent = self._dfs(source, sink, INF, level, next_edge)
                if sent <= EPS:
                    break
                flow += sent

    def source_side(self, source: int) -> list[bool]:
        """Return the nodes reachable from source in the residual graph."""
        reachable = [False] * self.node_count
        reachable[source] = True
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge.cap > EPS and not reachable[edge.to]:
                    reachable[edge.to] = True
                    queue.append(edge.to)
        return reachable


def _pixel(row: int, column: int, width: int) -> int:
    return row * width + column


def find_vertical_seam_graphcut(energy: np.ndarray) -> list[int]:
    """Return an optimal vertical seam using a minimum cut.

    Time is the cost of one max-flow on a graph with ``H*W + 2`` nodes and
    O(H*W) edges.  Space O(H*W).
    """
    h, w = energy.shape
    if h == 0 or w == 0:
        return []
    if w == 1:
        return [0] * h
    if np.any(energy < 0):
        raise ValueError("graph-cut seam finding needs non-negative energies")
    if not np.all(np.isfinite(energy)):
        raise ValueError("graph-cut seam finding needs finite energies")

    source = h * w
    sink = source + 1
    dinic = _Dinic(h * w + 2)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * h * w + 100))

    for i in range(h):
        dinic.add_edge(source, _pixel(i, 0, w), INF)
        # A finite sink edge lets a seam touch the right border.  Making this
        # edge infinite (as in the assignment sketch) would exclude column W-1.
        dinic.add_edge(_pixel(i, w - 1, w), sink, float(energy[i, w - 1]))
        for j in range(w - 1):
            u = _pixel(i, j, w)
            v = _pixel(i, j + 1, w)
            dinic.add_edge(u, v, float(energy[i, j]))
            dinic.add_edge(v, u, INF)

    for i in range(h):
        for j in range(1, w):
            u = _pixel(i, j, w)
            if i > 0:
                dinic.add_edge(u, _pixel(i - 1, j - 1, w), INF)
            if i < h - 1:
                dinic.add_edge(u, _pixel(i + 1, j - 1, w), INF)

    dinic.max_flow(source, sink)
    on_source_side = dinic.source_side(source)

    seam: list[int] = []
    for i in range(h):
        row = [j for j in range(w) if on_source_side[_pixel(i, j, w)]]
        if not row:
            raise RuntimeError("min cut produced an empty row")
        seam.append(max(row))
    return seam


def find_horizontal_seam_graphcut(energy: np.ndarray) -> list[int]:
    """Return an optimal horizontal seam by transposing the energy map."""
    return find_vertical_seam_graphcut(energy.T)
