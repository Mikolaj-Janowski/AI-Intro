"""Weighted directed graph used to represent the city map (task step 2).

Cities are the nodes (0..n-1), roads are the edges, the weight of an edge
(u, v) is the cost of travelling from city u to city v.
"""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

INF = float("inf")


class Graph:
    """Directed weighted graph stored as one adjacency dictionary per node."""

    def __init__(self, n: int, names: Optional[Sequence[str]] = None, cities=None):
        self.n = n
        self.names: List[str] = list(names) if names is not None else [f"C{i}" for i in range(n)]
        self.cities = cities  # optional list of City objects (for plotting / reporting)
        self._adj: List[Dict[int, float]] = [dict() for _ in range(n)]
        self._edge_count = 0

    # ------------------------------------------------------------------ edges
    def add_edge(self, u: int, v: int, weight: float) -> None:
        if u == v:
            raise ValueError("self loops are not allowed")
        if v not in self._adj[u]:
            self._edge_count += 1
        self._adj[u][v] = weight

    def has_edge(self, u: int, v: int) -> bool:
        return v in self._adj[u]

    def cost(self, u: int, v: int) -> float:
        """Cost of the road u -> v, or +inf if there is no such road."""
        return self._adj[u].get(v, INF)

    def neighbors(self, u: int) -> Iterable[Tuple[int, float]]:
        """(v, cost) pairs for every road leaving u."""
        return self._adj[u].items()

    def out_degree(self, u: int) -> int:
        return len(self._adj[u])

    def edges(self) -> Iterator[Tuple[int, int, float]]:
        for u in range(self.n):
            for v, w in self._adj[u].items():
                yield u, v, w

    # ------------------------------------------------------------- statistics
    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def possible_edges(self) -> int:
        return self.n * (self.n - 1)

    @property
    def density(self) -> float:
        return self._edge_count / self.possible_edges if self.possible_edges else 0.0

    def is_symmetric(self) -> bool:
        for u, v, w in self.edges():
            if self._adj[v].get(u) != w:
                return False
        return True

    def min_edge_cost(self) -> float:
        return min((w for _, _, w in self.edges()), default=INF)

    def average_edge_cost(self) -> float:
        if self._edge_count == 0:
            return INF
        return sum(w for _, _, w in self.edges()) / self._edge_count

    # -------------------------------------------------------------- exporting
    def to_matrix(self) -> List[List[float]]:
        """Cost matrix; INF where there is no road, 0 on the diagonal."""
        m = [[INF] * self.n for _ in range(self.n)]
        for u in range(self.n):
            m[u][u] = 0.0
            for v, w in self._adj[u].items():
                m[u][v] = w
        return m

    def format_matrix(self, precision: int = 1) -> str:
        width = max(7, max(len(nm) for nm in self.names) + 1)
        header = " " * width + "".join(nm.rjust(width) for nm in self.names)
        lines = [header]
        for u, row in enumerate(self.to_matrix()):
            cells = []
            for w in row:
                cells.append(("-" if w == INF else f"{w:.{precision}f}").rjust(width))
            lines.append(self.names[u].ljust(width) + "".join(cells))
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Graphviz description of the graph (handy for a report)."""
        out = ["digraph map {", "  node [shape=circle];"]
        for u in range(self.n):
            out.append(f'  {u} [label="{self.names[u]}"];')
        for u, v, w in self.edges():
            out.append(f'  {u} -> {v} [label="{w:.1f}"];')
        out.append("}")
        return "\n".join(out)

    def summary(self) -> str:
        return (f"Graph: {self.n} nodes, {self.edge_count} directed edges "
                f"({self.density * 100:.1f}% of {self.possible_edges} possible), "
                f"{'symmetric' if self.is_symmetric() else 'asymmetric'} costs")

    def __repr__(self) -> str:
        return f"Graph(n={self.n}, edges={self.edge_count})"
