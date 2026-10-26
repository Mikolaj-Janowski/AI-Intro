"""The TSP as a search problem over a state tree (task step 3).

A *state* is a partial (or full) path that starts in the chosen start city,
together with its accumulated cost.  The root of the state tree is the path
``(start,)``; the children of a state are obtained by extending the path with
one unvisited city that can be reached by a road.  When every city has been
visited the only possible child is the return to the start city; such a state
(a closed tour of n + 1 entries) is a *goal state*.

Because a state is identified by the whole path, two different paths are two
different nodes of the tree - this is what makes the tree of size O((n-1)!).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .graph import INF, Graph


class State:
    """A node of the state tree: the path taken so far and its cost."""

    __slots__ = ("path", "mask", "cost")

    def __init__(self, path: Tuple[int, ...], mask: int, cost: float):
        self.path = path      # cities in visiting order, starts with the start city
        self.mask = mask      # bitmask of visited cities (fast membership test)
        self.cost = cost      # accumulated path cost g(state)

    @property
    def current(self) -> int:
        return self.path[-1]

    @property
    def depth(self) -> int:
        return len(self.path) - 1

    def __repr__(self) -> str:
        return f"State(path={self.path}, cost={self.cost:.2f})"


class TSPProblem:
    """Search-problem view of a TSP instance: initial state, successors, goal test."""

    def __init__(self, graph: Graph, start: int = 0):
        if not 0 <= start < graph.n:
            raise ValueError("start city out of range")
        self.graph = graph
        self.n = graph.n
        self.start = start
        self.full_mask = (1 << graph.n) - 1

    # --------------------------------------------------------- tree interface
    def initial_state(self) -> State:
        return State((self.start,), 1 << self.start, 0.0)

    def is_goal(self, state: State) -> bool:
        return len(state.path) == self.n + 1

    def successors(self, state: State) -> Iterator[State]:
        """Children of ``state`` in the state tree."""
        path = state.path
        current = path[-1]
        mask = state.mask
        if mask == self.full_mask:
            if len(path) == self.n:                       # all visited -> go home
                w = self.graph.cost(current, self.start)
                if w != INF:
                    yield State(path + (self.start,), mask, state.cost + w)
            return
        cost = state.cost
        for v, w in self.graph.neighbors(current):
            if not (mask >> v) & 1:
                yield State(path + (v,), mask | (1 << v), cost + w)

    def unvisited(self, state: State) -> List[int]:
        mask = state.mask
        return [v for v in range(self.n) if not (mask >> v) & 1]

    # ----------------------------------------------------------------- utils
    def path_cost(self, path: Sequence[int]) -> float:
        """Recompute the cost of a closed tour from the graph (validation)."""
        if len(path) != self.n + 1 or path[0] != self.start or path[-1] != self.start:
            return INF
        if len(set(path[:-1])) != self.n:
            return INF
        total = 0.0
        for a, b in zip(path, path[1:]):
            w = self.graph.cost(a, b)
            if w == INF:
                return INF
            total += w
        return total

    def describe_path(self, path: Optional[Sequence[int]]) -> str:
        if not path:
            return "(no tour)"
        return " -> ".join(self.graph.names[c] for c in path)


# ---------------------------------------------------------------------------
# Result of a search and the time / node budget shared by the algorithms
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    algorithm: str
    path: Optional[Tuple[int, ...]]
    cost: float
    nodes_expanded: int = 0        # states whose successors were generated
    nodes_generated: int = 0       # states created
    max_frontier: int = 0          # largest number of states held in memory at once
    completed: bool = True         # False when stopped by the time / memory budget
    exact: bool = False            # True when the algorithm guarantees optimality of ``path``
    status: str = "ok"             # ok | no-tour | timeout | node-limit (+ "nn-completion")
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.path is not None and self.cost != INF


class Budget:
    """Wall-clock and memory (number of stored states) limit for a search."""

    def __init__(self, time_limit: Optional[float] = None, node_limit: Optional[int] = None):
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.started = perf_counter()
        self.reason: Optional[str] = None
        self._calls = 0

    def exceeded(self, nodes_in_memory: int = 0) -> bool:
        if self.node_limit is not None and nodes_in_memory > self.node_limit:
            self.reason = "node-limit"
            return True
        self._calls += 1
        if self.time_limit is not None and (self._calls & 255) == 0:
            if perf_counter() - self.started > self.time_limit:
                self.reason = "timeout"
                return True
        return False

    def elapsed(self) -> float:
        return perf_counter() - self.started


# ---------------------------------------------------------------------------
# Explicit state tree (for small instances / illustration)
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    state: State
    children: List["TreeNode"] = field(default_factory=list)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def leaves(self) -> int:
        return 1 if not self.children else sum(c.leaves() for c in self.children)


def build_state_tree(problem: TSPProblem, max_nodes: int = 50_000) -> TreeNode:
    """Materialise the whole state tree (only sensible for n <= 7 or so)."""
    root = TreeNode(problem.initial_state())
    pending = [root]
    count = 1
    while pending:
        node = pending.pop()
        for child_state in problem.successors(node.state):
            child = TreeNode(child_state)
            node.children.append(child)
            pending.append(child)
            count += 1
            if count > max_nodes:
                raise ValueError(f"state tree larger than {max_nodes} nodes; use a smaller n")
    return root


def format_state_tree(problem: TSPProblem, root: TreeNode) -> str:
    names = problem.graph.names
    lines: List[str] = []

    def walk(node: TreeNode, prefix: str, is_last: bool, is_root: bool) -> None:
        s = node.state
        label = names[s.current] if not is_root else names[s.current] + " (start)"
        tag = ""
        if problem.is_goal(s):
            tag = f"   <-- tour, cost {s.cost:.2f}"
        elif not node.children:
            tag = "   <-- dead end (no road)"
        connector = "" if is_root else ("`-- " if is_last else "|-- ")
        lines.append(f"{prefix}{connector}{label}  [g={s.cost:.2f}]{tag}")
        child_prefix = prefix if is_root else prefix + ("    " if is_last else "|   ")
        for i, child in enumerate(node.children):
            walk(child, child_prefix, i == len(node.children) - 1, False)

    walk(root, "", True, True)
    return "\n".join(lines)
