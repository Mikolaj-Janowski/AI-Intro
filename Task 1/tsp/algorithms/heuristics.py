"""Heuristics h(state) for A* (task 3c).

All heuristics estimate the cost of the *remaining* part of the tour: from the
current city, through every unvisited city, back to the start city.

Admissible (never over-estimate -> A* is exact):
    ZeroHeuristic        h = 0                         (A* degenerates to Dijkstra/UCS)
    MinInEdgeHeuristic   sum of the cheapest incoming road of every city that
                         still has to be entered (the unvisited ones + start)
    MSTHeuristic         weight of a minimum spanning tree over
                         {current} + unvisited + {start}; computed with Prim's
                         algorithm on the undirected weights min(c(u,v), c(v,u)).
                         The remaining path is a spanning tree of that vertex
                         set, so the MST can never cost more -> admissible
                         (and consistent).
Inadmissible (may over-estimate -> fast, but only approximate):
    NNCompletionHeuristic  cost of finishing the tour with the nearest-neighbour
                           rule - a real completion, hence an upper bound
    WeightedHeuristic      w * h for an admissible h with w > 1 (weighted A*)

When the heuristic can prove that no completion exists (the remaining cities
are disconnected using the remaining roads) it returns +inf and the branch is
pruned - that is still admissible because the true cost is +inf as well.
"""
from __future__ import annotations

from typing import List

from ..problem import INF, State, TSPProblem


class Heuristic:
    name = "h"
    admissible = True

    def __init__(self, problem: TSPProblem):
        self.problem = problem
        self.n = problem.n
        self.start = problem.start

    def _remaining(self, state: State) -> List[int]:
        mask = state.mask
        return [v for v in range(self.n) if not (mask >> v) & 1]

    def __call__(self, state: State) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


class ZeroHeuristic(Heuristic):
    name = "0"
    admissible = True

    def __call__(self, state: State) -> float:
        return 0.0


class MinInEdgeHeuristic(Heuristic):
    name = "min-in-edge"
    admissible = True

    def __call__(self, state: State) -> float:
        if self.problem.is_goal(state):
            return 0.0
        graph = self.problem.graph
        unvisited = self._remaining(state)
        current = state.current
        sources = unvisited + [current]
        targets = unvisited + [self.start]       # every one of these must still be entered
        total = 0.0
        for t in targets:
            best = INF
            for s in sources:
                if s != t:
                    w = graph.cost(s, t)
                    if w < best:
                        best = w
            if best == INF:
                return INF
            total += best
        return total


class MSTHeuristic(Heuristic):
    name = "MST"
    admissible = True

    def __init__(self, problem: TSPProblem):
        super().__init__(problem)
        g = problem.graph
        n = self.n
        # undirected weights: cheapest of the two directions
        self.w = [[min(g.cost(u, v), g.cost(v, u)) for v in range(n)] for u in range(n)]

    def __call__(self, state: State) -> float:
        if self.problem.is_goal(state):
            return 0.0
        current = state.current
        verts = [current] + self._remaining(state)
        if current != self.start:
            verts.append(self.start)
        k = len(verts)
        if k < 2:
            return 0.0
        # Prim's algorithm on the induced sub-graph
        w = self.w
        in_tree = [False] * k
        dist = [INF] * k
        dist[0] = 0.0
        total = 0.0
        for _ in range(k):
            best_i = -1
            best_d = INF
            for i in range(k):
                if not in_tree[i] and dist[i] < best_d:
                    best_d = dist[i]
                    best_i = i
            if best_i < 0:
                return INF                        # disconnected -> no tour possible
            in_tree[best_i] = True
            total += best_d
            row = w[verts[best_i]]
            for i in range(k):
                if not in_tree[i]:
                    d = row[verts[i]]
                    if d < dist[i]:
                        dist[i] = d
        return total


class NNCompletionHeuristic(Heuristic):
    """Cost of completing the tour greedily (nearest neighbour with backtracking)."""

    name = "NN-completion"
    admissible = False

    def __call__(self, state: State) -> float:
        from .greedy import nn_complete
        goal, _ = nn_complete(self.problem, state, backtrack=True)
        return INF if goal is None else goal.cost - state.cost


class WeightedHeuristic(Heuristic):
    """w * h(state); inadmissible for w > 1 (weighted A*)."""

    def __init__(self, base: Heuristic, weight: float):
        super().__init__(base.problem)
        self.base = base
        self.weight = weight
        self.name = f"{weight:g}x{base.name}"
        self.admissible = base.admissible and weight <= 1.0

    def __call__(self, state: State) -> float:
        return self.weight * self.base(state)
