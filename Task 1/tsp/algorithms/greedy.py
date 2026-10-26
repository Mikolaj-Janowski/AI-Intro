"""Greedy approximations (task 3b): Nearest Neighbour and Dijkstra.

Nearest Neighbour (NN)
    Walks down the state tree choosing, at every node, the child reached by
    the cheapest road (locally greedy, no look-ahead).  With an incomplete
    road network the pure NN walk can run into a dead end (no road to any
    unvisited city, or no road back home); ``backtrack=True`` then steps back
    to the previous node and tries its next-cheapest child, i.e. NN becomes a
    depth-first walk with cost-ordered children that stops at the first closed
    tour.  On a complete road network backtracking never triggers and the
    result is the classical NN tour.

Dijkstra
    Dijkstra's algorithm run on the state tree: the frontier node with the
    smallest accumulated cost g is always expanded next (a greedy choice on the
    cost-so-far instead of on the single next road).  This is uniform-cost
    search, i.e. A* with h = 0.  Run to completion it is exact; because its
    frontier grows exponentially, a time/memory budget can be given - when the
    budget runs out the cheapest frontier node is completed with NN, which
    turns the method into an approximation.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..problem import INF, Budget, SearchResult, State, TSPProblem


def nn_complete(problem: TSPProblem, state: State, backtrack: bool = True
                ) -> Tuple[Optional[State], dict]:
    """Extend ``state`` to a closed tour with the nearest-neighbour rule.

    Returns (goal_state or None, statistics).  With ``backtrack`` the walk
    is a cost-ordered DFS that returns the first closed tour it finds (so it
    fails only when no completion exists at all).
    """
    stats = {"expanded": 0, "generated": 0, "backtracks": 0, "max_frontier": 0}
    if problem.is_goal(state):
        return state, stats

    def ordered_children(s: State) -> List[State]:
        children = sorted(problem.successors(s), key=lambda c: c.cost)
        stats["expanded"] += 1
        stats["generated"] += len(children)
        return children if backtrack else children[:1]

    stack: List[Tuple[List[State], int]] = [(ordered_children(state), 0)]
    held = len(stack[0][0])
    stats["max_frontier"] = held
    while stack:
        children, idx = stack[-1]
        if idx >= len(children):               # this node is exhausted
            stack.pop()
            held -= len(children)
            if not backtrack or not stack:
                return None, stats
            stats["backtracks"] += 1
            continue
        stack[-1] = (children, idx + 1)
        child = children[idx]
        if problem.is_goal(child):
            return child, stats
        grand = ordered_children(child)
        stack.append((grand, 0))
        held += len(grand)
        if held > stats["max_frontier"]:
            stats["max_frontier"] = held
    return None, stats


def greedy_nn(problem: TSPProblem, budget: Optional[Budget] = None,
              backtrack: bool = True) -> SearchResult:
    goal, stats = nn_complete(problem, problem.initial_state(), backtrack=backtrack)
    return SearchResult(
        algorithm="Greedy NN",
        path=goal.path if goal is not None else None,
        cost=goal.cost if goal is not None else INF,
        nodes_expanded=stats["expanded"],
        nodes_generated=stats["generated"],
        max_frontier=max(stats["max_frontier"], problem.n),
        completed=True,
        exact=False,
        status="ok" if goal is not None else "no-tour",
        extra={"backtracks": stats["backtracks"]},
    )


def dijkstra(problem: TSPProblem, budget: Optional[Budget] = None,
             prune_dominated: bool = True) -> SearchResult:
    """Dijkstra / uniform-cost search on the state tree (A* with h = 0)."""
    from .astar import best_first              # local import avoids a cycle
    from .heuristics import ZeroHeuristic

    return best_first(problem, ZeroHeuristic(problem), budget=budget,
                      prune_dominated=prune_dominated, name="Dijkstra (UCS)")
