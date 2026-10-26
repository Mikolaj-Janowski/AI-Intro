"""A* search on the TSP state tree (task 3c).

Frontier ordered by f(state) = g(state) + h(state), where g is the cost of the
partial path and h a heuristic estimate of the remaining cost (see
``heuristics.py``).  With an admissible heuristic the first goal state popped
from the frontier is an optimal tour; with an inadmissible one the search is
much faster but only approximate.

``prune_dominated`` (on by default) adds the usual A* graph-search bookkeeping:
two tree nodes that have visited the same *set* of cities and stand in the same
current city can only differ in the order of the visited cities, so the more
expensive one can never lead to a better tour and is dropped.  The tree is still
the search space - this only avoids re-expanding dominated branches.  With a
consistent heuristic (0, min-in-edge, MST) the result stays optimal.

When the budget runs out before a goal is popped, the cheapest frontier node
is completed with the nearest-neighbour rule so that the method still returns
an (approximate) tour; such results are flagged ``completed=False``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..datastructures import MinHeap
from ..problem import INF, Budget, SearchResult, State, TSPProblem
from .greedy import nn_complete
from .heuristics import Heuristic


def best_first(problem: TSPProblem, heuristic: Heuristic, budget: Optional[Budget] = None,
               prune_dominated: bool = True, name: str = "A*",
               nn_fallback: bool = True) -> SearchResult:
    frontier: MinHeap[State] = MinHeap()
    root = problem.initial_state()
    h0 = heuristic(root)
    frontier.push((root.cost + h0, -root.cost), root)
    closed: Dict[Tuple[int, int], float] = {}
    expanded = 0
    generated = 1
    max_frontier = 1
    completed = True
    status = "ok"
    goal: Optional[State] = None

    while frontier:
        if budget is not None and budget.exceeded(len(frontier) + len(closed)):
            completed = False
            status = budget.reason or "stopped"
            break
        _, state = frontier.pop()
        if problem.is_goal(state):
            goal = state
            break
        if prune_dominated:
            key = (state.mask, state.path[-1])
            old = closed.get(key)
            if old is not None and old <= state.cost:
                continue                                   # dominated branch
            closed[key] = state.cost
        expanded += 1
        for child in problem.successors(state):
            generated += 1
            h = heuristic(child)
            if h == INF:
                continue                                   # provably no completion
            frontier.push((child.cost + h, -child.cost), child)
        size = len(frontier) + len(closed)
        if size > max_frontier:
            max_frontier = size

    extra: Dict[str, object] = {"heuristic": heuristic.name, "admissible": heuristic.admissible,
                                "prune_dominated": prune_dominated}
    if goal is None and not completed and nn_fallback and frontier:
        _, best_partial = frontier.peek()
        goal, _ = nn_complete(problem, best_partial, backtrack=True)
        if goal is not None:
            status += "+nn-completion"
            extra["completed_from_depth"] = best_partial.depth
    if goal is None and completed:
        status = "no-tour"

    return SearchResult(
        algorithm=name,
        path=goal.path if goal is not None else None,
        cost=goal.cost if goal is not None else INF,
        nodes_expanded=expanded,
        nodes_generated=generated,
        max_frontier=max_frontier,
        completed=completed,
        exact=completed and heuristic.admissible and goal is not None,
        status=status,
        extra=extra,
    )


def astar(problem: TSPProblem, heuristic: Heuristic, budget: Optional[Budget] = None,
          prune_dominated: bool = True, name: Optional[str] = None) -> SearchResult:
    label = name or f"A* ({heuristic.name}, {'admissible' if heuristic.admissible else 'inadmissible'})"
    return best_first(problem, heuristic, budget=budget, prune_dominated=prune_dominated, name=label)
