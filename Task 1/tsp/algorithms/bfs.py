"""Breadth-first full search of the TSP state tree (task 3a).

The tree is visited level by level (all paths of length 1, then 2, ...).  Every
goal state (closed tour) is a leaf at depth n, so BFS has to generate the entire
tree before it can report the best tour - it is exact, but its FIFO frontier
holds whole levels of the tree, which is what makes it the most memory hungry
method of all.
"""
from __future__ import annotations

from typing import Optional

from ..datastructures import Queue
from ..problem import INF, Budget, SearchResult, State, TSPProblem


def bfs(problem: TSPProblem, budget: Optional[Budget] = None) -> SearchResult:
    frontier: Queue[State] = Queue()
    frontier.push(problem.initial_state())
    best: Optional[State] = None
    expanded = 0
    generated = 1
    max_frontier = 1
    completed = True
    status = "ok"

    while frontier:
        if budget is not None and budget.exceeded(len(frontier)):
            completed = False
            status = budget.reason or "stopped"
            break
        state = frontier.pop()
        if problem.is_goal(state):
            if best is None or state.cost < best.cost:
                best = state
            continue
        expanded += 1
        for child in problem.successors(state):
            frontier.push(child)
            generated += 1
        size = len(frontier)
        if size > max_frontier:
            max_frontier = size

    if best is None and completed:
        status = "no-tour"
    return SearchResult(
        algorithm="BFS",
        path=best.path if best is not None else None,
        cost=best.cost if best is not None else INF,
        nodes_expanded=expanded,
        nodes_generated=generated,
        max_frontier=max_frontier,
        completed=completed,
        exact=completed and best is not None,
        status=status,
    )
