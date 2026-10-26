"""Depth-first full search of the TSP state tree (task 3a).

The tree is explored branch by branch with an explicit LIFO stack.  DFS finds
its first complete tour after only n steps and then keeps improving the best
tour found so far; its frontier never holds more than O(n^2) states, so it is
memory-cheap - but it still has to visit the whole tree to prove optimality.

``prune=True`` turns the plain full search into branch-and-bound: a branch whose
partial cost already exceeds the best tour found so far is cut off.  This is
an optional extra and is *off* by default (the task asks for a full search).
"""
from __future__ import annotations

from typing import Optional

from ..datastructures import Stack
from ..problem import INF, Budget, SearchResult, State, TSPProblem


def dfs(problem: TSPProblem, budget: Optional[Budget] = None, prune: bool = False) -> SearchResult:
    frontier: Stack[State] = Stack()
    frontier.push(problem.initial_state())
    best: Optional[State] = None
    best_cost = INF
    expanded = 0
    generated = 1
    max_frontier = 1
    pruned = 0
    completed = True
    status = "ok"

    while frontier:
        if budget is not None and budget.exceeded(len(frontier)):
            completed = False
            status = budget.reason or "stopped"
            break
        state = frontier.pop()
        if prune and state.cost >= best_cost:
            pruned += 1
            continue
        if problem.is_goal(state):
            if state.cost < best_cost:
                best, best_cost = state, state.cost
            continue
        expanded += 1
        children = list(problem.successors(state))
        generated += len(children)
        # push in reverse so that children are explored in generation order
        for child in reversed(children):
            frontier.push(child)
        size = len(frontier)
        if size > max_frontier:
            max_frontier = size

    if best is None and completed:
        status = "no-tour"
    return SearchResult(
        algorithm="DFS" + (" (B&B)" if prune else ""),
        path=best.path if best is not None else None,
        cost=best_cost,
        nodes_expanded=expanded,
        nodes_generated=generated,
        max_frontier=max_frontier,
        completed=completed,
        exact=completed and best is not None,
        status=status,
        extra={"pruned": pruned} if prune else {},
    )
