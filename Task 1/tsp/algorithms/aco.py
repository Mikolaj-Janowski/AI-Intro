"""Ant Colony Optimisation for the TSP (task 3d) - elitist Ant System.

Every iteration ``n_ants`` artificial ants build a tour from the start city.
Standing in city i, an ant moves to an unvisited neighbour j with probability

    p(i -> j)  ~  tau(i, j)^alpha * eta(i, j)^beta,      eta(i, j) = 1 / cost(i, j)

where tau is the pheromone on road i -> j.  Ants that reach a dead end (no road
to an unvisited city or no road back home) are discarded.  After the iteration
the pheromone evaporates (tau <- (1 - rho) tau), every successful ant deposits
Q / L on the roads of its tour of length L, and - elitist variant - the best
tour found so far receives an additional ``elite_weight * Q / L_best``.

The pheromone matrix is directed, so the same code handles the symmetric and
the asymmetric scenarios.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from ..problem import INF, Budget, SearchResult, TSPProblem
from .greedy import nn_complete


def aco(problem: TSPProblem, budget: Optional[Budget] = None, rng: Optional[random.Random] = None,
        n_ants: Optional[int] = None, iterations: int = 200, alpha: float = 1.0, beta: float = 3.0,
        rho: float = 0.3, q: float = 1.0, elite_weight: Optional[float] = None,
        stagnation_limit: Optional[int] = None) -> SearchResult:
    rng = rng if rng is not None else random.Random(0)
    graph = problem.graph
    n = problem.n
    start = problem.start
    n_ants = n_ants or n
    elite_weight = n if elite_weight is None else elite_weight

    # neighbour lists with heuristic desirability eta = 1 / cost
    neigh: List[List[Tuple[int, float, float]]] = []
    for u in range(n):
        lst = [(v, w, (1.0 / w) ** beta if w > 0 else 1e12) for v, w in graph.neighbors(u)]
        neigh.append(lst)

    # initial pheromone tau0 = m / C_nn (classic choice)
    nn_goal, _ = nn_complete(problem, problem.initial_state(), backtrack=True)
    c_ref = nn_goal.cost if nn_goal is not None else n * graph.average_edge_cost()
    tau0 = n_ants / c_ref if c_ref not in (0.0, INF) else 1.0
    tau = [[tau0] * n for _ in range(n)]

    best_path: Optional[Tuple[int, ...]] = None
    best_cost = INF
    best_iter = -1
    ants_total = 0
    ants_failed = 0
    steps = 0
    iterations_run = 0
    completed = True
    status = "ok"

    for it in range(iterations):
        if budget is not None and budget.exceeded(n * n + n_ants * n):
            completed = False
            status = budget.reason or "stopped"
            break
        iterations_run = it + 1
        tours: List[Tuple[List[int], float]] = []
        for _ in range(n_ants):
            ants_total += 1
            path = [start]
            mask = 1 << start
            current = start
            cost = 0.0
            ok = True
            for _step in range(n - 1):
                steps += 1
                tau_row = tau[current]
                cands = []
                weights = []
                total = 0.0
                for v, w, eta_b in neigh[current]:
                    if not (mask >> v) & 1:
                        p = (tau_row[v] ** alpha) * eta_b
                        cands.append((v, w))
                        weights.append(p)
                        total += p
                if not cands:
                    ok = False
                    break
                r = rng.random() * total
                acc = 0.0
                chosen = len(cands) - 1
                for i, p in enumerate(weights):
                    acc += p
                    if acc >= r:
                        chosen = i
                        break
                v, w = cands[chosen]
                path.append(v)
                mask |= 1 << v
                cost += w
                current = v
            if ok:
                w_home = graph.cost(current, start)
                if w_home == INF:
                    ok = False
                else:
                    path.append(start)
                    cost += w_home
            if not ok:
                ants_failed += 1
                continue
            tours.append((path, cost))
            if cost < best_cost:
                best_cost = cost
                best_path = tuple(path)
                best_iter = it

        # pheromone update
        keep = 1.0 - rho
        for u in range(n):
            row = tau[u]
            for v in range(n):
                row[v] *= keep
        for path, cost in tours:
            d = q / cost
            for a, b in zip(path, path[1:]):
                tau[a][b] += d
        if best_path is not None and elite_weight > 0:
            d = elite_weight * q / best_cost
            for a, b in zip(best_path, best_path[1:]):
                tau[a][b] += d

        if stagnation_limit is not None and best_iter >= 0 and it - best_iter >= stagnation_limit:
            break

    return SearchResult(
        algorithm="ACO",
        path=best_path,
        cost=best_cost,
        nodes_expanded=ants_total,          # tours constructed
        nodes_generated=steps,              # single city-to-city moves
        max_frontier=n * n + n_ants * (n + 1),   # pheromone matrix + tours of one iteration
        completed=completed,
        exact=False,
        status=status if best_path is not None else "no-tour",
        extra={"iterations": iterations_run, "ants_failed": ants_failed, "best_iteration": best_iter,
               "alpha": alpha, "beta": beta, "rho": rho, "n_ants": n_ants, "elite_weight": elite_weight},
    )
