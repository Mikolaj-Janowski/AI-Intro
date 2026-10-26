# Travelling Salesman on a random 3D city map

Pure-Python (3.8+) implementation of the task: random cities in 3D, four road-network
scenarios, the TSP expressed as a state-tree search, and six families of solvers
(BFS, DFS, greedy NN, Dijkstra/UCS, A* with admissible and inadmissible heuristics,
ACO) benchmarked for n = 5…20 cities in terms of tour cost, time and memory.

Every search algorithm, heuristic and container (FIFO queue, stack, binary heap) is
written from scratch in this repository; the standard library is used only for
`random`, `math`, timing (`time.perf_counter`), memory tracing (`tracemalloc`) and
CSV output. `matplotlib` is optional and only used for the figures.

```
main.py                   command-line interface (demo / tree / benchmark / report)
tsp/cities.py             step 1 - cities, scenarios, travel-cost rule
tsp/graph.py              step 2 - weighted directed graph
tsp/problem.py            step 3 - states, successor function, goal test, explicit state tree
tsp/datastructures.py     Queue, Stack, MinHeap
tsp/algorithms/bfs.py     3a  breadth-first full search
tsp/algorithms/dfs.py     3a  depth-first full search (+ optional branch & bound)
tsp/algorithms/greedy.py  3b  nearest neighbour, Dijkstra (uniform-cost search)
tsp/algorithms/astar.py   3c  A* (best-first with f = g + h)
tsp/algorithms/heuristics.py  admissible: 0, min-in-edge, MST; inadmissible: NN-completion, w*MST
tsp/algorithms/aco.py     3d  elitist Ant System
tsp/benchmark.py          step 4 - measurements, CSV, summary tables
tsp/plotting.py           3D map and benchmark charts
results/                  output of the full benchmark run (CSV, tables, charts, log)
```

## Quick start

```powershell
python main.py demo --n 7                      # all four scenarios, every algorithm, one instance each
python main.py demo --n 6 --scenario 80pct-asym --tree --plot   # also print the state tree + save a 3D map
python main.py tree --n 5 --scenario full-sym  # just the state tree of a small instance
python main.py benchmark                       # n = 5..20, 4 scenarios, 8 algorithms  (~30-40 min)
python main.py benchmark --n-max 12 --time-limit 10 --algorithms BFS,DFS,NN,ACO
python main.py report --csv results/benchmark.csv   # rebuild tables/plots from a CSV
```

Useful options: `--seed`, `--start` (start city), `--time-limit` / `--node-limit`
(budget per run), `--extras` (adds DFS branch-and-bound, NN without backtracking and
the tree-search variants of Dijkstra/A* without dominance pruning), `--no-memory`
(skip the memory pass), `--no-plots`.

## 1. Map and scenarios (`tsp/cities.py`)

* `n` cities with `x, y ~ U(-100, 100)` and `z ~ U(0, 50)` drawn from a seeded `random.Random`.
  All instance sizes share one seeded pool of 20 cities, so the instance with `n` cities
  is a prefix of the instance with `n + 1` cities and the same cities are used in every scenario.
* Cost of the road A → B = Euclidean distance in 3D. Asymmetric variant:
  `×1.10` when going uphill (`z_B > z_A`), `×0.90` when going downhill.
* Scenarios (`SCENARIOS`): `full-sym`, `full-asym`, `80pct-sym`, `80pct-asym`.
  A road is a physical object usable in both directions; in the asymmetric case the two
  directions simply get different costs. With 80 % of the connections,
  `round(0.8 · n(n−1)/2)` undirected pairs are chosen at random. The roads of one random
  Hamiltonian cycle are always kept (`ensure_tour_exists=True`) so that every instance has at
  least one feasible tour – otherwise small 80 % instances frequently have no valid tour at all
  and nothing could be compared. Which roads are missing still differs from city to city, so
  dead ends in the state tree are common (see the tree printout in the demo).

## 2. Graph (`tsp/graph.py`)

`Graph` is a directed weighted graph stored as one adjacency dictionary per node
(`cost(u, v)` is `+inf` when there is no road). It can print the cost matrix, report
density/symmetry and export Graphviz DOT.

## 3. The TSP as a state tree (`tsp/problem.py`)

* **State** = the path from the start city taken so far + its cost `g`
  (`State.path`, `State.mask` – visited-set bitmask, `State.cost`).
* Root: `(start,)`. **Successors**: extend the path with any unvisited city reachable by a
  road; when all cities are visited the only child is the return to the start city.
* **Goal**: a closed tour with `n + 1` entries. Leaves that cannot be extended (no road to an
  unvisited city / no road home) are dead ends.
* `build_state_tree` / `format_state_tree` materialise and pretty-print the whole tree for
  small `n` (`python main.py tree`). For a full graph the tree has
  `Σ_k (n−1)!/(n−1−k)!` nodes and `(n−1)!` tours – 13 nodes for n=4, 153 for n=6, ≈ 10⁶ for n=10.

All algorithms work only through `initial_state / successors / is_goal`, so they are literally
searches over this tree. `SearchResult` carries the tour, its cost, `nodes_expanded`,
`nodes_generated`, `max_frontier` (largest number of states held at once), and whether the
run completed within its budget. `Budget` enforces a wall-clock limit and a limit on the number
of states kept in memory.

### 3a. Full search – BFS and DFS

* **BFS** (`bfs.py`): FIFO queue, level by level. All goals are leaves at depth `n`, so the whole
  tree is generated before the best tour is known; the frontier holds entire levels
  (`(n−1)!` states at the deepest level) → by far the most memory-hungry method.
* **DFS** (`dfs.py`): LIFO stack. Reaches a first tour after `n` steps and keeps the best one;
  memory is only `O(n²)`, but it still has to visit the whole tree to prove optimality.
  `prune=True` (label `DFS-B&B`, enabled with `--extras`) adds branch-and-bound.

Both are exact but `O((n−1)!)`: in the benchmark they complete up to n ≈ 10–11 within 30 s.

### 3b. Greedy – NN and Dijkstra

* **Nearest Neighbour** (`greedy_nn`): at every node take the child reached by the cheapest road.
  On an incomplete road network pure NN can dead-end (no road to any unvisited city or no road
  home). By default NN then *backtracks* – it becomes a depth-first walk with cost-ordered
  children that stops at the first closed tour (`extra["backtracks"]` counts how often that was
  needed; on full graphs it is 0 and the result is the classical NN tour). `--extras` adds the
  pure version (`NN-pure`) which simply reports "no tour" when stuck.
* **Dijkstra** (`dijkstra`): Dijkstra's algorithm on the state tree – always expand the frontier
  node with the smallest accumulated cost `g` (greedy on the cost-so-far instead of on the single
  next road). This is uniform-cost search ≡ A* with `h = 0`. Run to completion it is exact; its
  frontier grows exponentially, so under the budget the cheapest frontier state is completed
  with NN and the result is flagged as approximate (`status = node-limit+nn-completion`).

### 3c. A* with admissible / inadmissible heuristics

`best_first` (`astar.py`) keeps a binary min-heap ordered by `f = g + h` (ties → larger `g`).
Heuristics (`heuristics.py`) estimate the cost of the remaining path
*current → all unvisited → start*:

| heuristic | admissible | idea |
|---|---|---|
| `0` | yes | Dijkstra / UCS |
| `min-in-edge` | yes | Σ over every city still to be entered (unvisited + start) of its cheapest incoming road from {current} ∪ unvisited |
| `MST` | yes (and consistent) | weight of a minimum spanning tree (Prim) over {current} ∪ unvisited ∪ {start} on undirected weights `min(c(u,v), c(v,u))`; the remaining path is a spanning tree of that set, so it can never be cheaper |
| `NN-completion` | **no** | cost of finishing the tour with the NN rule – a real completion, so an upper bound |
| `2×MST` | **no** | weighted A* (`w = 2`) – finds a tour at most 2× the optimum, usually much better |

When the remaining cities are disconnected using the remaining roads the heuristic returns
`+inf` and the branch is pruned (still admissible: the true cost is `+inf` too).

**Dominance pruning** (`prune_dominated=True`, default for Dijkstra and A*): two tree nodes that
have visited the same *set* of cities and stand in the same city differ only in the order of the
visited cities; the more expensive one can never lead to a better tour and is dropped
(`closed[(visited_mask, current)] = best g`). The tree is still the search space – this just
avoids re-expanding dominated branches, exactly like the closed list of graph-search A*. With a
consistent heuristic the result is still optimal; the `--extras` variants `Dijkstra-tree` and
`A*-MST-tree` run without it for comparison.

### 3d. ACO

`aco.py` is an elitist Ant System: `n` ants per iteration build tours from the start city
choosing the next city with probability ∝ `τ^α · (1/cost)^β` among the unvisited neighbours;
ants that dead-end are discarded. Pheromone evaporates with rate `ρ`, every successful ant
deposits `Q/L`, the best-so-far tour additionally deposits `n·Q/L_best`. The pheromone matrix is
directed, so the asymmetric scenarios need no special treatment. Defaults: 200 iterations,
`α = 1, β = 3, ρ = 0.3, τ₀ = n / C_NN`, seeded RNG (deterministic runs).

## 4. Benchmark (`tsp/benchmark.py`)

`python main.py benchmark` runs every algorithm × scenario × n = 5…20 with a budget of 30 s
and 1 000 000 states per run and writes:

* `results/benchmark.csv` – one row per run (cost, time, peak memory, nodes expanded/generated,
  max states in memory, status, tour),
* `results/summary.txt` – per scenario: cost, cost / best-known, time, peak memory and max-states
  tables (`*` = stopped by the budget, `-` = skipped, `none` = no tour returned),
* `results/benchmark_*.png` – one chart per metric with one panel per scenario,
* `results/benchmark_log.txt` – the console log of the run.

Measurement details:

* Each run is deterministic (seeded), so **time is measured in an untraced pass and peak memory
  in a second pass under `tracemalloc`** (tracing slows allocation-heavy searches ~10×, which would
  distort the timings). When the traced pass itself hits the budget the memory value is a lower
  bound and is flagged with `*`. `max_frontier` (number of states held at once) is reported as a
  tracing-independent memory measure.
* Exact methods that run out of budget at some n are **skipped for larger n** of the same scenario
  (they would only run out again); the attempt that hit the budget is kept and flagged. BFS then
  reports no tour (it has not reached the leaves yet), DFS reports the best tour of the part of
  the tree it did visit, Dijkstra/A* report an NN completion of their cheapest frontier state.
* Every returned tour is re-validated against the graph (`TSPProblem.path_cost`), and the smoke
  test used during development compared all exact methods with brute-force enumeration of all
  permutations for n = 5…8 in every scenario.

### Results (seed 42, Python 3.12, Windows 10)

Full tables in `results/summary.txt`, raw data in `results/benchmark.csv`, charts in
`results/benchmark_{cost,cost_ratio,time,memory,frontier}.png`. 64 instances
(4 scenarios × n = 5…20), 8 algorithms, 512 runs, ≈ 28 min in total. Main observations:

* **BFS / DFS** are exact but factorial. Both finish n ≤ 9 in < 0.5 s and the largest instance
  they complete (n = 10 on full networks, n = 11 on 80 % networks) in 3–5 s; one step further BFS
  exceeds the 1 M-state limit after ~2.7 s (its deepest level alone would need 10! ≈ 3.6 M states)
  and DFS exceeds the 30 s limit (the best tour of the part it visited happened to be optimal).
  BFS memory grows like `(n−1)!`: 0.8 MB at n = 8, 10 MB at n = 9, 90 MB at n = 10, > 220 MB
  at the cutoff; DFS never needs more than a few KB (≤ 46 states) – the textbook BFS/DFS trade-off.
  The 80 % networks give a smaller tree (dead ends), which buys BFS/DFS exactly one extra n.
* **Greedy NN** answers in microseconds but is on average 10 % above the optimum
  (max 33 %, optimal on only 2 of 64 instances). On the 80 % networks pure NN hits a dead end on
  8 of 16 instances per scenario and only the backtracking variant returns a tour.
* **Dijkstra / UCS** is exact and – thanks to dominance pruning – far cheaper than BFS
  (`≤ n·2ⁿ` distinct (visited-set, city) pairs instead of `(n−1)!` paths): n = 13 in ~2 s,
  n = 15 in 12–18 s with ~310–465 k states and ~70 MB; at n = 16 it hits the 1 M-state limit
  in every scenario. Its NN completion of the cheapest frontier state is a poor approximation
  (1.5–1.6 × the optimum, no tour at all once) because the cheapest frontier state is a shallow,
  unpromising prefix.
* **A\* with the MST heuristic** is the only exact method that solves every instance up to n = 20
  within the budget: 0.03 s at n = 12, 1–13 s at n = 20 (worst case 17.6 s at n = 19, 80pct-asym),
  at most 469 k states / 62 MB. Compared with Dijkstra the heuristic cuts the number of expanded
  states by two to three orders of magnitude (e.g. n = 15 full-sym: 4 k vs 462 k states).
  The asymmetric and 80 % instances are noticeably harder for it (the undirected MST bound is
  looser), but it still finishes.
* **Inadmissible A\*** expands only ~n states and answers in milliseconds. Weighted A* (2×MST) is on
  average 6 % above the optimum (max 17 %), the NN-completion heuristic 3 % (max 10 %, optimal on
  15/64 instances) – better than plain NN at the same negligible cost.
* **ACO** (200 iterations, n ants) takes 0.01–0.6 s and is the best approximation: optimal on
  28/64 instances, on average 1.1 % above the optimum, never more than 6.4 % above it, with flat
  (polynomial) time and memory growth, independent of symmetry or road density.
* Symmetry does not change the size of the state tree, only which tour is optimal; the directed
  cost matrix / pheromone matrix handles it transparently. The missing roads of the 80 % scenarios
  shrink the tree (dead ends) but make the greedy methods less reliable (dead ends, discarded ants).
