"""Benchmark: every algorithm x every scenario x n = 5..20 (task step 4).

For each run we record
    cost            cost of the tour that was returned (inf when none)
    time_s          wall-clock time of the search
    peak_memory_kb  peak Python heap allocated during the search (tracemalloc)
    max_frontier    largest number of states kept in memory by the algorithm
    nodes_expanded / nodes_generated
    status          ok | timeout | node-limit | no-tour | skipped (+nn-completion)

Time and memory are measured in two separate passes of the same deterministic
run: ``tracemalloc`` slows allocation-heavy searches down by an order of
magnitude, so timing is taken from an untraced pass and the peak memory from a
traced pass with the same budget (``memory_completed`` tells whether the traced
pass finished within the budget as well; when it did not, the memory figure is
a lower bound and is flagged with ``*`` in the tables).

Exact methods whose budget ran out at some n are skipped for all larger n of
the same scenario (they would only run out again), which keeps the whole
benchmark to a reasonable duration.
"""
from __future__ import annotations

import csv
import random
import tracemalloc
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .algorithms import (MSTHeuristic, NNCompletionHeuristic, WeightedHeuristic, aco, astar, bfs,
                         dfs, dijkstra, greedy_nn)
from .cities import SCENARIOS, Scenario, make_instance
from .graph import INF
from .problem import Budget, SearchResult, TSPProblem

Runner = Callable[[TSPProblem, Budget, random.Random], SearchResult]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    run: Runner
    exact: bool          # exact methods get the "skip after budget exhaustion" rule
    short: str = ""

    @property
    def label(self) -> str:
        return self.short or self.name


def default_algorithms() -> List[AlgorithmSpec]:
    return [
        AlgorithmSpec("BFS", lambda p, b, r: bfs(p, b), True, "BFS"),
        AlgorithmSpec("DFS", lambda p, b, r: dfs(p, b), True, "DFS"),
        AlgorithmSpec("Greedy NN", lambda p, b, r: greedy_nn(p, b), False, "NN"),
        AlgorithmSpec("Dijkstra (UCS)", lambda p, b, r: dijkstra(p, b), True, "Dijkstra"),
        AlgorithmSpec("A* MST (admissible)",
                      lambda p, b, r: astar(p, MSTHeuristic(p), b, name="A* MST (admissible)"), True, "A*-MST"),
        AlgorithmSpec("A* 2xMST (inadmissible)",
                      lambda p, b, r: astar(p, WeightedHeuristic(MSTHeuristic(p), 2.0), b,
                                            name="A* 2xMST (inadmissible)"), False, "A*-2xMST"),
        AlgorithmSpec("A* NN-completion (inadmissible)",
                      lambda p, b, r: astar(p, NNCompletionHeuristic(p), b,
                                            name="A* NN-completion (inadmissible)"), False, "A*-NN"),
        AlgorithmSpec("ACO", lambda p, b, r: aco(p, b, rng=r), False, "ACO"),
    ]


def optional_algorithms() -> List[AlgorithmSpec]:
    """Extra variants that can be enabled from the command line."""
    return [
        AlgorithmSpec("DFS B&B", lambda p, b, r: dfs(p, b, prune=True), True, "DFS-B&B"),
        AlgorithmSpec("Greedy NN (no backtracking)", lambda p, b, r: greedy_nn(p, b, backtrack=False),
                      False, "NN-pure"),
        AlgorithmSpec("Dijkstra tree (no pruning)", lambda p, b, r: dijkstra(p, b, prune_dominated=False),
                      True, "Dijkstra-tree"),
        AlgorithmSpec("A* MST tree (no pruning)",
                      lambda p, b, r: astar(p, MSTHeuristic(p), b, prune_dominated=False,
                                            name="A* MST tree (no pruning)"), True, "A*-MST-tree"),
    ]


@dataclass
class RunRecord:
    scenario: str
    n: int
    algorithm: str
    status: str
    cost: float
    time_s: float
    peak_memory_kb: float
    nodes_expanded: int
    nodes_generated: int
    max_frontier: int
    completed: bool
    exact: bool
    memory_completed: bool
    path: str

    @property
    def found(self) -> bool:
        return self.cost != INF and self.status != "skipped"


CSV_FIELDS = [f.name for f in fields(RunRecord)]


def _execute(spec: AlgorithmSpec, problem: TSPProblem, time_limit: Optional[float],
             node_limit: Optional[int], seed: int, trace_memory: bool):
    """One execution of ``spec``; returns (result, elapsed seconds, peak bytes)."""
    rng = random.Random(seed)
    budget = Budget(time_limit, node_limit)
    if trace_memory:
        tracemalloc.start()
    t0 = perf_counter()
    try:
        result = spec.run(problem, budget, rng)
    finally:
        elapsed = perf_counter() - t0
        peak = 0
        if trace_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    if result.path is not None:
        check = problem.path_cost(result.path)
        if abs(check - result.cost) > 1e-6:
            raise AssertionError(f"{spec.name} returned an inconsistent tour: {result.cost} vs {check}")
    return result, elapsed, peak


def run_single(spec: AlgorithmSpec, problem: TSPProblem, scenario_name: str, time_limit: Optional[float],
               node_limit: Optional[int], seed: int, trace_memory: bool = True) -> RunRecord:
    """Timing pass (untraced) + optional memory pass (tracemalloc) of the same deterministic run."""
    result, elapsed, _ = _execute(spec, problem, time_limit, node_limit, seed, trace_memory=False)
    peak = 0
    memory_completed = True
    if trace_memory:
        mem_result, _, peak = _execute(spec, problem, time_limit, node_limit, seed, trace_memory=True)
        memory_completed = mem_result.completed
    return RunRecord(
        scenario=scenario_name, n=problem.n, algorithm=spec.name, status=result.status, cost=result.cost,
        time_s=elapsed, peak_memory_kb=peak / 1024.0, nodes_expanded=result.nodes_expanded,
        nodes_generated=result.nodes_generated, max_frontier=result.max_frontier,
        completed=result.completed, exact=result.exact, memory_completed=memory_completed,
        path="-".join(str(c) for c in result.path) if result.path else "",
    )


def run_benchmark(n_values: Sequence[int], scenarios: Sequence[Scenario] = SCENARIOS,
                  specs: Optional[Sequence[AlgorithmSpec]] = None, seed: int = 42,
                  time_limit: Optional[float] = 30.0, node_limit: Optional[int] = 1_000_000,
                  trace_memory: bool = True, start: int = 0,
                  log: Optional[Callable[[str], None]] = print,
                  csv_path: Optional[Path] = None) -> List[RunRecord]:
    specs = list(specs) if specs is not None else default_algorithms()
    records: List[RunRecord] = []
    for scenario in scenarios:
        gave_up: Dict[str, int] = {}           # exact algorithm -> n at which the budget ran out
        for n in n_values:
            cities, graph = make_instance(n, scenario, seed)
            problem = TSPProblem(graph, start)
            for spec in specs:
                if spec.name in gave_up:
                    rec = RunRecord(scenario.name, n, spec.name, "skipped", INF, 0.0, 0.0, 0, 0, 0,
                                    False, False, False, "")
                else:
                    rec = run_single(spec, problem, scenario.name, time_limit, node_limit, seed, trace_memory)
                    if spec.exact and not rec.completed:
                        gave_up[spec.name] = n
                records.append(rec)
                if log:
                    cost = "-" if rec.status == "skipped" else ("none" if rec.cost == INF else f"{rec.cost:8.2f}")
                    log(f"[{scenario.name:>10}] n={n:2d} {spec.name:<34} cost={cost:>8} "
                        f"time={rec.time_s:8.3f}s mem={rec.peak_memory_kb:10.1f}KB "
                        f"frontier={rec.max_frontier:>8} status={rec.status}")
                if csv_path is not None:
                    write_csv(records, csv_path)
    return records


# ------------------------------------------------------------------ CSV I/O
def write_csv(records: Iterable[RunRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def read_csv(path: Path) -> List[RunRecord]:
    records = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            records.append(RunRecord(
                scenario=row["scenario"], n=int(row["n"]), algorithm=row["algorithm"], status=row["status"],
                cost=float(row["cost"]), time_s=float(row["time_s"]), peak_memory_kb=float(row["peak_memory_kb"]),
                nodes_expanded=int(row["nodes_expanded"]), nodes_generated=int(row["nodes_generated"]),
                max_frontier=int(row["max_frontier"]), completed=row["completed"] == "True",
                exact=row["exact"] == "True", memory_completed=row.get("memory_completed", "True") == "True",
                path=row["path"]))
    return records


# ------------------------------------------------------------- summaries
def _fmt_cost(rec: RunRecord) -> str:
    if rec.status == "skipped":
        return "-"
    if rec.cost == INF:
        return "none"
    return f"{rec.cost:.1f}" + ("" if rec.completed else "*")


def _fmt_time(rec: RunRecord) -> str:
    if rec.status == "skipped":
        return "-"
    return f"{rec.time_s:.3f}" + ("" if rec.completed else "*")


def _fmt_mem(rec: RunRecord) -> str:
    if rec.status == "skipped":
        return "-"
    kb = rec.peak_memory_kb
    s = f"{kb / 1024:.1f}M" if kb >= 1024 else f"{kb:.0f}K"
    return s + ("" if rec.memory_completed else "*")


def _fmt_ratio(rec: RunRecord, best: float) -> str:
    if rec.status == "skipped":
        return "-"
    if rec.cost == INF or best == INF:
        return "none"
    return f"{rec.cost / best:.3f}" + ("" if rec.completed else "*")


def _table(title: str, rows: List[int], cols: List[str], cell: Callable[[int, str], str],
           short: Dict[str, str]) -> str:
    width = max(9, max(len(short.get(c, c)) for c in cols) + 1)
    lines = [title, "  n " + "".join(short.get(c, c).rjust(width) for c in cols)]
    for n in rows:
        lines.append(f"{n:3d} " + "".join(cell(n, c).rjust(width) for c in cols))
    return "\n".join(lines)


def summary_tables(records: Sequence[RunRecord], specs: Optional[Sequence[AlgorithmSpec]] = None) -> str:
    specs = list(specs) if specs is not None else default_algorithms() + optional_algorithms()
    short = {s.name: s.label for s in specs}
    by_key: Dict[tuple, RunRecord] = {(r.scenario, r.n, r.algorithm): r for r in records}
    scenarios = list(dict.fromkeys(r.scenario for r in records))
    algos = list(dict.fromkeys(r.algorithm for r in records))
    ns = sorted(set(r.n for r in records))
    out: List[str] = []
    legend = ("Legend: * = stopped by the time/memory budget (approximate or partial result; "
              "in the memory table: the traced pass was cut, value is a lower bound), "
              "- = skipped (budget already exhausted at a smaller n), none = no tour returned")
    for sc in scenarios:
        def rec(n: int, a: str) -> Optional[RunRecord]:
            return by_key.get((sc, n, a))

        def best_known(n: int) -> float:
            vals = [r.cost for r in (rec(n, a) for a in algos) if r is not None and r.cost != INF]
            return min(vals) if vals else INF

        def cost_cell(n: int, a: str) -> str:
            r = rec(n, a)
            return "" if r is None else _fmt_cost(r)

        def ratio_cell(n: int, a: str) -> str:
            r = rec(n, a)
            return "" if r is None else _fmt_ratio(r, best_known(n))

        def time_cell(n: int, a: str) -> str:
            r = rec(n, a)
            return "" if r is None else _fmt_time(r)

        def mem_cell(n: int, a: str) -> str:
            r = rec(n, a)
            return "" if r is None else _fmt_mem(r)

        def frontier_cell(n: int, a: str) -> str:
            r = rec(n, a)
            return "" if r is None else ("-" if r.status == "skipped" else str(r.max_frontier))

        out.append("=" * 100)
        out.append(f"Scenario: {sc}")
        out.append("=" * 100)
        out.append(_table("Tour cost", ns, algos, cost_cell, short))
        out.append("")
        out.append(_table("Cost / best known cost", ns, algos, ratio_cell, short))
        out.append("")
        out.append(_table("Time [s]", ns, algos, time_cell, short))
        out.append("")
        out.append(_table("Peak memory (tracemalloc)", ns, algos, mem_cell, short))
        out.append("")
        out.append(_table("Max states in memory", ns, algos, frontier_cell, short))
        out.append("")
    out.append(legend)
    return "\n".join(out)
