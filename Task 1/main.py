"""Command-line entry point.

    python main.py demo       [--n 7] [--scenario full-sym|full-asym|80pct-sym|80pct-asym|all]
                              [--seed 42] [--start 0] [--tree] [--matrix] [--plot]
    python main.py tree       [--n 5] [--scenario ...] [--seed 42]      (print the full state tree)
    python main.py benchmark  [--n-min 5] [--n-max 20] [--time-limit 30] [--node-limit 1000000]
                              [--seed 42] [--out results] [--scenario ...] [--algorithms ...]
                              [--extras] [--no-memory] [--no-plots]
    python main.py report     [--csv results/benchmark.csv] [--out results]   (tables + plots from a CSV)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from tsp.benchmark import (default_algorithms, optional_algorithms, read_csv, run_benchmark,
                           run_single, summary_tables, write_csv)
from tsp.cities import SCENARIOS, SCENARIOS_BY_NAME, format_cities, make_instance
from tsp.graph import INF
from tsp.problem import TSPProblem, build_state_tree, format_state_tree


def _scenarios(arg: str):
    if arg == "all":
        return list(SCENARIOS)
    out = []
    for name in arg.split(","):
        if name not in SCENARIOS_BY_NAME:
            sys.exit(f"unknown scenario '{name}', choose from: {', '.join(SCENARIOS_BY_NAME)}, all")
        out.append(SCENARIOS_BY_NAME[name])
    return out


def _specs(names: str, extras: bool):
    pool = default_algorithms() + (optional_algorithms() if extras else [])
    if names == "all":
        return pool
    by_label = {s.label.lower(): s for s in pool}
    by_name = {s.name.lower(): s for s in pool}
    out = []
    for item in names.split(","):
        key = item.strip().lower()
        spec = by_label.get(key) or by_name.get(key)
        if spec is None:
            sys.exit(f"unknown algorithm '{item}', choose from: " + ", ".join(s.label for s in pool))
        out.append(spec)
    return out


# ----------------------------------------------------------------- commands
def cmd_demo(args: argparse.Namespace) -> None:
    specs = _specs(args.algorithms, args.extras)
    out_dir = Path(args.out)
    for scenario in _scenarios(args.scenario):
        cities, graph = make_instance(args.n, scenario, args.seed)
        problem = TSPProblem(graph, args.start)
        print("=" * 100)
        print(f"Scenario {scenario.name}: {scenario.description}")
        print("=" * 100)
        print("Cities (x, y in <-100,100>, z in <0,50>):")
        print(format_cities(cities))
        print()
        print(graph.summary())
        if args.matrix or args.n <= 10:
            print("Cost matrix (row = from, column = to, '-' = no road):")
            print(graph.format_matrix())
        print()
        if args.tree:
            try:
                root = build_state_tree(problem)
                print(f"State tree from {graph.names[args.start]}: {root.size()} nodes, {root.leaves()} leaves")
                print(format_state_tree(problem, root))
            except ValueError as exc:
                print(f"State tree not printed: {exc}")
            print()
        print(f"TSP from {graph.names[args.start]} - results "
              f"(time limit {args.time_limit}s, node limit {args.node_limit}):")
        header = f"{'algorithm':<34}{'cost':>10}{'time[s]':>10}{'mem[KB]':>11}{'expanded':>10}{'generated':>11}{'states':>9}  status"
        print(header)
        print("-" * len(header))
        best_path = None
        best_cost = INF
        for spec in specs:
            rec = run_single(spec, problem, scenario.name, args.time_limit, args.node_limit, args.seed)
            cost = "none" if rec.cost == INF else f"{rec.cost:.2f}" + ("" if rec.completed else "*")
            print(f"{spec.name:<34}{cost:>10}{rec.time_s:>10.3f}{rec.peak_memory_kb:>11.1f}"
                  f"{rec.nodes_expanded:>10}{rec.nodes_generated:>11}{rec.max_frontier:>9}  {rec.status}")
            if rec.path:
                path = tuple(int(c) for c in rec.path.split("-"))
                print(f"{'':<34}  {problem.describe_path(path)}")
                if rec.cost < best_cost:
                    best_cost, best_path = rec.cost, path
        print("* = stopped by the budget (approximate / partial result)")
        if args.plot:
            from tsp.plotting import plot_map
            f = plot_map(graph, best_path, title=f"{scenario.name}, n={args.n}, best tour cost {best_cost:.1f}",
                         out_file=out_dir / f"map_{scenario.name}_n{args.n}.png")
            print(f"Map written to {f}")
        print()


def cmd_tree(args: argparse.Namespace) -> None:
    scenario = _scenarios(args.scenario)[0]
    cities, graph = make_instance(args.n, scenario, args.seed)
    problem = TSPProblem(graph, args.start)
    print(format_cities(cities))
    print(graph.summary())
    print(graph.format_matrix())
    root = build_state_tree(problem)
    print(f"\nState tree from {graph.names[args.start]}: {root.size()} nodes, {root.leaves()} leaves "
          f"(each leaf is either a closed tour or a dead end)")
    print(format_state_tree(problem, root))


def cmd_benchmark(args: argparse.Namespace) -> None:
    specs = _specs(args.algorithms, args.extras)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark.csv"
    n_values = list(range(args.n_min, args.n_max + 1))
    t0 = perf_counter()
    records = run_benchmark(n_values, _scenarios(args.scenario), specs, seed=args.seed,
                            time_limit=args.time_limit, node_limit=args.node_limit,
                            trace_memory=not args.no_memory, start=args.start, csv_path=csv_path)
    write_csv(records, csv_path)
    text = summary_tables(records, specs)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"\nTotal benchmark time: {perf_counter() - t0:.1f}s")
    print(f"Raw results: {csv_path}\nTables:      {out_dir / 'summary.txt'}")
    if not args.no_plots:
        from tsp.plotting import plot_benchmark
        files = plot_benchmark(records, out_dir, {s.name: s.label for s in specs})
        print("Plots:       " + ", ".join(str(f) for f in files))


def cmd_report(args: argparse.Namespace) -> None:
    records = read_csv(Path(args.csv))
    specs = default_algorithms() + optional_algorithms()
    text = summary_tables(records, specs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    if not args.no_plots:
        from tsp.plotting import plot_benchmark
        files = plot_benchmark(records, out_dir, {s.name: s.label for s in specs})
        print("Plots: " + ", ".join(str(f) for f in files))


# -------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSP on a random 3D city map: tree search, greedy, A*, ACO")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--start", type=int, default=0, help="index of the start city")
        p.add_argument("--scenario", default="all", help="full-sym, full-asym, 80pct-sym, 80pct-asym or all")

    p = sub.add_parser("demo", help="solve one instance with every algorithm")
    common(p)
    p.add_argument("--n", type=int, default=7)
    p.add_argument("--tree", action="store_true", help="print the whole state tree (small n only)")
    p.add_argument("--matrix", action="store_true", help="print the cost matrix even for large n")
    p.add_argument("--plot", action="store_true", help="save a 3D map with the best tour")
    p.add_argument("--time-limit", type=float, default=30.0)
    p.add_argument("--node-limit", type=int, default=1_000_000)
    p.add_argument("--algorithms", default="all")
    p.add_argument("--extras", action="store_true", help="include the optional algorithm variants")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("tree", help="print the state tree of a small instance")
    common(p)
    p.add_argument("--n", type=int, default=5)
    p.set_defaults(func=cmd_tree)

    p = sub.add_parser("benchmark", help="run every algorithm for n = n-min..n-max in every scenario")
    common(p)
    p.add_argument("--n-min", type=int, default=5)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--time-limit", type=float, default=30.0, help="seconds per run (0 = unlimited)")
    p.add_argument("--node-limit", type=int, default=1_000_000, help="max states in memory per run (0 = unlimited)")
    p.add_argument("--algorithms", default="all", help="comma separated labels, e.g. BFS,DFS,NN,ACO")
    p.add_argument("--extras", action="store_true", help="include the optional algorithm variants")
    p.add_argument("--no-memory", action="store_true", help="do not trace memory (faster, raw timings)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("report", help="re-create tables and plots from an existing CSV")
    p.add_argument("--csv", default="results/benchmark.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if hasattr(args, "time_limit") and args.time_limit is not None and args.time_limit <= 0:
        args.time_limit = None
    if hasattr(args, "node_limit") and args.node_limit is not None and args.node_limit <= 0:
        args.node_limit = None
    args.func(args)


if __name__ == "__main__":
    main()
