"""Optional matplotlib figures: the city map / graph and the benchmark charts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .benchmark import RunRecord
from .graph import INF, Graph


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_map(graph: Graph, path: Optional[Sequence[int]] = None, title: str = "",
             out_file: Optional[Path] = None) -> Path:
    """3D view of the cities, the roads and (optionally) a tour."""
    plt = _plt()
    cities = graph.cities
    if cities is None:
        raise ValueError("graph has no city coordinates")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    drawn = set()
    for u, v, _ in graph.edges():
        key = (min(u, v), max(u, v))
        if key in drawn:
            continue
        drawn.add(key)
        a, b = cities[u], cities[v]
        ax.plot([a.x, b.x], [a.y, b.y], [a.z, b.z], color="lightgray", linewidth=0.6, zorder=1)
    if path:
        xs = [cities[c].x for c in path]
        ys = [cities[c].y for c in path]
        zs = [cities[c].z for c in path]
        ax.plot(xs, ys, zs, color="tab:red", linewidth=2.0, zorder=3)
    ax.scatter([c.x for c in cities], [c.y for c in cities], [c.z for c in cities],
               c=[c.z for c in cities], cmap="viridis", s=40, zorder=4, depthshade=False)
    for c in cities:
        ax.text(c.x, c.y, c.z + 1.5, c.name, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (height)")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(0, 50)
    ax.set_title(title)
    out_file = Path(out_file) if out_file else Path("map.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)
    return out_file


def plot_benchmark(records: Sequence[RunRecord], out_dir: Path, short: Optional[Dict[str, str]] = None) -> List[Path]:
    """One figure per metric; each figure has one panel per scenario."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = short or {}
    scenarios = list(dict.fromkeys(r.scenario for r in records))
    algos = list(dict.fromkeys(r.algorithm for r in records))
    metrics = [
        ("cost", "Tour cost", lambda r: r.cost, False, lambda r: r.completed),
        ("cost_ratio", "Cost / best known cost", None, False, lambda r: r.completed),
        ("time", "Time [s] (log scale)", lambda r: r.time_s, True, lambda r: r.completed),
        ("memory", "Peak memory [KB] (log scale)", lambda r: r.peak_memory_kb, True,
         lambda r: r.memory_completed),
        ("frontier", "Max states in memory (log scale)", lambda r: r.max_frontier, True, lambda r: r.completed),
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    files = []
    for key, label, getter, log, is_complete in metrics:
        ncols = 2
        nrows = (len(scenarios) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), squeeze=False)
        for idx, sc in enumerate(scenarios):
            ax = axes[idx // ncols][idx % ncols]
            sc_recs = [r for r in records if r.scenario == sc]
            best: Dict[int, float] = {}
            for r in sc_recs:
                if r.cost != INF and r.status != "skipped":
                    best[r.n] = min(best.get(r.n, INF), r.cost)
            for ai, a in enumerate(algos):
                pts_ok = []
                pts_partial = []
                for r in sorted((r for r in sc_recs if r.algorithm == a), key=lambda r: r.n):
                    if r.status == "skipped":
                        continue
                    if getter is None:
                        if r.cost == INF or r.n not in best:
                            continue
                        y = r.cost / best[r.n]
                    else:
                        y = getter(r)
                        if y == INF:
                            continue
                        if log and y <= 0:
                            y = 1e-6
                    (pts_ok if is_complete(r) else pts_partial).append((r.n, y))
                m = markers[ai % len(markers)]
                lbl = short.get(a, a)
                if pts_ok or pts_partial:
                    all_pts = sorted(pts_ok + pts_partial)
                    line, = ax.plot([p[0] for p in all_pts], [p[1] for p in all_pts], marker=m,
                                    markersize=4, linewidth=1.1, label=lbl, alpha=0.9)
                    if pts_partial:
                        ax.scatter([p[0] for p in pts_partial], [p[1] for p in pts_partial],
                                   facecolors="none", edgecolors=line.get_color(), s=90, linewidths=1.3)
            ax.set_title(sc)
            ax.set_xlabel("n (cities)")
            ax.set_ylabel(label)
            ax.set_xticks(sorted(set(r.n for r in sc_recs)))
            if log:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
        for idx in range(len(scenarios), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), frameon=False)
        fig.suptitle(label + "   (hollow circle = stopped by the budget; partial/approximate result)")
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        path = out_dir / f"benchmark_{key}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        files.append(path)
    return files
