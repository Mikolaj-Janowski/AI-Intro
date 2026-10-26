"""Random city generation and the four test scenarios (task step 1).

Cities are points (x, y, z) with x, y in <-100, 100> and z (height) in <0, 50>.
The cost of going from city A to city B is the Euclidean distance between them
(if a road exists).  In the *asymmetric* variant the cost is increased by 10%
when going uphill (z_B > z_A) and decreased by 10% when going downhill.

Scenarios are built from two criteria:
    a. all direct connections  /  c.a. 80% of the possible connections
    b. symmetric costs         /  asymmetric costs
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from .graph import Graph

X_RANGE = (-100.0, 100.0)
Y_RANGE = (-100.0, 100.0)
Z_RANGE = (0.0, 50.0)
UPHILL_FACTOR = 1.10
DOWNHILL_FACTOR = 0.90


@dataclass(frozen=True)
class City:
    id: int
    x: float
    y: float
    z: float

    @property
    def name(self) -> str:
        return f"C{self.id}"

    def distance_to(self, other: "City") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def __str__(self) -> str:
        return f"{self.name}({self.x:7.2f}, {self.y:7.2f}, {self.z:5.2f})"


def generate_cities(n: int, rng: random.Random) -> List[City]:
    """n cities with coordinates drawn uniformly from the task ranges."""
    return [City(i, rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE), rng.uniform(*Z_RANGE)) for i in range(n)]


def travel_cost(a: City, b: City, symmetric: bool) -> float:
    """Cost of the road a -> b."""
    d = a.distance_to(b)
    if symmetric or b.z == a.z:
        return d
    return d * UPHILL_FACTOR if b.z > a.z else d * DOWNHILL_FACTOR


@dataclass(frozen=True)
class Scenario:
    full_connections: bool
    symmetric: bool
    connection_ratio: float = 0.8  # used only when full_connections is False

    @property
    def name(self) -> str:
        conn = "full" if self.full_connections else f"{int(round(self.connection_ratio * 100))}pct"
        return f"{conn}-{'sym' if self.symmetric else 'asym'}"

    @property
    def description(self) -> str:
        conn = ("all direct connections" if self.full_connections
                else f"c.a. {self.connection_ratio * 100:.0f}% of possible connections")
        cost = "symmetric costs" if self.symmetric else "asymmetric costs (uphill +10%, downhill -10%)"
        return f"{conn}, {cost}"


SCENARIOS: List[Scenario] = [
    Scenario(full_connections=True, symmetric=True),
    Scenario(full_connections=True, symmetric=False),
    Scenario(full_connections=False, symmetric=True),
    Scenario(full_connections=False, symmetric=False),
]
SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


def choose_roads(n: int, scenario: Scenario, rng: random.Random,
                 ensure_tour_exists: bool = True) -> Set[Tuple[int, int]]:
    """Select which (undirected) roads exist, as a set of pairs (u, v) with u < v.

    A road is a physical object and can be driven in both directions; in the
    asymmetric scenario the two directions simply have different costs.

    With ``ensure_tour_exists`` the roads of one random Hamiltonian cycle are
    always kept, so that the TSP instance has at least one feasible tour even
    with only 80% of the connections (otherwise small instances frequently have
    no valid tour at all and nothing could be compared).
    """
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    if scenario.full_connections:
        return set(all_pairs)

    target = int(round(scenario.connection_ratio * len(all_pairs)))
    chosen: Set[Tuple[int, int]] = set()
    if ensure_tour_exists:
        order = list(range(n))
        rng.shuffle(order)
        for i in range(n):
            u, v = order[i], order[(i + 1) % n]
            chosen.add((min(u, v), max(u, v)))
    remaining = [p for p in all_pairs if p not in chosen]
    rng.shuffle(remaining)
    for p in remaining:
        if len(chosen) >= target:
            break
        chosen.add(p)
    return chosen


def build_graph(cities: Sequence[City], scenario: Scenario, rng: random.Random,
                ensure_tour_exists: bool = True) -> Graph:
    """Map -> weighted directed graph (task step 2)."""
    n = len(cities)
    graph = Graph(n, names=[c.name for c in cities], cities=list(cities))
    for u, v in sorted(choose_roads(n, scenario, rng, ensure_tour_exists)):
        graph.add_edge(u, v, travel_cost(cities[u], cities[v], scenario.symmetric))
        graph.add_edge(v, u, travel_cost(cities[v], cities[u], scenario.symmetric))
    return graph


def make_instance(n: int, scenario: Scenario, seed: int, pool_size: int = 20):
    """Deterministic instance (cities, graph) for a given size and scenario.

    All sizes share one seeded pool of cities, so the instance with n cities is
    a prefix of the instance with n + 1 cities (same cities in every scenario).
    """
    rng = random.Random(seed)
    cities = generate_cities(max(n, pool_size), rng)[:n]
    road_rng = random.Random(f"{seed}-{scenario.name}-{n}")
    graph = build_graph(cities, scenario, road_rng)
    return cities, graph


def format_cities(cities: Sequence[City]) -> str:
    lines = ["  name        x        y      z"]
    for c in cities:
        lines.append(f"  {c.name:<5} {c.x:8.2f} {c.y:8.2f} {c.z:6.2f}")
    return "\n".join(lines)
