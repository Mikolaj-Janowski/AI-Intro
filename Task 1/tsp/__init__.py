"""Travelling Salesman Problem on a randomly generated 3D city map.

Package layout
--------------
cities.py          - random city generation, the four scenarios, travel-cost rule (task step 1)
graph.py           - weighted directed graph (task step 2)
datastructures.py  - hand-written FIFO queue, stack and binary min-heap
problem.py         - TSP search problem: states, successor function, state tree (task step 3)
algorithms/        - BFS, DFS, greedy NN, Dijkstra/UCS, A* (+ heuristics), ACO
benchmark.py       - cost / time / memory measurements (task step 4)
plotting.py        - optional matplotlib figures
"""
