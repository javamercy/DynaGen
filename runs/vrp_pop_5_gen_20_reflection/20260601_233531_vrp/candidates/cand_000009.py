import numpy as np
import random
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    customers = list(range(1, n))
    routes = [[0, c, 0] for c in customers]
    dists = [2 * distance_matrix[0, c] for c in customers]
    initial_remaining = len(routes)
    while len(routes) > truck_count:
        remaining = len(routes)
        pairs = list(combinations(range(remaining), 2))
        # compute score for each pair: (new_max, new_dist)
        scored = []
        current_max = max(dists)
        for i, j in pairs:
            r1 = routes[i]
            r2 = routes[j]
            # merged distance: r1 without last depot + r2 without first depot
            new_dist = (dists[i] - distance_matrix[r1[-2], 0] - distance_matrix[0, r2[1]] +
                        distance_matrix[r1[-2], r2[1]])
            new_max = max(current_max, new_dist)
            scored.append((new_max, new_dist, i, j))
        scored.sort(key=lambda x: (x[0], x[1]))
        # adaptive selection
        p = 0.5 * (remaining / initial_remaining)
        p = max(0.05, min(0.5, p))
        k = max(1, int(len(scored) * 0.1))
        if random.random() < p:
            idx = random.randrange(k)
        else:
            idx = 0
        best = scored[idx]
        i, j = best[2], best[3]
        new_dist = best[1]
        r1 = routes[i]
        r2 = routes[j]
        merged = r1[:-1] + r2[1:]
        routes[i] = merged
        dists[i] = new_dist
        # remove larger index first
        del routes[j]
        del dists[j]
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)
    report_best_vrp(routes)
    return routes