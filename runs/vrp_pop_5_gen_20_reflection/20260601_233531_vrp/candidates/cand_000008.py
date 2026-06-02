import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    routes = [[0, c, 0] for c in customers]
    dists = [2 * distance_matrix[0, c] for c in customers]
    while len(routes) > truck_count:
        best = None
        best_i = best_j = None
        best_new_total = float('inf')
        best_new_max = float('inf')
        current_total = sum(dists)
        for i, j in combinations(range(len(routes)), 2):
            r1 = routes[i]
            r2 = routes[j]
            new_dist = (dists[i] - distance_matrix[r1[-2], 0] - distance_matrix[0, r2[1]] +
                        distance_matrix[r1[-2], r2[1]])
            new_total = current_total - dists[i] - dists[j] + new_dist
            new_max = max(new_dist, max(dists[:i] + dists[i+1:j] + dists[j+1:]), default=0.0)
            if (new_total < best_new_total or
                (new_total == best_new_total and new_max < best_new_max)):
                best_new_total = new_total
                best_new_max = new_max
                best = (i, j, new_dist, r1, r2)
        i, j, new_dist, r1, r2 = best
        merged = r1[:-1] + r2[1:]
        routes[i] = merged
        dists[i] = new_dist
        del routes[j]
        del dists[j]
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)
    report_best_vrp(routes)
    return routes