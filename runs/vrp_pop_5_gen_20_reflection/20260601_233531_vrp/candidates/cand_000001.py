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
        best_new_max = float('inf')
        best_new_dist = float('inf')
        current_max = max(dists)
        for i, j in combinations(range(len(routes)), 2):
            r1 = routes[i]
            r2 = routes[j]
            # compute new route distance if merged
            # option: r1 then r2
            new_dist = (dists[i] - distance_matrix[r1[-2], 0] - distance_matrix[0, r2[1]] +
                        distance_matrix[r1[-2], r2[1]])
            new_max = max(current_max, new_dist)
            if (new_max < best_new_max or
                (new_max == best_new_max and new_dist < best_new_dist)):
                best_new_max = new_max
                best_new_dist = new_dist
                best = (i, j, new_dist, r1, r2)
        i, j, new_dist, r1, r2 = best
        # build merged route: r1 without last depot + r2 without first depot
        merged = r1[:-1] + r2[1:]
        routes[i] = merged
        dists[i] = new_dist
        # remove j (larger index first to avoid index shift)
        del routes[j]
        del dists[j]
    # if we have fewer routes than truck_count, pad with empty routes
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)
    # report best
    report_best_vrp(routes)
    return routes