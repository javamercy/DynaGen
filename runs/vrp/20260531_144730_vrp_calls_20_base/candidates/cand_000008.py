import numpy as np
import math
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    assert n > 0
    if truck_count <= 0:
        return []
    # initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(c, routes, route_dists):
        """Return (best_new_max, best_route_idx, best_pos, second_best_new_max)."""
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            # compute current max of other routes
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    # regret construction
    while unassigned:
        bests = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            bests.append((-regret, c, best_route, best_pos, best_new_max))
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)
        report_best_vrp(routes)

    # intra-route 2-opt improvement
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old_edges = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new_edges = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    delta = new_edges - old_edges
                    if delta < -1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        route_dists[r_idx] = route_dist(route)
                        break
                if improved:
                    break
    report_best_vrp(routes)

    # inter-route relocate improvement
    max_iter = n * truck_count
    for _ in range(max_iter):
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        moved = False
        route_max = routes[max_idx]
        for i in range(1, len(route_max)-1):
            c = route_max[i]
            pred = route_max[i-1]
            succ = route_max[i+1]
            new_max_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                best_other_new = float('inf')
                best_pos = -1
                for pos in range(1, len(other_route)):
                    pred_o = other_route[pos-1]
                    succ_o = other_route[pos]
                    new_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                    if new_other < best_other_new:
                        best_other_new = new_other
                        best_pos = pos
                other_max = 0.0
                for j, d in enumerate(route_dists):
                    if j != max_idx and j != other_idx and d > other_max:
                        other_max = d
                new_overall_max = max(other_max, new_max_dist, best_other_new)
                if new_overall_max < max_dist - 1e-12:
                    route_max.pop(i)
                    other_route.insert(best_pos, c)
                    route_dists[max_idx] = new_max_dist
                    route_dists[other_idx] = best_other_new
                    moved = True
                    report_best_vrp(routes)
                    break
            if moved:
                break
        if not moved:
            break
    return routes