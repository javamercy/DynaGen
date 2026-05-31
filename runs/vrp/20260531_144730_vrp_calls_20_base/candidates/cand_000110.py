import numpy as np
import random
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []

    def report_best_vrp(routes):
        pass

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i + 1]]
        return d

    # Construction: best insertion to minimize new max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    customers = list(range(1, n))
    random.shuffle(customers)
    for c in customers:
        best_new_max = float('inf')
        best_route = None
        best_pos = None
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                pred = route[pos - 1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = new_dist
                for j, d in enumerate(route_dists):
                    if j != r_idx and d > new_max:
                        new_max = d
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route = r_idx
                    best_pos = pos
        routes[best_route].insert(best_pos, c)
        route_dists[best_route] = route_dist(routes[best_route])
        report_best_vrp(routes)

    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)

    # Intra-route 2-opt on each route
    for r_idx in range(truck_count):
        improved = True
        while improved:
            improved = False
            route = routes[r_idx]
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    old = distance_matrix[route[i - 1], route[i]] + distance_matrix[route[j], route[j + 1]]
                    new = distance_matrix[route[i - 1], route[j]] + distance_matrix[route[i], route[j + 1]]
                    if new < old - 1e-12:
                        route[i:j + 1] = reversed(route[i:j + 1])
                        improved = True
                        route_dists[r_idx] = route_dist(route)
                        break
                if improved:
                    break
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    # Iterated local search with simple perturbation
    outer_iters = min(100, n * 2)  # bounded by instance size
    for _ in range(outer_iters):
        # start from best solution
        routes = [r[:] for r in best_routes]
        route_dists = [route_dist(r) for r in routes]
        # find longest route
        max_dist = max(route_dists)
        longest_idx = route_dists.index(max_dist)
        longest_route = routes[longest_idx]
        if len(longest_route) > 2:
            # pick random customer from longest route (excluding depots)
            i = random.randint(1, len(longest_route) - 2)
            c = longest_route.pop(i)
            route_dists[longest_idx] = route_dist(longest_route)
            # choose random other route
            other_indices = [r for r in range(truck_count) if r != longest_idx]
            other_idx = random.choice(other_indices)
            other_route = routes[other_idx]
            pos = random.randint(1, len(other_route) - 1)
            other_route.insert(pos, c)
            route_dists[other_idx] = route_dist(other_route)
        # re-apply intra-route 2-opt
        for r_idx in range(truck_count):
            improved = True
            while improved:
                improved = False
                route = routes[r_idx]
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        old = distance_matrix[route[i - 1], route[i]] + distance_matrix[route[j], route[j + 1]]
                        new = distance_matrix[route[i - 1], route[j]] + distance_matrix[route[i], route[j + 1]]
                        if new < old - 1e-12:
                            route[i:j + 1] = reversed(route[i:j + 1])
                            improved = True
                            route_dists[r_idx] = route_dist(route)
                            break
                    if improved:
                        break
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    return best_routes