import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    def insertion_delta(route, pos, cust):
        prev = route[pos-1]
        nxt = route[pos]
        return dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]

    # Farthest-first initial construction
    customers = sorted(range(1, n), key=lambda c: -dist[0][c])
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                delta = insertion_delta(route, pos, cust)
                new_dist = route_dists[t] + delta
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + delta
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
        route_dists[best_truck] += insertion_delta(route, best_pos, cust)

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # Parameters
    max_iter = min(3000, 20 * n)
    removal_fraction = 0.2
    beta_start = 0.1
    beta_end = 0.01

    for it in range(max_iter):
        num_removals = max(1, int(removal_fraction * (n - 1)))
        beta = beta_start + (beta_end - beta_start) * (it / max_iter)

        # Random removal
        all_customers = [c for r in current_routes for c in r[1:-1]]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        new_routes = []
        new_dists = []
        for route in current_routes:
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        removed = list(to_remove)

        # Greedy insertion
        routes_repair = [list(r) for r in new_routes]
        dists_repair = list(new_dists)
        unassigned = list(removed)
        current_max_repair = max(dists_repair)
        for cust in unassigned:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            best_delta = None
            for t, route in enumerate(routes_repair):
                old_dist = dists_repair[t]
                for pos in range(1, len(route)):
                    delta = insertion_delta(route, pos, cust)
                    new_dist = old_dist + delta
                    new_max_val = max(current_max_repair, new_dist)
                    new_total_val = sum(dists_repair) + delta
                    if new_max_val < best_new_max or (new_max_val == best_new_max and new_total_val < best_new_total):
                        best_new_max = new_max_val
                        best_new_total = new_total_val
                        best_truck = t
                        best_pos = pos
                        best_delta = delta
            route = routes_repair[best_truck]
            routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            dists_repair[best_truck] += best_delta
            if dists_repair[best_truck] > current_max_repair:
                current_max_repair = dists_repair[best_truck]
        new_routes_final = routes_repair
        new_dists_final = dists_repair

        new_max = max(new_dists_final)
        new_total = sum(new_dists_final)
        threshold = best_max * (1.0 + beta)
        if new_max <= threshold:
            current_routes = [list(r) for r in new_routes_final]
            current_dists = list(new_dists_final)
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes_final]
                best_dists = list(new_dists_final)
                report_best_vrp(best_routes)

    return best_routes