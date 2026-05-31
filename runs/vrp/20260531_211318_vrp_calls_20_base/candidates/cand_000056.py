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

    # Initial construction: greedy insertion minimizing max distance
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_routes = routes[:t] + [new_route] + routes[t+1:]
                new_max = max(route_distance(r) for r in new_routes)
                new_total = sum(route_distance(r) for r in new_routes)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)

    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    current_routes = [list(r) for r in routes]
    max_iter = max(10, 2 * n)  # bounded iterations
    removal_count = max(1, int(0.1 * (n-1)))

    for _ in range(max_iter):
        # Random removal
        all_customers = [c for r in current_routes for c in r[1:-1]]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:removal_count])
        partial_routes = []
        for route in current_routes:
            partial_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        # Greedy repair
        unassigned = to_remove
        for cust in unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_truck = None
            best_pos = None
            for t, route in enumerate(partial_routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_routes = partial_routes[:t] + [new_route] + partial_routes[t+1:]
                    new_max_val = max(route_distance(r) for r in new_routes)
                    new_total_val = sum(route_distance(r) for r in new_routes)
                    if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                        best_max = new_max_val
                        best_total = new_total_val
                        best_truck = t
                        best_pos = pos
            partial_routes[best_truck].insert(best_pos, cust)
        current_routes = [list(r) for r in partial_routes]
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)
        if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and current_total < sum(route_distance(r) for r in best_routes)):
            best_max = current_max
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)

    return best_routes