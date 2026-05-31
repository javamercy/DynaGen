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

    # Initial construction: random order greedy insertion minimizing max distance
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

    current_routes = [list(r) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    best_total = sum(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # ALNS parameters
    max_iter = 20 * n
    removal_fraction = 0.2
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max / 2.0
    T = T0

    for it in range(max_iter):
        # Random removal
        all_customers = [c for r in current_routes for c in r[1:-1]]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        partial = []
        for route in current_routes:
            partial.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        unassigned = list(to_remove)

        # Greedy repair: minimize max distance then total
        new_routes = [list(r) for r in partial]
        for cust in unassigned:
            best_max_ins = float('inf')
            best_total_ins = float('inf')
            best_truck = None
            best_pos = None
            for t, route in enumerate(new_routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    candidate_routes = new_routes[:t] + [new_route] + new_routes[t+1:]
                    new_max = max(route_distance(r) for r in candidate_routes)
                    new_total = sum(route_distance(r) for r in candidate_routes)
                    if new_max < best_max_ins or (new_max == best_max_ins and new_total < best_total_ins):
                        best_max_ins = new_max
                        best_total_ins = new_total
                        best_truck = t
                        best_pos = pos
            new_routes[best_truck].insert(best_pos, cust)

        # Evaluate
        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)

        delta = new_max - current_max
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_routes = [list(r) for r in new_routes]
            if new_max < best_max or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)

        # Temperature update (exponential cooling)
        T = T0 * math.exp(-it / (max_iter / 2.0))

    return best_routes