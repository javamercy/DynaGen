import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # Initial construction: greedy insertion in customer order (1..n-1) minimizing max distance
    customers = list(range(1, n))
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
    report_best_vrp(best_routes)

    # Deterministic ALNS parameters
    max_iter = 50 * n
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n-1)))

    def worst_removal(routes, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            base = route_distance(route)
            for i in range(1, len(route)-1):
                prev = route[i-1]
                nxt = route[i+1]
                with_ = dist[prev, route[i]] + dist[route[i], nxt]
                without = dist[prev, nxt]
                contrib = with_ - without
                all_contribs.append((-contrib, t, i, route[i]))
        # Sort descending by contribution, tie by customer index
        all_contribs.sort(key=lambda x: (-x[0], x[3]))
        to_remove = set()
        for _, t, i, cust in all_contribs[:num_removals]:
            to_remove.add(cust)
        new_routes = []
        for t, route in enumerate(routes):
            new_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        return list(to_remove), new_routes

    def regret2_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max = float('inf')
                best_total = float('inf')
                second_best_max = float('inf')
                second_best_total = float('inf')
                best_truck = None
                best_pos = None
                for t, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_routes = routes[:t] + [new_route] + routes[t+1:]
                        new_max_val = max(route_distance(r) for r in new_routes)
                        new_total_val = sum(route_distance(r) for r in new_routes)
                        if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                            second_best_max = best_max
                            second_best_total = best_total
                            best_max = new_max_val
                            best_total = new_total_val
                            best_truck = t
                            best_pos = pos
                        elif new_max_val < second_best_max or (new_max_val == second_best_max and new_total_val < second_best_total):
                            second_best_max = new_max_val
                            second_best_total = new_total_val
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max
                # Deterministic tie-breaking: higher regret first, then lower best_max, then lower best_total, then lower customer index
                if best_info is None:
                    best_info = (-regret, best_max, best_total, cust, best_truck, best_pos)
                else:
                    candidate = (-regret, best_max, best_total, cust, best_truck, best_pos)
                    if candidate > best_info:  # Note: we want max regret so we store negative
                        best_info = candidate
            _, best_max_val, best_total_val, cust, best_truck, best_pos = best_info
            routes[best_truck].insert(best_pos, cust)
            unassigned.remove(cust)
        return routes

    for it in range(max_iter):
        removed, partial = worst_removal(current_routes, num_removals)
        new_routes = regret2_repair(partial, removed)

        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)

        if new_max < current_max or (new_max == current_max and new_total < current_total):
            current_routes = [list(r) for r in new_routes]
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                best_max = new_max
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)

    return best_routes