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
        return sum(dist[route[i], route[i+1]] for i in range(len(route)-1))

    # farthest insertion initialization (from cand_000072)
    customers = sorted(range(1, n), key=lambda c: -dist[0][c])
    routes = [[0,0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists[:t]) + new_dist + sum(route_dists[t+1:])
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)
        route_dists[best_truck] = route_distance(routes[best_truck])

    # 2-opt local search (single pass, repeat until no improvement)
    def local_search(routes, dists):
        improved = True
        while improved:
            improved = False
            for t, route in enumerate(routes):
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < dists[t] - 1e-9:
                            new_max = max(dists[:t] + [new_dist] + dists[t+1:])
                            old_max = max(dists)
                            new_total = sum(dists) - dists[t] + new_dist
                            old_total = sum(dists)
                            if new_max < old_max - 1e-9 or (abs(new_max - old_max) < 1e-9 and new_total < old_total):
                                routes[t] = new_route
                                dists[t] = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes, dists

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_routes, current_dists = local_search(current_routes, current_dists)
    best_routes = [list(r) for r in current_routes]
    best_dists = list(current_dists)
    best_max = max(best_dists)
    best_total = sum(best_dists)
    report_best_vrp(best_routes)

    # ILS parameters
    max_iter = min(100, 10 * n)
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n-1)))

    for it in range(max_iter):
        # perturbation: random removal and greedy reinsertion
        all_customers = [c for r in current_routes for c in r[1:-1]]
        random.shuffle(all_customers)
        removed = set(all_customers[:num_removals])
        partial_routes = []
        partial_dists = []
        for t, route in enumerate(current_routes):
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            partial_routes.append(new_route)
            partial_dists.append(route_distance(new_route))
        unassigned = list(removed)
        # greedy repair (minimizing max then total distance)
        for cust in unassigned:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            for t, route in enumerate(partial_routes):
                old_dist = partial_dists[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route)
                    new_max = max(max(partial_dists[:t]) if t > 0 else -float('inf'), new_dist, max(partial_dists[t+1:]) if t+1 < len(partial_dists) else -float('inf'))
                    new_total = sum(partial_dists) - old_dist + new_dist
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_truck = t
                        best_pos = pos
            partial_routes[best_truck].insert(best_pos, cust)
            partial_dists[best_truck] = route_distance(partial_routes[best_truck])
        # local search on perturbed solution
        new_routes, new_dists = local_search(partial_routes, partial_dists)
        new_max = max(new_dists)
        new_total = sum(new_dists)
        # update best if improved
        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
            best_routes = [list(r) for r in new_routes]
            best_dists = list(new_dists)
            best_max = new_max
            best_total = new_total
            report_best_vrp(best_routes)
        # move to new solution (even if worse) for diversification
        current_routes = [list(r) for r in new_routes]
        current_dists = list(new_dists)

    return best_routes