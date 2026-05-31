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

    # Greedy construction: iteratively assign customer to the route that minimizes max distance
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_max = float('inf')
        best_total = float('inf')
        best_truck = None
        best_pos = None
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

    # Local search: 2-opt intra-route
    def improve_intra(routes):
        improved = True
        while improved:
            improved = False
            for t, route in enumerate(routes):
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        # reverse segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_routes = routes[:t] + [new_route] + routes[t+1:]
                        new_max = max(route_distance(r) for r in new_routes)
                        new_total = sum(route_distance(r) for r in new_routes)
                        old_max = max(route_distance(r) for r in routes)
                        old_total = sum(route_distance(r) for r in routes)
                        if new_max < old_max or (new_max == old_max and new_total < old_total):
                            routes = new_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Relocation inter-route: move a customer from the longest route to another
    def improve_inter(routes):
        improved = True
        while improved:
            improved = False
            # find longest route
            max_dist = -1
            long_idx = -1
            for t, route in enumerate(routes):
                d = route_distance(route)
                if d > max_dist:
                    max_dist = d
                    long_idx = t
            long_route = routes[long_idx]
            if len(long_route) <= 2:
                break
            # try moving each customer from longest route
            best_max = max_dist
            best_total = sum(route_distance(r) for r in routes)
            best_move = None
            for idx in range(1, len(long_route)-1):
                cust = long_route[idx]
                temp_route = long_route[:idx] + long_route[idx+1:]
                for t2, route2 in enumerate(routes):
                    if t2 == long_idx:
                        continue
                    for pos2 in range(1, len(route2)):
                        new_route2 = route2[:pos2] + [cust] + route2[pos2:]
                        new_routes = routes.copy()
                        if t2 < long_idx:
                            new_routes[t2] = new_route2
                            new_routes[long_idx] = temp_route
                        else:
                            new_routes[long_idx] = temp_route
                            new_routes[t2] = new_route2
                        new_max = max(route_distance(r) for r in new_routes)
                        new_total = sum(route_distance(r) for r in new_routes)
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            best_max = new_max
                            best_total = new_total
                            best_move = (long_idx, idx, t2, pos2)
            if best_move is not None:
                long_idx, idx, t2, pos2 = best_move
                cust = routes[long_idx][idx]
                temp_route = routes[long_idx][:idx] + routes[long_idx][idx+1:]
                new_route2 = routes[t2][:pos2] + [cust] + routes[t2][pos2:]
                if t2 < long_idx:
                    routes[t2] = new_route2
                    routes[long_idx] = temp_route
                else:
                    routes[long_idx] = temp_route
                    routes[t2] = new_route2
                improved = True
                # update best if improvement
                new_max = max(route_distance(r) for r in routes)
                new_total = sum(route_distance(r) for r in routes)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
        return routes

    # Apply local search
    current_routes = improve_intra(current_routes)
    current_routes = improve_inter(current_routes)

    # Update best
    current_max = max(route_distance(r) for r in current_routes)
    current_total = sum(route_distance(r) for r in current_routes)
    if current_max < best_max or (current_max == best_max and current_total < best_total):
        best_max = current_max
        best_total = current_total
        best_routes = [list(r) for r in current_routes]
        report_best_vrp(best_routes)

    # Intensification: repeated removal and reinsertion focusing on worst routes
    max_iter = 5 * n
    for it in range(max_iter):
        # Identify worst routes (those with distance > average)
        avg_max = best_max / truck_count
        worst_routes_idx = [t for t, r in enumerate(current_routes) if route_distance(r) > avg_max]
        if not worst_routes_idx:
            # If all good, pick the one with largest distance
            worst_routes_idx = [max(range(truck_count), key=lambda t: route_distance(current_routes[t]))]
        # Remove some customers from worst routes
        removal_targets = []
        for t in worst_routes_idx:
            route = current_routes[t]
            if len(route) > 2:
                # Remove a fraction of customers from this route
                num_rem = max(1, int(0.2 * (len(route)-2)))
                # Remove random customers
                candidates = route[1:-1]
                random.shuffle(candidates)
                removal_targets.extend(candidates[:num_rem])
        if not removal_targets:
            break
        # Remove them from routes
        partial = []
        for route in current_routes:
            partial.append([0] + [c for c in route[1:-1] if c not in removal_targets] + [0])
        unassigned = list(removal_targets)
        # Reinsert greedily
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
        # Apply local search to new_routes
        new_routes = improve_intra(new_routes)
        new_routes = improve_inter(new_routes)
        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        # Accept if improvement
        if new_max < current_max or (new_max == current_max and new_total < current_total):
            current_routes = [list(r) for r in new_routes]
            current_max = new_max
            current_total = new_total
            if new_max < best_max or (new_max == best_max and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
        # else: maybe accept with small probability? We'll only accept improving to stay exploitation-focused.
        # No acceptance of worse solutions.

    return best_routes