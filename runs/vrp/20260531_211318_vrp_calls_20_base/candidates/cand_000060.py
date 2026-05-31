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

    def compute_routes_max_and_total(routes):
        dmax = 0.0
        dsum = 0.0
        for r in routes:
            d = route_distance(r)
            dsum += d
            if d > dmax:
                dmax = d
        return dmax, dsum

    def apply_relocate_move(routes, cust, from_truck, to_truck, pos):
        new_routes = [list(r) for r in routes]
        # Remove cust from from_truck
        new_routes[from_truck] = [0] + [c for c in new_routes[from_truck][1:-1] if c != cust] + [0]
        # Insert cust at pos in to_truck
        route_to = new_routes[to_truck]
        new_routes[to_truck] = route_to[:pos] + [cust] + route_to[pos:]
        return new_routes

    def find_best_relocate(routes, current_max, current_total):
        best_routes = None
        best_max = current_max
        best_total = current_total
        for t1, route1 in enumerate(routes):
            if len(route1) <= 2:
                continue
            # Consider moving each customer in route1
            for i in range(1, len(route1)-1):
                cust = route1[i]
                # Remove cust from route1 temporarily (compute new distance for route1)
                new_route1 = route1[:i] + route1[i+1:]
                # Try inserting into every other route and position
                for t2 in range(len(routes)):
                    if t2 == t1:
                        continue
                    route2 = routes[t2]
                    for j in range(1, len(route2)+1):
                        new_route2 = route2[:j] + [cust] + route2[j:]
                        new_routes = list(routes)
                        new_routes[t1] = new_route1
                        new_routes[t2] = new_route2
                        new_max, new_total = compute_routes_max_and_total(new_routes)
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            best_max = new_max
                            best_total = new_total
                            best_routes = [list(r) for r in new_routes]
        return best_routes, best_max, best_total

    def find_best_swap(routes, current_max, current_total):
        best_routes = None
        best_max = current_max
        best_total = current_total
        for t1 in range(len(routes)):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1)-1):
                cust1 = route1[i]
                for t2 in range(t1+1, len(routes)):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for j in range(1, len(route2)-1):
                        cust2 = route2[j]
                        # Swap cust1 and cust2
                        new_route1 = route1[:i] + [cust2] + route1[i+1:]
                        new_route2 = route2[:j] + [cust1] + route2[j+1:]
                        new_routes = list(routes)
                        new_routes[t1] = new_route1
                        new_routes[t2] = new_route2
                        new_max, new_total = compute_routes_max_and_total(new_routes)
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            best_max = new_max
                            best_total = new_total
                            best_routes = [list(r) for r in new_routes]
        return best_routes, best_max, best_total

    def vnd(routes):
        current_max, current_total = compute_routes_max_and_total(routes)
        improved = True
        while improved:
            improved = False
            # Neighborhood 1: relocate
            best_routes, best_max, best_total = find_best_relocate(routes, current_max, current_total)
            if best_routes is not None:
                routes = best_routes
                current_max, current_total = best_max, best_total
                improved = True
                continue
            # Neighborhood 2: swap
            best_routes, best_max, best_total = find_best_swap(routes, current_max, current_total)
            if best_routes is not None:
                routes = best_routes
                current_max, current_total = best_max, best_total
                improved = True
        return routes, current_max, current_total

    # Iterated local search
    max_iter = max(20, 5 * n)
    no_improve_iter = 0
    restart_threshold = int(0.3 * max_iter)
    for it in range(max_iter):
        # Apply VND to current solution
        current_routes, current_max, current_total = vnd(current_routes)
        if current_max < best_max or (abs(current_max - best_max) < 1e-9 and current_total < best_total):
            best_max = current_max
            best_total = current_total
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
            no_improve_iter = 0
        else:
            no_improve_iter += 1

        # Shake: remove~20% of customers and reinsert greedily
        if no_improve_iter >= restart_threshold or it == max_iter-1:
            # Collect all customers
            all_cust = []
            for r in current_routes:
                for c in r[1:-1]:
                    all_cust.append(c)
            random.shuffle(all_cust)
            num_remove = max(1, int(0.2 * (n-1)))
            to_remove = set(all_cust[:num_remove])
            new_routes = []
            for r in current_routes:
                new_route = [0] + [c for c in r[1:-1] if c not in to_remove] + [0]
                if len(new_route) == 1:
                    new_route = [0, 0]
                new_routes.append(new_route)
            # Reinsert removed customers greedily (minimize max)
            removed = list(to_remove)
            random.shuffle(removed)
            for cust in removed:
                best_truck = None
                best_pos = None
                best_max = float('inf')
                best_total = float('inf')
                for t, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        test_routes = new_routes[:t] + [new_route] + new_routes[t+1:]
                        new_max, new_total = compute_routes_max_and_total(test_routes)
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            best_max = new_max
                            best_total = new_total
                            best_truck = t
                            best_pos = pos
                new_routes[best_truck].insert(best_pos, cust)
            current_routes = new_routes
            no_improve_iter = 0

    return best_routes