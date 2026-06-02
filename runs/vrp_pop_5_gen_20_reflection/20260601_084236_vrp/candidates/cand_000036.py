import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best_route = route[:]
        best_dist = route_distance(route)
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_dist - 1e-12:
                        best_dist = d
                        best_route = new_route
                        improved = True
                if improved:
                    break
            route = best_route
        return route

    def construct(seed):
        random.seed(seed)
        order = customers[:]
        random.shuffle(order)
        # start with empty routes
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in order:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            # for each route, try all insertion positions
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    # compute new max distance
                    new_dist = route_distance(new_route)
                    other_max = max(route_distance(r) for idx, r in enumerate(routes) if idx != r_idx)
                    new_max = max(new_dist, other_max)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
            # insert in best place
            route = routes[best_route_idx]
            routes[best_route_idx] = route[:best_pos] + [cust] + route[best_pos:]
        # apply 2-opt to each route
        for i in range(truck_count):
            routes[i] = two_opt(routes[i])
        return routes

    best_routes = None
    best_max = float('inf')
    num_restarts = min(5, truck_count)  # small number to avoid timeout
    for restart in range(num_restarts):
        routes = construct(restart)
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_routes = [r[:] for r in routes]
            best_max = current_max
            report_best_vrp(best_routes)
        # local search
        improved = True
        max_iters = n * truck_count
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # find routes with max distance (may be multiple)
            max_val = max_distance(routes)
            max_routes_idx = [i for i, r in enumerate(routes) if abs(route_distance(r) - max_val) < 1e-12]
            # first try relocate from a max route to another route
            for r_idx in max_routes_idx:
                route = routes[r_idx]
                if len(route) <= 2:
                    continue
                custs = route[1:-1]
                for cust in custs:
                    # remove cust from its route
                    new_route_a = [0] + [c for c in route[1:-1] if c != cust] + [0]
                    for other_idx in range(truck_count):
                        if other_idx == r_idx:
                            continue
                        other_route = routes[other_idx]
                        for pos in range(1, len(other_route)):
                            new_route_b = other_route[:pos] + [cust] + other_route[pos:]
                            # compute new max
                            new_max_candidate = max(route_distance(new_route_a),
                                                    route_distance(new_route_b),
                                                    max(route_distance(r) for i, r in enumerate(routes) if i not in (r_idx, other_idx)))
                            if new_max_candidate < max_val - 1e-12:
                                # accept
                                routes[r_idx] = two_opt(new_route_a)
                                routes[other_idx] = two_opt(new_route_b)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                current_max = max_distance(routes)
                if current_max < best_max - 1e-12:
                    best_routes = [r[:] for r in routes]
                    best_max = current_max
                    report_best_vrp(best_routes)
                continue
            # if no relocate, try swap between a max route and another route
            for r_idx in max_routes_idx:
                route_a = routes[r_idx]
                if len(route_a) <= 2:
                    continue
                for other_idx in range(r_idx+1, truck_count):
                    route_b = routes[other_idx]
                    if len(route_b) <= 2:
                        continue
                    custs_a = route_a[1:-1]
                    custs_b = route_b[1:-1]
                    for cust_a in custs_a:
                        for cust_b in custs_b:
                            new_route_a = [0] + [c for c in route_a[1:-1] if c != cust_a] + [cust_b] + [0]
                            new_route_b = [0] + [c for c in route_b[1:-1] if c != cust_b] + [cust_a] + [0]
                            new_route_a = two_opt(new_route_a)
                            new_route_b = two_opt(new_route_b)
                            new_max_candidate = max(route_distance(new_route_a),
                                                    route_distance(new_route_b),
                                                    max(route_distance(r) for i, r in enumerate(routes) if i not in (r_idx, other_idx)))
                            if new_max_candidate < max_val - 1e-12:
                                routes[r_idx] = new_route_a
                                routes[other_idx] = new_route_b
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                current_max = max_distance(routes)
                if current_max < best_max - 1e-12:
                    best_routes = [r[:] for r in routes]
                    best_max = current_max
                    report_best_vrp(best_routes)
    return best_routes