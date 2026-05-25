import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    route_dists = [0.0 for _ in routes]
    current_max = 0.0
    best_routes = [r[:] for r in routes]
    best_max = 0.0

    report_best_vrp(routes)

    unassigned = set(range(1, n))

    while unassigned:
        best_regret = -1.0
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_new_max = float('inf')

        for cust in sorted(unassigned):
            first_max = float('inf')
            second_max = float('inf')
            first_route = -1
            first_pos = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_dist(new_route)
                    other_max = 0.0
                    for other_idx, od in enumerate(route_dists):
                        if other_idx == r_idx:
                            other_max = max(other_max, new_dist)
                        else:
                            other_max = max(other_max, od)
                    new_max = other_max
                    if new_max < first_max:
                        second_max = first_max
                        first_max = new_max
                        first_route = r_idx
                        first_pos = pos
                    elif new_max < second_max:
                        second_max = new_max
            regret = second_max - first_max
            if regret > best_regret or (regret == best_regret and (best_cust is None or cust < best_cust)):
                best_regret = regret
                best_cust = cust
                best_new_max = first_max
                best_route_idx = first_route
                best_pos = first_pos

        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        route_dists[best_route_idx] = route_dist(route)
        current_max = max(route_dists)
        unassigned.remove(best_cust)
        if current_max < best_max or best_max == 0:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    max_iter = n * n
    for _ in range(max_iter):
        improved = False
        long_idx = max(range(len(route_dists)), key=lambda i: route_dists[i])
        long_dist = route_dists[long_idx]
        if long_dist <= best_max * 0.999:
            break

        long_route = routes[long_idx]
        for idx in range(1, len(long_route)-1):
            cust = long_route[idx]
            new_long = long_route[:idx] + long_route[idx+1:]
            new_long_dist = route_dist(new_long)
            best_new_max = float('inf')
            best_new_routes = None
            for r_idx, route in enumerate(routes):
                if r_idx == long_idx:
                    for pos in range(1, len(new_long)):
                        new_route = new_long[:pos] + [cust] + new_long[pos:]
                        new_dist = route_dist(new_route)
                        new_routes = routes[:]
                        new_routes[long_idx] = new_route
                        new_dists = [route_dist(r) for r in new_routes]
                        new_max = max(new_dists)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_new_routes = new_routes
                else:
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_dist(new_route)
                        new_routes = routes[:]
                        new_routes[r_idx] = new_route
                        new_routes[long_idx] = new_long
                        new_dists = [route_dist(r) for r in new_routes]
                        new_max = max(new_dists)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_new_routes = new_routes
            if best_new_max < best_max:
                routes = best_new_routes
                route_dists = [route_dist(r) for r in routes]
                current_max = max(route_dists)
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(routes)
                improved = True
                break

        if improved:
            continue

        long_route = routes[long_idx]
        n_cust = len(long_route) - 2
        best_imp = 0.0
        best_i = best_j = -1
        for i in range(1, n_cust+1):
            for j in range(i+1, n_cust+1):
                new_route = long_route[:i] + long_route[i:j+1][::-1] + long_route[j+1:]
                new_dist = route_dist(new_route)
                if new_dist < route_dists[long_idx] - 1e-9:
                    reduction = route_dists[long_idx] - new_dist
                    if reduction > best_imp:
                        best_imp = reduction
                        best_i, best_j = i, j
        if best_imp > 0:
            new_route = long_route[:best_i] + long_route[best_i:best_j+1][::-1] + long_route[best_j+1:]
            routes[long_idx] = new_route
            route_dists[long_idx] = route_dist(new_route)
            current_max = max(route_dists)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(routes)
            improved = True

        if not improved:
            break

    return best_routes