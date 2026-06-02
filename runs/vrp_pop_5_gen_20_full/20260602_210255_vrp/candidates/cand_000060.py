import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_dist = [0.0 for _ in range(truck_count)]

    def route_dist_of(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    unvisited = list(range(1, n))
    for cust in unvisited:
        best_inc = float('inf')
        best_route = None
        best_pos = None
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                if inc < best_inc - 1e-12:
                    best_inc = inc
                    best_route = r_idx
                    best_pos = pos
        route = routes[best_route]
        route.insert(best_pos, cust)
        route_dist[best_route] = route_dist_of(route)

    best_routes = [list(r) for r in routes]
    best_max = max(route_dist)
    report_best_vrp(best_routes)

    max_iter = n * truck_count
    for _ in range(max_iter):
        max_dist = max(route_dist)
        long_indices = [i for i, d in enumerate(route_dist) if abs(d - max_dist) < 1e-12]
        if not long_indices:
            break
        improved = False
        for long_idx in long_indices:
            route = routes[long_idx]
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                nxt = route[pos+1]
                new_long_route = route[:pos] + route[pos+1:]
                new_long_dist = route_dist_of(new_long_route)
                for short_idx, short_route in enumerate(routes):
                    if short_idx == long_idx:
                        continue
                    for p in range(1, len(short_route)):
                        sprev = short_route[p-1]
                        snxt = short_route[p]
                        inc = distance_matrix[sprev][cust] + distance_matrix[cust][snxt] - distance_matrix[sprev][snxt]
                        new_short_route = short_route[:p] + [cust] + short_route[p:]
                        new_short_dist = route_dist_of(new_short_route)
                        new_max = max(new_long_dist, new_short_dist, max(d for i,d in enumerate(route_dist) if i not in (long_idx, short_idx)))
                        if new_max < best_max - 1e-12:
                            routes[long_idx] = new_long_route
                            routes[short_idx] = new_short_route
                            route_dist[long_idx] = new_long_dist
                            route_dist[short_idx] = new_short_dist
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_routes