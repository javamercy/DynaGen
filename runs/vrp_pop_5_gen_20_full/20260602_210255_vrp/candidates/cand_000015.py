import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_dist = [0.0] * truck_count
    unvisited = list(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Get initial partial distances
    for i in range(truck_count):
        route_dist[i] = route_distance(routes[i])

    # Construction: iterate customers in sorted order
    for cust in sorted(unvisited):
        best_inc = float('inf')
        best_r = -1
        best_p = -1
        for r_idx in range(truck_count):
            route = routes[r_idx]
            # positions from 1 to len(route)-1 (before last depot)
            for p in range(1, len(route)):
                prev = route[p-1]
                nxt = route[p]
                inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                if inc < best_inc - 1e-12:
                    best_inc = inc
                    best_r = r_idx
                    best_p = p
                elif abs(inc - best_inc) < 1e-12:
                    # tie-breaking: smaller route index, then smaller position
                    if r_idx < best_r or (r_idx == best_r and p < best_p):
                        best_r = r_idx
                        best_p = p
        # insert
        routes[best_r].insert(best_p, cust)
        route_dist[best_r] = route_distance(routes[best_r])

    best_routes = [list(r) for r in routes]
    best_max = max(route_dist)
    report_best_vrp(best_routes)

    # Improvement: relocate from longest route
    max_iter = n * truck_count  # bounded
    for _ in range(max_iter):
        improved = False
        # find longest route
        max_dist = max(route_dist)
        long_indices = [i for i, d in enumerate(route_dist) if abs(d - max_dist) < 1e-12]
        for long_idx in long_indices:
            route = routes[long_idx]
            if len(route) <= 2:
                continue
            # iterate customers in order (excluding depots)
            for cust in route[1:-1]:
                # remove cust from its route
                new_long = [x for x in route if x != cust]
                dist_long_new = route_distance(new_long)
                # try inserting into other routes
                for short_idx in range(truck_count):
                    if short_idx == long_idx:
                        continue
                    short_route = routes[short_idx]
                    for p in range(1, len(short_route)):
                        prev = short_route[p-1]
                        nxt = short_route[p]
                        inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_short = short_route[:p] + [cust] + short_route[p:]
                        dist_short_new = route_distance(new_short)
                        # compute new max distance
                        new_max = max(dist_long_new, dist_short_new)
                        for i, d in enumerate(route_dist):
                            if i not in (long_idx, short_idx):
                                new_max = max(new_max, d)
                        if new_max < best_max - 1e-12:
                            # apply move
                            routes[long_idx] = new_long
                            routes[short_idx] = new_short
                            route_dist[long_idx] = dist_long_new
                            route_dist[short_idx] = dist_short_new
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