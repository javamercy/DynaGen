import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_dist = [0.0 for _ in range(truck_count)]

    # Helper to compute route length
    def route_length(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Insert customers in order 1..n-1
    for cust in range(1, n):
        best_max = float('inf')
        best_route = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            # Insert between positions 1 and len(route)-1
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                added = distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                removed = distance_matrix[prev, nxt]
                new_dist = route_dist[r] + added - removed
                # Compute new max
                other_dists = route_dist[:r] + route_dist[r+1:]
                new_max = max(new_dist, max(other_dists, default=0.0))
                if new_max < best_max or (new_max == best_max and (r < best_route or (r == best_route and pos < best_pos))):
                    best_max = new_max
                    best_route = r
                    best_pos = pos
        # Perform insertion
        route = routes[best_route]
        prev = route[best_pos-1]
        nxt = route[best_pos]
        route_dist[best_route] += distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
        route.insert(best_pos, cust)

    # Report initial solution
    report_best_vrp(routes)

    # Local search: bounded number of iterations
    max_iter = n * 2
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        current_max = max(route_dist)

        # Inter-route relocate
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1)-1):
                cust = route1[i]
                # Remove cust temporarily
                prev_rem = route1[i-1]
                next_rem = route1[i+1]
                removed_cost = distance_matrix[prev_rem, cust] + distance_matrix[cust, next_rem] - distance_matrix[prev_rem, next_rem]
                new_dist_r1 = route_dist[r1] - removed_cost
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    for pos in range(1, len(route2)):
                        prev_ins = route2[pos-1]
                        next_ins = route2[pos]
                        added_cost = distance_matrix[prev_ins, cust] + distance_matrix[cust, next_ins] - distance_matrix[prev_ins, next_ins]
                        new_dist_r2 = route_dist[r2] + added_cost
                        other_dists = [d for idx, d in enumerate(route_dist) if idx not in (r1, r2)]
                        new_max = max(new_dist_r1, new_dist_r2, max(other_dists, default=0.0))
                        if new_max < current_max - 1e-12:
                            # Perform move
                            route1.pop(i)
                            route_dist[r1] = new_dist_r1
                            route2.insert(pos, cust)
                            route_dist[r2] = new_dist_r2
                            improved = True
                            current_max = new_max
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            iteration += 1
            continue

        # Intra-route 2-opt
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_length(new_route)
                    if new_len < route_dist[r] - 1e-12:
                        other_dists = route_dist[:r] + route_dist[r+1:]
                        new_max = max(new_len, max(other_dists, default=0.0))
                        if new_max < current_max - 1e-12:
                            route[:] = new_route
                            route_dist[r] = new_len
                            improved = True
                            current_max = new_max
                            report_best_vrp(routes)
                            break
                if improved:
                    break
            if improved:
                break
        iteration += 1

    return routes