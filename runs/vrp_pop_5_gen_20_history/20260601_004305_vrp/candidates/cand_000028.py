import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        for _ in range(truck_count - m):
            routes.append([0, 0])
        return routes

    def route_dist(route):
        total = 0
        for a in range(len(route) - 1):
            total += distance_matrix[route[a]][route[a + 1]]
        return total

    def compute_max():
        maxd = 0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    # Sequential insertion construction
    # Sort customers by distance to depot descending, tie-break by index ascending
    dep_dist = [distance_matrix[0][i] for i in range(1, n)]
    order = sorted(range(1, n), key=lambda i: (-dep_dist[i-1], i))
    routes = [[0, 0] for _ in range(truck_count)]
    current_route_dists = [0.0] * truck_count

    for c in order:
        best_route_idx = None
        best_new_max = math.inf
        best_new_route = None
        best_new_dist = 0.0
        for ri, route in enumerate(routes):
            # Special case: empty route
            if route == [0, 0]:
                new_route = [0, c, 0]
                new_dist = distance_matrix[0][c] + distance_matrix[c][0]
            else:
                # Find best insertion position in this route
                best_increase = math.inf
                best_pos = -1
                for pos in range(1, len(route)):
                    a = route[pos-1]
                    b = route[pos]
                    increase = distance_matrix[a][c] + distance_matrix[c][b] - distance_matrix[a][b]
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = pos
                new_route = route[:best_pos] + [c] + route[best_pos:]
                new_dist = current_route_dists[ri] + best_increase
            # Compute new max if we insert here
            other_max = 0.0
            for rj, r in enumerate(routes):
                if rj == ri:
                    continue
                d = current_route_dists[rj]
                if d > other_max:
                    other_max = d
            candidate_max = max(other_max, new_dist)
            if candidate_max < best_new_max:
                best_new_max = candidate_max
                best_route_idx = ri
                best_new_route = new_route
                best_new_dist = new_dist
        # Apply best insertion
        routes[best_route_idx] = best_new_route
        current_route_dists[best_route_idx] = best_new_dist

    # Initial reporting
    current_max = compute_max()
    report_best_vrp(routes)

    # Local search and perturbation (same as parent)
    improved = True
    max_iter = 100 * n
    iteration = 0
    perturb_count = 0
    max_perturb = 3
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        # Relocate
        for c in range(1, n):
            r_idx = None
            pos_c = None
            for ri, route in enumerate(routes):
                if c in route:
                    r_idx = ri
                    pos_c = route.index(c)
                    break
            if r_idx is None:
                continue
            old_route_r = routes[r_idx][:]
            routes[r_idx].pop(pos_c)
            for s_idx in range(truck_count):
                if s_idx == r_idx:
                    continue
                route_s = routes[s_idx]
                for pos in range(1, len(route_s)):
                    old_route_s = route_s[:]
                    routes[s_idx].insert(pos, c)
                    new_max = compute_max()
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        report_best_vrp(routes)
                        break
                    else:
                        routes[s_idx].pop(pos)
                if improved:
                    break
            if not improved:
                routes[r_idx] = old_route_r[:]
            else:
                break
        if improved:
            continue
        # Swap
        for i in range(1, n):
            ri = None
            pos_i = None
            for ri_idx, route in enumerate(routes):
                if i in route:
                    ri = ri_idx
                    pos_i = route.index(i)
                    break
            if ri is None:
                continue
            for j in range(i + 1, n):
                rj = None
                pos_j = None
                for rj_idx, route in enumerate(routes):
                    if j in route:
                        rj = rj_idx
                        pos_j = route.index(j)
                        break
                if rj is None or ri == rj:
                    continue
                old_i_route = routes[ri][:]
                old_j_route = routes[rj][:]
                routes[ri].pop(pos_i)
                routes[rj].pop(pos_j)
                routes[ri].insert(pos_i, j)
                routes[rj].insert(pos_j, i)
                new_max = compute_max()
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    report_best_vrp(routes)
                    break
                else:
                    routes[ri] = old_i_route[:]
                    routes[rj] = old_j_route[:]
            if improved:
                break
        if improved:
            continue
        # 2-opt within a route
        for ri in range(truck_count):
            route = routes[ri]
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    old_dist = route_dist(route)
                    new_dist = route_dist(new_route)
                    other_max = 0
                    for rr in range(truck_count):
                        if rr != ri:
                            d = route_dist(routes[rr])
                            if d > other_max:
                                other_max = d
                    new_max = max(other_max, new_dist)
                    if new_max < current_max:
                        routes[ri] = new_route
                        current_max = new_max
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
            if improved:
                break
        # If no improvement, apply deterministic perturbation: reverse longest route
        if not improved and perturb_count < max_perturb:
            max_dist = 0
            longest_idx = 0
            for ri, route in enumerate(routes):
                d = route_dist(route)
                if d > max_dist:
                    max_dist = d
                    longest_idx = ri
            route = routes[longest_idx]
            if len(route) > 3:
                new_route = [route[0]] + route[1:-1][::-1] + [route[-1]]
                routes[longest_idx] = new_route
                current_max = compute_max()
                improved = True  # force another loop
                perturb_count += 1
    return routes