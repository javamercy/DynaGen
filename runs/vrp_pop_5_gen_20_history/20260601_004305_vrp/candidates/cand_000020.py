import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Nearest neighbor TSP tour from depot
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        best = -1
        best_dist = float('inf')
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current][v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    m = n - 1
    k = truck_count
    # Precompute segment distances for the tour
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0][tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r - 2]][tour[r - 1]]
            if r == l + 1:
                route_dist = distance_matrix[0][tour[l]] + distance_matrix[tour[l]][0]
            else:
                route_dist = acc + distance_matrix[tour[r - 1]][0]
            seg_dist[l][r] = route_dist

    # DP: minimize maximum route distance
    INF = float('inf')
    dp = [[INF] * (k + 1) for _ in range(m + 1)]
    choice = [[-1] * (k + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        for t in range(1, min(i, k) + 1):
            best = INF
            best_j = -1
            for j in range(t - 1, i):
                if dp[j][t - 1] < INF:
                    cand = max(dp[j][t - 1], seg_dist[j][i])
                    if cand < best or (cand == best and j < best_j):
                        best = cand
                        best_j = j
            dp[i][t] = best
            choice[i][t] = best_j

    # Reconstruct routes
    routes = []
    i = m
    t = k
    while t > 0:
        j = choice[i][t]
        seg = tour[j:i]
        routes.append([0] + seg + [0])
        i = j
        t -= 1
    routes.reverse()
    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        total = 0
        for a in range(len(route) - 1):
            total += distance_matrix[route[a]][route[a + 1]]
        return total

    def compute_max():
        return max(route_dist(r) for r in routes)

    current_max = compute_max()
    # Call report_best_vrp initially
    try:
        report_best_vrp(routes)
    except NameError:
        pass

    # Local search – bounded iterations
    max_iterations = 100 * n
    iteration = 0
    improved = True
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Relocate
        for c in range(1, n):
            ri = None
            pos_c = None
            for r_idx, route in enumerate(routes):
                if c in route:
                    ri = r_idx
                    pos_c = route.index(c)
                    break
            if ri is None:
                continue
            old_routes = [r[:] for r in routes]
            # remove c
            routes[ri].pop(pos_c)
            for s_idx in range(truck_count):
                if s_idx == ri:
                    continue
                route_s = routes[s_idx]
                for pos in range(1, len(route_s)):
                    routes[s_idx].insert(pos, c)
                    new_max = compute_max()
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        try:
                            report_best_vrp(routes)
                        except NameError:
                            pass
                        break
                    else:
                        routes[s_idx].pop(pos)
                if improved:
                    break
            if not improved:
                routes = [r[:] for r in old_routes]
            else:
                break
        if improved:
            continue

        # 2-opt within routes
        for ri in range(truck_count):
            route = routes[ri]
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
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
                        try:
                            report_best_vrp(routes)
                        except NameError:
                            pass
                        break
                if improved:
                    break
            if improved:
                break

    return routes