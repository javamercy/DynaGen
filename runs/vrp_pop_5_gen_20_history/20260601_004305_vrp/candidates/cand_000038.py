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

    # --- Two-phase construction: TSP tour + DP split ---
    # Nearest neighbor TSP starting from depot, deterministic smallest-index tie-break
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current, v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    # Precompute segment distances
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0, tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r - 2], tour[r - 1]]
            if r == l + 1:
                route_dist = distance_matrix[0, tour[l]] + distance_matrix[tour[l], 0]
            else:
                route_dist = acc + distance_matrix[tour[r - 1], 0]
            seg_dist[l][r] = route_dist

    # DP: dp[i][k] = min max distance for first i customers with k routes
    dp = [[math.inf] * (truck_count + 1) for _ in range(m + 1)]
    choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        for t in range(1, min(i, truck_count) + 1):
            best_val = math.inf
            best_j = -1
            for j in range(t - 1, i):
                if dp[j][t - 1] < math.inf:
                    cand = max(dp[j][t - 1], seg_dist[j][i])
                    if cand < best_val or (cand == best_val and j < best_j):
                        best_val = cand
                        best_j = j
            dp[i][t] = best_val
            choice[i][t] = best_j

    # Reconstruct routes
    routes = []
    i = m
    t = truck_count
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
            total += distance_matrix[route[a], route[a + 1]]
        return total

    def compute_max():
        maxd = 0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    best_max = compute_max()
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    # --- Improvement: best 2-opt then best swap ---
    max_passes = n * n
    for _ in range(max_passes):
        improved = False
        # Sort routes by distance descending, index ascending
        dists = [route_dist(r) for r in routes]
        order = sorted(range(len(routes)), key=lambda i: (-dists[i], i))

        # Best 2-opt on each route in order
        for idx in order:
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_local = route_dist(route)
            best_route = route[:]
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_local:
                        best_local = new_dist
                        best_route = new_route
            if best_local < route_dist(route):
                routes[idx] = best_route
                new_max = compute_max()
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                improved = True

        if improved:
            continue

        # If no 2-opt improvement, try best inter-route swap (exchange of customers)
        best_improvement = 0.0
        best_move = None
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                ri = routes[i]
                rj = routes[j]
                if len(ri) <= 2 or len(rj) <= 2:
                    continue
                for pos_i in range(1, len(ri)-1):
                    cust_i = ri[pos_i]
                    for pos_j in range(1, len(rj)-1):
                        cust_j = rj[pos_j]
                        new_ri = ri[:pos_i] + [cust_j] + ri[pos_i+1:]
                        new_rj = rj[:pos_j] + [cust_i] + rj[pos_j+1:]
                        dist_i = route_dist(new_ri)
                        dist_j = route_dist(new_rj)
                        other_dists = [route_dist(r) for k, r in enumerate(routes) if k not in (i, j)]
                        new_max = max([dist_i, dist_j] + other_dists)
                        improvement = compute_max() - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (i, pos_i, j, pos_j, cust_i, cust_j)
                        elif improvement == best_improvement and best_move is not None:
                            # tie-break: smallest cust_i, then pos_i, then j, then pos_j
                            oi, opos_i, oj, opos_j, ocust_i, ocust_j = best_move
                            if (cust_i < ocust_i or 
                                (cust_i == ocust_i and pos_i < opos_i) or
                                (cust_i == ocust_i and pos_i == opos_i and j < oj) or
                                (cust_i == ocust_i and pos_i == opos_i and j == oj and pos_j < opos_j)):
                                best_move = (i, pos_i, j, pos_j, cust_i, cust_j)

        if best_move and best_improvement > 0:
            i, pos_i, j, pos_j, cust_i, cust_j = best_move
            routes[i] = routes[i][:pos_i] + [cust_j] + routes[i][pos_i+1:]
            routes[j] = routes[j][:pos_j] + [cust_i] + routes[j][pos_j+1:]
            new_max = compute_max()
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            improved = True

        if not improved:
            break

    return best_routes