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

    # --- Simplified improvement: sequential best-improvement 2-opt then relocate ---
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

        # If no 2-opt improvement, try best relocate from longest route
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: (dists[i], -i))  # longest, tie: lower index
        best_improvement = 0.0
        best_move = None
        src_route = routes[longest_idx]
        for cust_pos in range(1, len(src_route) - 1):
            cust = src_route[cust_pos]
            new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
            dist_src = route_dist(new_src)
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    dist_dst = route_dist(new_dst)
                    other_dists = [route_dist(r) for i, r in enumerate(routes) if i not in (longest_idx, dst_idx)]
                    new_max = max([dist_src, dist_dst] + other_dists)
                    improvement = dists[longest_idx] - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (cust, cust_pos, dst_idx, pos)
                    elif improvement == best_improvement and best_move is not None:
                        # tie-breaking: smaller customer, then smaller cust_pos, then smaller dst_idx, then smaller pos
                        ocust, ocust_pos, odst_idx, opos = best_move
                        if (cust < ocust or 
                            (cust == ocust and cust_pos < ocust_pos) or
                            (cust == ocust and cust_pos == ocust_pos and dst_idx < odst_idx) or
                            (cust == ocust and cust_pos == ocust_pos and dst_idx == odst_idx and pos < opos)):
                            best_move = (cust, cust_pos, dst_idx, pos)

        if best_move and best_improvement > 0:
            cust, cust_pos, dst_idx, pos = best_move
            routes[longest_idx].pop(cust_pos)
            routes[dst_idx].insert(pos, cust)
            new_max = compute_max()
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            improved = True

        if not improved:
            break

    return best_routes