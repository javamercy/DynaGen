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

    # Use sequential order of customers
    custs = customers  # [1, 2, ..., n-1]
    k = truck_count
    # Precompute route distances for segments
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0][custs[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[custs[r - 2]][custs[r - 1]]
            if r == l + 1:
                route_dist = distance_matrix[0][custs[l]] + distance_matrix[custs[l]][0]
            else:
                route_dist = acc + distance_matrix[custs[r - 1]][0]
            seg_dist[l][r] = route_dist

    # DP: minimax split
    dp = [[math.inf] * (k + 1) for _ in range(m + 1)]
    choice = [[-1] * (k + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        for t in range(1, min(i, k) + 1):
            best = math.inf
            best_j = -1
            for j in range(t - 1, i):
                if dp[j][t - 1] < math.inf:
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
        seg = custs[j:i]
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
        maxd = 0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    def two_opt_route(route):
        if len(route) <= 3:
            return route, False
        improved = False
        best_route = route[:]
        best_dist = route_dist(best_route)
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_dist(new_route)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
        return best_route, improved

    # Initial best
    current_max = compute_max()
    report_best_vrp(routes)

    max_iter = 100 * n
    iteration = 0
    improved = True
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        # Intra-route 2-opt
        for ri in range(truck_count):
            if len(routes[ri]) <= 3:
                continue
            new_route, imp = two_opt_route(routes[ri])
            if imp:
                routes[ri] = new_route
                new_max = compute_max()
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    report_best_vrp(routes)
        if improved:
            continue
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
    # Ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    return routes