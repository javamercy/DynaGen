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

    # Two-phase construction: TSP tour + DP split
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
        return max(route_dist(r) for r in routes)

    best_max = compute_max()
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    current_max = best_max
    threshold = 0.1 * current_max
    cooling = 0.95
    max_iter = n * n
    stagnation = 0
    for _ in range(max_iter):
        improved = False
        # 2-opt on each route (first-improvement)
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_local = route_dist(route)
            best_route = route[:]
            found = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    # compute new max assuming this move
                    old_max = current_max
                    new_max = old_max
                    if new_dist != best_local:
                        # only need to consider routes with new distance and other max
                        other_dists = [route_dist(r) for r in routes]
                        other_dists[idx] = new_dist
                        new_max = max(other_dists)
                    else:
                        new_max = old_max
                    if new_max < old_max - threshold:
                        best_local = new_dist
                        best_route = new_route
                        found = True
                        break
                if found:
                    break
            if found:
                routes[idx] = best_route
                current_max = max(route_dist(r) for r in routes)
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                break
        if improved:
            stagnation = 0
            threshold *= cooling
            continue

        # Relocate from longest route
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        src_route = routes[longest_idx]
        found = False
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
                    if new_max < current_max - threshold:
                        routes[longest_idx] = new_src
                        routes[dst_idx] = new_dst
                        current_max = new_max
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if improved:
            stagnation = 0
            threshold *= cooling
            continue

        # Swap from longest route
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        src_route = routes[longest_idx]
        found = False
        for pos_i in range(1, len(src_route) - 1):
            cust_i = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos_j in range(1, len(dst_route) - 1):
                    cust_j = dst_route[pos_j]
                    new_src = src_route[:pos_i] + [cust_j] + src_route[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust_i] + dst_route[pos_j+1:]
                    new_dist_src = route_dist(new_src)
                    new_dist_dst = route_dist(new_dst)
                    other_dists = [route_dist(r) for i, r in enumerate(routes) if i not in (longest_idx, dst_idx)]
                    new_max = max([new_dist_src, new_dist_dst] + other_dists)
                    if new_max < current_max - threshold:
                        routes[longest_idx] = new_src
                        routes[dst_idx] = new_dst
                        current_max = new_max
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not improved:
            stagnation += 1
            if stagnation > 3:
                break
        else:
            stagnation = 0
            threshold *= cooling

    return best_routes