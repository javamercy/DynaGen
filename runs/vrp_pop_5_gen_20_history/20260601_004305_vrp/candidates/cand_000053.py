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
                route_dist_val = distance_matrix[0, tour[l]] + distance_matrix[tour[l], 0]
            else:
                route_dist_val = acc + distance_matrix[tour[r - 1], 0]
            seg_dist[l][r] = route_dist_val

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
        maxd = 0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    best_max = compute_max()
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    # --- Global best-improvement ---
    max_passes = n * n
    for _ in range(max_passes):
        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        best_impr = 0.0
        best_key = None
        best_move = None

        # 2-opt moves
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    new_max = new_dist
                    for k, r in enumerate(routes):
                        if k != idx:
                            new_max = max(new_max, route_dist(r))
                    improvement = current_max - new_max
                    if improvement > 1e-12:
                        key = (improvement, -0, -idx, -i, -j)
                        if best_key is None or key > best_key:
                            best_impr = improvement
                            best_key = key
                            best_move = ('2opt', idx, new_route)

        # Relocate moves
        for src_idx, src_route in enumerate(routes):
            for cust_pos in range(1, len(src_route) - 1):
                cust = src_route[cust_pos]
                new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
                dist_src = route_dist(new_src)
                for dst_idx, dst_route in enumerate(routes):
                    if dst_idx == src_idx:
                        continue
                    for pos in range(1, len(dst_route)):
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        dist_dst = route_dist(new_dst)
                        new_max = dist_src
                        if dist_dst > new_max:
                            new_max = dist_dst
                        for k, r in enumerate(routes):
                            if k != src_idx and k != dst_idx:
                                d = route_dist(r)
                                if d > new_max:
                                    new_max = d
                        improvement = current_max - new_max
                        if improvement > 1e-12:
                            key = (improvement, -1, -src_idx, -cust_pos, -dst_idx, -pos)
                            if best_key is None or key > best_key:
                                best_impr = improvement
                                best_key = key
                                best_move = ('relocate', src_idx, cust_pos, dst_idx, pos)

        # Swap moves
        for src_idx, src_route in enumerate(routes):
            for pos_i in range(1, len(src_route) - 1):
                cust_i = src_route[pos_i]
                for dst_idx, dst_route in enumerate(routes):
                    if dst_idx <= src_idx:
                        continue
                    for pos_j in range(1, len(dst_route) - 1):
                        cust_j = dst_route[pos_j]
                        new_src = src_route[:pos_i] + [cust_j] + src_route[pos_i+1:]
                        new_dst = dst_route[:pos_j] + [cust_i] + dst_route[pos_j+1:]
                        new_dist_src = route_dist(new_src)
                        new_dist_dst = route_dist(new_dst)
                        new_max = new_dist_src if new_dist_src > new_dist_dst else new_dist_dst
                        for k, r in enumerate(routes):
                            if k != src_idx and k != dst_idx:
                                d = route_dist(r)
                                if d > new_max:
                                    new_max = d
                        improvement = current_max - new_max
                        if improvement > 1e-12:
                            key = (improvement, -2, -src_idx, -pos_i, -dst_idx, -pos_j)
                            if best_key is None or key > best_key:
                                best_impr = improvement
                                best_key = key
                                best_move = ('swap', src_idx, pos_i, dst_idx, pos_j)

        if best_move is None:
            break

        move_type = best_move[0]
        if move_type == '2opt':
            _, idx, new_route = best_move
            routes[idx] = new_route
        elif move_type == 'relocate':
            _, src_idx, cust_pos, dst_idx, pos = best_move
            cust = routes[src_idx].pop(cust_pos)
            routes[dst_idx].insert(pos, cust)
        else:  # swap
            _, src_idx, pos_i, dst_idx, pos_j = best_move
            routes[src_idx][pos_i], routes[dst_idx][pos_j] = routes[dst_idx][pos_j], routes[src_idx][pos_i]

        new_max = compute_max()
        if new_max < best_max:
            best_max = new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    return best_routes