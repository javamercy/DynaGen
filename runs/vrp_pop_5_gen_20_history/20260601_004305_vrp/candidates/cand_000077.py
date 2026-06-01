import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
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

    # Precompute segment distances for DP
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0, tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r - 2], tour[r - 1]]
            if r == l + 1:
                seg_dist[l][r] = distance_matrix[0, tour[l]] + distance_matrix[tour[l], 0]
            else:
                seg_dist[l][r] = acc + distance_matrix[tour[r - 1], 0]

    # DP for minimax split
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

    # Reconstruct routes from DP
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

    best_routes = [list(r) for r in routes]
    best_max = compute_max()
    report_best_vrp(best_routes)

    # --- VND improvement ---
    max_iter = n * 2
    for _ in range(max_iter):
        # Identify longest route
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: (dists[i], i))
        longest = routes[longest_idx]

        improved = False

        # 1. 2-opt on longest route
        if len(longest) > 3:
            best_local_dist = dists[longest_idx]
            best_local_route = longest[:]
            for i in range(1, len(longest) - 2):
                for j in range(i + 1, len(longest) - 1):
                    new_route = longest[:i] + longest[i:j+1][::-1] + longest[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_local_dist:
                        best_local_dist = new_dist
                        best_local_route = new_route
            if best_local_dist < dists[longest_idx]:
                routes[longest_idx] = best_local_route
                improved = True
                new_max = compute_max()
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)

        if improved:
            continue

        # 2. Relocate from longest route with regret-like selection
        # Evaluate all possible relocate moves, choose the one that minimizes new max
        best_new_max = math.inf
        best_move = None
        for cust_pos in range(1, len(longest) - 1):
            cust = longest[cust_pos]
            new_longest = longest[:cust_pos] + longest[cust_pos+1:]
            if len(new_longest) < 2:
                continue
            new_longest_dist = route_dist(new_longest)
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    new_dst_dist = route_dist(new_dst)
                    other_dists = [route_dist(r) for i, r in enumerate(routes) if i not in (longest_idx, dst_idx)]
                    candidate_max = max(new_longest_dist, new_dst_dist, *other_dists)
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_move = (cust_pos, longest_idx, dst_idx, pos, new_longest, new_dst)
        if best_move is not None and best_new_max < compute_max():
            cust_pos, src_idx, dst_idx, pos, new_longest, new_dst = best_move
            routes[src_idx] = new_longest
            routes[dst_idx] = new_dst
            improved = True
            new_max = compute_max()
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)

        if improved:
            continue

        # 3. Swap between longest and another route
        best_new_max = math.inf
        best_move = None
        for pos_i in range(1, len(longest) - 1):
            cust_i = longest[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos_j in range(1, len(dst_route) - 1):
                    cust_j = dst_route[pos_j]
                    new_src = longest[:pos_i] + [cust_j] + longest[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust_i] + dst_route[pos_j+1:]
                    new_src_dist = route_dist(new_src)
                    new_dst_dist = route_dist(new_dst)
                    other_dists = [route_dist(r) for i, r in enumerate(routes) if i not in (longest_idx, dst_idx)]
                    candidate_max = max(new_src_dist, new_dst_dist, *other_dists)
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_move = (longest_idx, dst_idx, pos_i, pos_j, new_src, new_dst)
        if best_move is not None and best_new_max < compute_max():
            src_idx, dst_idx, pos_i, pos_j, new_src, new_dst = best_move
            routes[src_idx] = new_src
            routes[dst_idx] = new_dst
            improved = True
            new_max = compute_max()
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)

        if not improved:
            break

    return best_routes