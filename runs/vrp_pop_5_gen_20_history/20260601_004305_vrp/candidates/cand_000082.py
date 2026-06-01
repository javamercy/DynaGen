import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # --- Initial solution: TSP tour + DP split (minimax) ---
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current][v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0][tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r - 2]][tour[r - 1]]
            if r == l + 1:
                seg_dist[l][r] = distance_matrix[0][tour[l]] + distance_matrix[tour[l]][0]
            else:
                seg_dist[l][r] = acc + distance_matrix[tour[r - 1]][0]

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

    # Helper functions
    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    current = copy_routes(routes)
    current_max = compute_max(current)
    best = copy_routes(current)
    best_max = current_max
    report_best_vrp(best)

    n_cust = m
    q = max(1, n_cust // 10)

    # Define destroy and repair functions
    def random_removal(routes, q):
        removed = []
        new_routes = copy_routes(routes)
        all_cust = [c for route in new_routes for c in route if c != 0]
        random.shuffle(all_cust)
        for c in all_cust[:q]:
            for route in new_routes:
                if c in route:
                    route.remove(c)
                    removed.append(c)
                    break
        return new_routes, removed

    def worst_removal(routes, q):
        removed = []
        new_routes = copy_routes(routes)
        detour = {}
        for route in new_routes:
            for p in range(1, len(route)-1):
                c = route[p]
                prev = route[p-1]
                nxt = route[p+1]
                det = distance_matrix[prev][c] + distance_matrix[c][nxt] - distance_matrix[prev][nxt]
                detour[c] = det
        sorted_cust = sorted(detour.items(), key=lambda x: -x[1])
        for c, _ in sorted_cust[:q]:
            for route in new_routes:
                if c in route:
                    route.remove(c)
                    removed.append(c)
                    break
        return new_routes, removed

    def greedy_repair(routes, removed):
        for c in removed:
            best_inc = math.inf
            best_ri = -1
            best_pos = -1
            for ri, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [c] + route[pos:]
                    new_dist = route_dist(new_route)
                    other_dists = [route_dist(r) for ri2, r in enumerate(routes) if ri2 != ri]
                    new_max = max(new_dist, *other_dists)
                    if new_max < best_inc or (new_max == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                        best_inc = new_max
                        best_ri = ri
                        best_pos = pos
            routes[best_ri].insert(best_pos, c)
        return routes

    def regret2_repair(routes, removed):
        while removed:
            best_c = -1
            best_regret = -1
            best_ri = -1
            best_pos = -1
            best_max_val = math.inf
            for c in removed:
                first = (math.inf, -1, -1)
                second = (math.inf, -1, -1)
                for ri, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [c] + route[pos:]
                        new_dist = route_dist(new_route)
                        other_dists = [route_dist(r) for ri2, r in enumerate(routes) if ri2 != ri]
                        new_max = max(new_dist, *other_dists)
                        if new_max < first[0]:
                            second = first
                            first = (new_max, ri, pos)
                        elif new_max < second[0]:
                            second = (new_max, ri, pos)
                if first[0] == math.inf:
                    continue
                regret = second[0] - first[0]
                if regret > best_regret or (regret == best_regret and c < best_c):
                    best_regret = regret
                    best_c = c
                    best_ri = first[1]
                    best_pos = first[2]
                    best_max_val = first[0]
            if best_c != -1:
                routes[best_ri].insert(best_pos, best_c)
                removed.remove(best_c)
        return routes

    operator_pairs = [
        (random_removal, greedy_repair),
        (random_removal, regret2_repair),
        (worst_removal, greedy_repair),
        (worst_removal, regret2_repair)
    ]

    # ALNS phase: each operator pair applied iter times
    iter_per_pair = 10
    for destroy, repair in operator_pairs:
        for _ in range(iter_per_pair):
            new_routes, removed = destroy(current, q)
            new_routes = repair(new_routes, removed)
            new_max = compute_max(new_routes)
            if new_max <= current_max:  # accept non-worsening
                current = new_routes
                current_max = new_max
                if new_max < best_max:
                    best = copy_routes(new_routes)
                    best_max = new_max
                    report_best_vrp(best)

    # Local search phase (deterministic best improvement)
    max_passes = n * n
    for _ in range(max_passes):
        dists = [route_dist(r) for r in current]
        current_max = max(dists)
        improved = False

        # Order routes by distance descending, then index ascending
        order = sorted(range(len(current)), key=lambda i: (-dists[i], i))

        # Best 2-opt on each route
        for idx in order:
            route = current[idx]
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
                current[idx] = best_route
                new_max = compute_max(current)
                if new_max < best_max:
                    best_max = new_max
                    best = copy_routes(current)
                    report_best_vrp(best)
                improved = True
                break  # restart pass after any improvement

        if improved:
            continue

        # Best relocate from the longest route
        dists = [route_dist(r) for r in current]
        current_max = max(dists)
        order = sorted(range(len(current)), key=lambda i: (-dists[i], i))
        longest_idx = order[0]

        best_improvement = 0.0
        best_move = None
        src_route = current[longest_idx]
        for cust_pos in range(1, len(src_route) - 1):
            cust = src_route[cust_pos]
            new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
            dist_src = route_dist(new_src)
            for dst_idx in range(len(current)):
                if dst_idx == longest_idx:
                    continue
                dst_route = current[dst_idx]
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    dist_dst = route_dist(new_dst)
                    other_dists = [route_dist(r) for i, r in enumerate(current) if i not in (longest_idx, dst_idx)]
                    new_max = max([dist_src, dist_dst] + other_dists)
                    improvement = current_max - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (cust, cust_pos, dst_idx, pos)
                    elif improvement == best_improvement and best_move is not None:
                        (ocust, ocust_pos, odst_idx, opos) = best_move
                        if cust < ocust or (cust == ocust and cust_pos < ocust_pos) or (cust == ocust and cust_pos == ocust_pos and dst_idx < odst_idx) or (cust == ocust and cust_pos == ocust_pos and dst_idx == odst_idx and pos < opos):
                            best_improvement = improvement
                            best_move = (cust, cust_pos, dst_idx, pos)

        if best_move and best_improvement > 0:
            cust, cust_pos, dst_idx, pos = best_move
            current[longest_idx].pop(cust_pos)
            current[dst_idx].insert(pos, cust)
            new_max = compute_max(current)
            if new_max < best_max:
                best_max = new_max
                best = copy_routes(current)
                report_best_vrp(best)
            improved = True

        if not improved:
            break

    return best