import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    customer_count = n - 1
    if truck_count <= 0:
        return []
    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        return routes

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    # 1. Build giant TSP tour via nearest neighbor (deterministic)
    perm = []
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: (distance_matrix[current, x], x))
        perm.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    # 2-opt on full tour (including depot)
    def two_opt_tour(tour):
        best = [0] + tour + [0]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_dist(new_route) < route_dist(best) - 1e-12:
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best[1:-1]
    perm = two_opt_tour(perm)
    m = len(perm)

    # 3. Precompute segment distances for DP
    start_to_depot = np.array([distance_matrix[0, c] for c in perm])
    end_to_depot = np.array([distance_matrix[c, 0] for c in perm])
    cum_inter = np.zeros(m + 1)
    for i in range(1, m):
        cum_inter[i] = cum_inter[i-1] + distance_matrix[perm[i-1], perm[i]]
    cum_inter[m] = cum_inter[m-1]
    def seg_dist(l, r):
        if l > r:
            return 0.0
        return start_to_depot[l] + (cum_inter[r] - cum_inter[l]) + end_to_depot[r]

    # DP optimal split minimizing max distance
    K = min(truck_count, m)
    INF = float('inf')
    dp = [[INF] * (m + 1) for _ in range(K + 1)]
    choice = [[-1] * (m + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0
    for t in range(1, K + 1):
        for i in range(t, m + 1):
            best_val = INF
            best_j = -1
            for j in range(t - 1, i):
                cand = max(dp[t-1][j], seg_dist(j, i-1))
                if cand < best_val - 1e-12:
                    best_val = cand
                    best_j = j
                elif abs(cand - best_val) < 1e-12 and best_j > j:
                    best_j = j
            dp[t][i] = best_val
            choice[t][i] = best_j

    # Reconstruct routes from DP
    def split_perm(perm):
        m = len(perm)
        K = min(truck_count, m)
        routes = []
        i = m
        for t in range(K, 0, -1):
            j = choice[t][i]
            l = j
            r = i - 1
            if l > r:
                routes.append([0, 0])
            else:
                route = [0] + perm[l:r+1] + [0]
                routes.append(route)
            i = j
        routes.reverse()
        # Add empty routes if necessary
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    routes = split_perm(perm)
    for idx in range(truck_count):
        routes[idx] = two_opt_tour(routes[idx][1:-1]) if len(routes[idx]) > 2 else routes[idx]
    best_routes = [r[:] for r in routes]
    best_max = max_dist(best_routes)
    report_best_vrp(best_routes)

    # 4. Permutation swap local search with DP re-split
    max_iter = m * truck_count
    for _ in range(max_iter):
        improved = False
        for i in range(m):
            for j in range(i + 1, m):
                new_perm = perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_routes = split_perm(new_perm)
                for idx in range(truck_count):
                    if len(new_routes[idx]) > 2:
                        new_routes[idx] = two_opt_tour(new_routes[idx][1:-1])
                new_max = max_dist(new_routes)
                if new_max < best_max - 1e-12:
                    best_max = new_max
                    best_routes = [r[:] for r in new_routes]
                    perm = new_perm
                    improved = True
                    report_best_vrp(best_routes)
                    break
            if improved:
                break
        if not improved:
            break

    # 5. Inter-route relocate from longest route
    max_iter2 = n * truck_count
    for _ in range(max_iter2):
        max_val = max_dist(routes)
        max_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        best_move = None
        best_new_max = max_val
        route_max = routes[max_idx]
        for i in range(1, len(route_max) - 1):
            c = route_max[i]
            pred = route_max[i-1]
            succ = route_max[i+1]
            new_max_dist = route_dist(route_max) - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    pred_o = other_route[pos-1]
                    succ_o = other_route[pos]
                    new_other = route_dist(other_route) - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                    other_max = 0.0
                    for k, d in enumerate([route_dist(r) for r in routes]):
                        if k != max_idx and k != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_max_dist, new_other)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (i, other_idx, pos, new_max_dist, new_other)
        if best_move is None:
            break
        i, other_idx, pos, new_max_dist, new_other = best_move
        c = routes[max_idx].pop(i)
        routes[other_idx].insert(pos, c)
        # Update distances implicitly via route_dist calls
        for r_idx in [max_idx, other_idx]:
            routes[r_idx] = two_opt_tour(routes[r_idx][1:-1]) if len(routes[r_idx]) > 2 else routes[r_idx]
        report_best_vrp(routes)
        cur_max = max_dist(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]

    # 6. Inter-route 2-opt* (swap suffixes)
    max_iter3 = n * truck_count
    for _ in range(max_iter3):
        max_val = max_dist(routes)
        max_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        best_move = None
        best_new_max = max_val
        route_max = routes[max_idx]
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_route = routes[other_idx]
            for i in range(1, len(route_max) - 1):
                for j in range(1, len(other_route) - 1):
                    old1 = distance_matrix[route_max[i], route_max[i+1]]
                    old2 = distance_matrix[other_route[j], other_route[j+1]]
                    new1 = distance_matrix[route_max[i], other_route[j+1]]
                    new2 = distance_matrix[other_route[j], route_max[i+1]]
                    new_dist_max = route_dist(route_max) - old1 + new1
                    new_dist_other = route_dist(other_route) - old2 + new2
                    other_max = 0.0
                    for k, d in enumerate([route_dist(r) for r in routes]):
                        if k != max_idx and k != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_dist_max, new_dist_other)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (i, other_idx, j, new_dist_max, new_dist_other)
        if best_move is None:
            break
        i, other_idx, j, new_dist_max, new_dist_other = best_move
        route_max = routes[max_idx]
        other_route = routes[other_idx]
        new_route_max = route_max[:i+1] + other_route[j+1:]
        new_route_other = other_route[:j+1] + route_max[i+1:]
        routes[max_idx] = new_route_max
        routes[other_idx] = new_route_other
        for r_idx in [max_idx, other_idx]:
            routes[r_idx] = two_opt_tour(routes[r_idx][1:-1]) if len(routes[r_idx]) > 2 else routes[r_idx]
        report_best_vrp(routes)
        cur_max = max_dist(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]

    return best_routes