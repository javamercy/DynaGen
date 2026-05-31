import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    n_cust = len(customers)
    if truck_count <= 0:
        return []
    if truck_count >= n_cust:
        routes = [[0, 0] for _ in range(truck_count)]
        for i, cust in enumerate(customers):
            routes[i] = [0, cust, 0]
        return routes

    # 1. Build giant TSP tour using nearest neighbor (deterministic)
    unvisited = set(customers)
    tour = [0]
    current = 0
    while unvisited:
        nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    tour.append(0)
    giant_tour = tour[1:-1]
    n_c = len(giant_tour)

    # 2. Precompute segment distances
    seg = [[0.0] * n_c for _ in range(n_c)]
    for i in range(n_c):
        d = distance_matrix[0, giant_tour[i]]
        seg[i][i] = d + distance_matrix[giant_tour[i], 0]
        for j in range(i+1, n_c):
            d += distance_matrix[giant_tour[j-1], giant_tour[j]]
            seg[i][j] = d + distance_matrix[giant_tour[j], 0]

    # 3. DP to split into exactly K non-empty routes (K = truck_count)
    K = min(truck_count, n_c)
    INF = 1e15
    dp = [[INF] * (n_c + 1) for _ in range(K + 1)]
    parent = [[-1] * (n_c + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0
    for k in range(1, K+1):
        for i in range(k, n_c+1):
            for j in range(k-1, i):
                cand = max(dp[k-1][j], seg[j][i-1])
                if cand < dp[k][i]:
                    dp[k][i] = cand
                    parent[k][i] = j

    # Reconstruct routes
    routes = []
    k = K
    i = n_c
    while k > 0:
        j = parent[k][i]
        segment = giant_tour[j:i]
        route = [0] + segment + [0]
        routes.append(route)
        i = j
        k -= 1
    routes.reverse()
    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    route_dists = [route_dist(r) for r in routes]
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # Intra-route 2-opt until local optimum
    for r_idx in range(truck_count):
        improved = True
        while improved:
            improved = False
            route = routes[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        route_dists[r_idx] = route_dist(route)
                        break
                if improved:
                    break
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    # Inter-route relocate: focus on longest route
    max_iter = n * truck_count
    for _ in range(max_iter):
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        moved = False
        best_move = None
        best_new_max = max_dist
        route_max = routes[max_idx]
        for i in range(1, len(route_max)-1):
            c = route_max[i]
            pred = route_max[i-1]
            succ = route_max[i+1]
            new_max_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    pred_o = other_route[pos-1]
                    succ_o = other_route[pos]
                    new_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                    other_max = 0.0
                    for j, d in enumerate(route_dists):
                        if j != max_idx and j != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_max_dist, new_other)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (i, other_idx, pos, new_max_dist, new_other)
        if best_move is not None:
            i, other_idx, pos, new_max_dist, new_other = best_move
            c = route_max.pop(i)
            routes[other_idx].insert(pos, c)
            route_dists[max_idx] = new_max_dist
            route_dists[other_idx] = new_other
            for r_idx in [max_idx, other_idx]:
                improved = True
                while improved:
                    improved = False
                    route = routes[r_idx]
                    for a in range(1, len(route)-2):
                        for b in range(a+1, len(route)-1):
                            old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                            new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                            if new < old - 1e-12:
                                route[a:b+1] = reversed(route[a:b+1])
                                improved = True
                                route_dists[r_idx] = route_dist(route)
                                break
                        if improved:
                            break
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        else:
            break

    # Inter-route 2-opt* (swap suffixes)
    max_iter2 = n * truck_count
    for _ in range(max_iter2):
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        best_move = None
        best_new_max = max_dist
        route_max = routes[max_idx]
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_route = routes[other_idx]
            for i in range(1, len(route_max)-1):
                for j in range(1, len(other_route)-1):
                    if route_max[-1] != 0 or other_route[-1] != 0:
                        continue
                    old1 = distance_matrix[route_max[i], route_max[i+1]]
                    old2 = distance_matrix[other_route[j], other_route[j+1]]
                    new1 = distance_matrix[route_max[i], other_route[j+1]]
                    new2 = distance_matrix[other_route[j], route_max[i+1]]
                    new_dist_max = route_dists[max_idx] - old1 + new1
                    new_dist_other = route_dists[other_idx] - old2 + new2
                    other_max = 0.0
                    for k, d in enumerate(route_dists):
                        if k != max_idx and k != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_dist_max, new_dist_other)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (i, other_idx, j, new_dist_max, new_dist_other)
        if best_move is not None:
            i, other_idx, j, new_dist_max, new_dist_other = best_move
            route_max = routes[max_idx]
            other_route = routes[other_idx]
            new_route_max = route_max[:i+1] + other_route[j+1:]
            new_route_other = other_route[:j+1] + route_max[i+1:]
            routes[max_idx] = new_route_max
            routes[other_idx] = new_route_other
            route_dists[max_idx] = route_dist(new_route_max)
            route_dists[other_idx] = route_dist(new_route_other)
            for r_idx in [max_idx, other_idx]:
                improved = True
                while improved:
                    improved = False
                    route = routes[r_idx]
                    for a in range(1, len(route)-2):
                        for b in range(a+1, len(route)-1):
                            old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                            new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                            if new < old - 1e-12:
                                route[a:b+1] = reversed(route[a:b+1])
                                improved = True
                                route_dists[r_idx] = route_dist(route)
                                break
                        if improved:
                            break
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        else:
            break

    return best_routes