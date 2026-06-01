import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        maxd = max(route_distance(r) for r in routes)
        if maxd < best_max - 1e-12:
            best_max = maxd
            best_routes = [list(r) for r in routes]

    # DP split: given permutation, produce routes minimizing max distance
    def split_permutation(perm, K):
        m = len(perm)
        INF = 1e100
        # precompute segment distances
        seg = [[0.0]*m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                d = distance_matrix[0, perm[i]] + distance_matrix[perm[j], 0]
                for k in range(i, j):
                    d += distance_matrix[perm[k], perm[k+1]]
                seg[i][j] = d
        dp = [[INF]*(K+1) for _ in range(m+1)]
        dp[0][0] = 0.0
        pred = [[-1]*(K+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for k in range(1, min(i, K)+1):
                best = INF
                best_j = -1
                for j in range(k-1, i):
                    if dp[j][k-1] >= INF:
                        continue
                    cand = max(dp[j][k-1], seg[j][i-1])
                    if cand < best - 1e-12:
                        best = cand
                        best_j = j
                dp[i][k] = best
                pred[i][k] = best_j
        # reconstruct routes
        routes = []
        i = m
        k = K
        while k > 0:
            j = pred[i][k]
            if i > j:
                segment = [0] + perm[j:i] + [0]
            else:
                segment = [0, 0]
            routes.insert(0, segment)
            i = j
            k -= 1
        while len(routes) < K:
            routes.append([0, 0])
        maxd = max(route_distance(r) for r in routes)
        return routes, maxd

    # Generate candidate permutations
    # Sorted by distance from depot ascending
    perm_asc = sorted(customers, key=lambda x: distance_matrix[0, x])
    # Sorted descending
    perm_desc = sorted(customers, key=lambda x: -distance_matrix[0, x])
    # Rotations of asc (deterministic)
    candidates = [perm_asc, perm_desc]
    for shift in range(1, 5):  # add 4 more rotations
        perm = perm_asc[shift:] + perm_asc[:shift]
        candidates.append(perm)

    # Evaluate each candidate
    for perm in candidates:
        routes, maxd = split_permutation(perm, truck_count)
        report_best_vrp(routes)

    # 2-opt local search on best routes
    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    gain = route_distance(route) - route_distance(new_route)
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_swap = new_route
            if best_swap is not None:
                route = best_swap
                improved = True
        return route

    if best_routes is not None:
        improved_routes = [two_opt(r) for r in best_routes]
        report_best_vrp(improved_routes)

    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes