import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    N = len(customers)
    if N == 0:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= N:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    # Build giant TSP tour using nearest neighbor from depot
    tour = [0]
    unvisited = set(customers)
    current = 0
    while unvisited:
        next_cust = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_cust)
        unvisited.remove(next_cust)
        current = next_cust
    tour.append(0)
    P = tour[1:-1]  # permutation of customers
    N = len(P)

    # Precompute prefix sums for internal distances in tour
    pref = [0] * N
    for k in range(1, N):
        pref[k] = pref[k-1] + distance_matrix[P[k-1], P[k]]
    def seg_dist(i, j):
        # segment from i to j inclusive
        return distance_matrix[0, P[i]] + (pref[j] - pref[i]) + distance_matrix[P[j], 0]

    # DP to split into exactly truck_count routes minimizing max segment distance
    K = truck_count
    INF = 1e18
    dp = [[INF] * (K+1) for _ in range(N+1)]
    split = [[-1] * (K+1) for _ in range(N+1)]
    dp[0][0] = 0
    for i in range(1, N+1):
        for k in range(1, min(i, K)+1):
            best = INF
            best_j = -1
            for j in range(i-1, -1, -1):
                if dp[j][k-1] >= INF:
                    continue
                seg = seg_dist(j, i-1)
                cand = max(dp[j][k-1], seg)
                if cand < best:
                    best = cand
                    best_j = j
            dp[i][k] = best
            split[i][k] = best_j

    # Backtrack
    routes = []
    i = N
    k = K
    while k > 0:
        j = split[i][k]
        seg_custs = P[j:i]
        route = [0] + seg_custs + [0]
        routes.append(route)
        i = j
        k -= 1
    routes.reverse()
    while len(routes) < K:
        routes.append([0, 0])

    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in routes)

    def report_best(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]
            try:
                report_best_vrp(routes)
            except NameError:
                pass

    report_best(best_routes)

    # Local search: Or-opt + SA
    random.seed(0)
    current_routes = [list(r) for r in routes]
    current_max = best_max
    max_iter = N * K * 2
    T_start = current_max * 0.1 if current_max > 0 else 1.0
    T_end = current_max * 0.001 if current_max > 0 else 0.01
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max_iter) if max_iter > 0 else 1.0

    for _ in range(max_iter):
        best_move = None
        best_new_max = current_max

        # Intra-route Or-opt: relocate chain of length 1-3 within same route
        for r_idx in range(K):
            route = current_routes[r_idx]
            if len(route) <= 4:
                continue
            for a in range(1, len(route)-2):
                for length in [1, 2, 3]:
                    b = a + length - 1
                    if b >= len(route)-2:
                        break
                    chain = route[a:b+1]
                    new_route = route[:a] + route[b+1:]
                    for p in range(1, len(new_route)-1):
                        cand_route = new_route[:p] + chain + new_route[p:]
                        new_dists = [route_distance(r) for r in current_routes]
                        new_dists[r_idx] = route_distance(cand_route)
                        new_max = max(new_dists)
                        if new_max < best_new_max - 1e-12 or (abs(new_max - best_new_max) < 1e-12 and (r_idx < (best_move[1] if best_move else K))):
                            best_new_max = new_max
                            best_move = ('intra', r_idx, a, b, p, cand_route)

        # Inter-route Or-opt: relocate chain from one route to another
        for r1_idx in range(K):
            r1 = current_routes[r1_idx]
            if len(r1) <= 4:
                continue
            for a in range(1, len(r1)-2):
                for length in [1, 2, 3]:
                    b = a + length - 1
                    if b >= len(r1)-2:
                        break
                    chain = r1[a:b+1]
                    new_r1 = r1[:a] + r1[b+1:]
                    for r2_idx in range(K):
                        if r1_idx == r2_idx:
                            continue
                        r2 = current_routes[r2_idx]
                        for p in range(1, len(r2)):
                            cand_r2 = r2[:p] + chain + r2[p:]
                            new_dists = [route_distance(r) for r in current_routes]
                            new_dists[r1_idx] = route_distance(new_r1)
                            new_dists[r2_idx] = route_distance(cand_r2)
                            new_max = max(new_dists)
                            if new_max < best_new_max - 1e-12 or (abs(new_max - best_new_max) < 1e-12 and (r1_idx < (best_move[1] if best_move else K) or (r1_idx == best_move[1] and r2_idx < best_move[2]))):
                                best_new_max = new_max
                                best_move = ('inter', r1_idx, r2_idx, a, b, p, new_r1, cand_r2)

        if best_move is None:
            break
        # Apply move
        if best_move[0] == 'intra':
            _, r_idx, a, b, p, new_route = best_move
            current_routes[r_idx] = new_route
        else:
            _, r1_idx, r2_idx, a, b, p, new_r1, new_r2 = best_move
            current_routes[r1_idx] = new_r1
            current_routes[r2_idx] = new_r2
        current_max = best_new_max
        if current_max < best_max:
            report_best(current_routes)
            best_max = current_max
        T *= cooling

    report_best(best_routes)
    return best_routes