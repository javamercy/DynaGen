import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def split_permutation(perm):
        # DP: dp[i][k] = min possible max distance for first i customers (0-indexed) with k routes
        # k from 1 to truck_count, i from 1 to len(perm)
        m = len(perm)
        # Precompute segment distances from i to j (customers i..j inclusive) including depot before and after
        # segment distance = dist(0, perm[i]) + sum_{t=i}^{j-1} dist(perm[t], perm[t+1]) + dist(perm[j], 0)
        seg_dist = [[0.0]*m for _ in range(m)]
        for i in range(m):
            d = distance_matrix[0][perm[i]]
            for j in range(i, m):
                if j > i:
                    d += distance_matrix[perm[j-1]][perm[j]]
                seg_dist[i][j] = d + distance_matrix[perm[j]][0]
        # DP
        INF = float('inf')
        # dp[k][i] for k routes covering first i customers (i from 0 to m)
        dp = [[INF]*(m+1) for _ in range(truck_count+1)]
        prev = [[-1]*(m+1) for _ in range(truck_count+1)]
        dp[0][0] = 0.0
        for k in range(1, truck_count+1):
            for i in range(k, m+1):
                for j in range(k-1, i):
                    cand = max(dp[k-1][j], seg_dist[j][i-1])
                    if cand < dp[k][i]:
                        dp[k][i] = cand
                        prev[k][i] = j
        best_max = dp[truck_count][m]
        # Reconstruct routes
        routes = []
        i = m
        k = truck_count
        while k > 0:
            j = prev[k][i]
            segment = perm[j:i]
            route = [0] + segment + [0]
            routes.insert(0, route)
            i = j
            k -= 1
        # Fill remaining trucks with empty routes if fewer than truck_count (should not happen for m >= truck_count, but just in case)
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes, best_max

    def improve_routes(routes):
        # Apply 2-opt to each route
        for r_idx in range(truck_count):
            route = routes[r_idx]
            improved = True
            while improved:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            break
                    if improved:
                        break
            routes[r_idx] = route
        return routes

    # Generate initial permutation: greedy nearest neighbor starting from depot
    perm = []
    current = 0
    unvisited = set(range(1, n))
    while unvisited:
        next_cust = min(unvisited, key=lambda x: distance_matrix[current][x])
        perm.append(next_cust)
        current = next_cust
        unvisited.remove(next_cust)
    # Optionally, also try random init? We'll keep greedy for deterministic start.
    # But we will later randomize during SA restarts.

    routes, best_max = split_permutation(perm)
    routes = improve_routes(routes)
    best_routes = [r[:] for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)

    # SA parameters
    max_iter = min(3000, n * 30)  # bounded by instance size
    T0 = 1000.0
    T_end = 1.0
    alpha = (T_end / T0) ** (1.0 / max_iter)
    T = T0

    stagnation_counter = 0
    no_improve_limit = min(100, max_iter // 10)

    for it in range(max_iter):
        # Adaptive temperature based on stagnation (like parent)
        if stagnation_counter > 0:
            T = max(T_end, T0 / (1.0 + 0.5 * stagnation_counter))
        else:
            T = T0 * (alpha ** it)

        # Generate neighbor: choose operator via softmax based on temperature
        ops = ['swap', 'invert']
        # Approximate temperature for selection: use T
        # Choose random if T high, else deterministic?
        if T > 100:
            op = random.choice(ops)
        elif random.random() < T / 100:
            op = random.choice(ops)
        else:
            # deterministic: prefer swap? Not needed; just use random
            op = random.choice(ops)
        new_perm = perm[:]
        if op == 'swap':
            i, j = random.sample(range(len(perm)), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:  # invert
            i, j = sorted(random.sample(range(len(perm)), 2))
            new_perm[i:j+1] = reversed(new_perm[i:j+1])

        new_routes, new_max = split_permutation(new_perm)
        new_routes = improve_routes(new_routes)
        new_max = max(route_dist(r) for r in new_routes)

        delta = new_max - best_max
        if delta < -1e-12 or (delta <= 0 and random.random() < math.exp(-delta / max(T, 1e-12))):
            perm = new_perm
            routes = new_routes
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        else:
            stagnation_counter += 1

        # Restart after prolonged stagnation
        if stagnation_counter >= no_improve_limit:
            # Random permutation restart
            perm = list(range(1, n))
            random.shuffle(perm)
            routes, new_max = split_permutation(perm)
            routes = improve_routes(routes)
            new_max = max(route_dist(r) for r in routes)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            stagnation_counter = 0

    return best_routes