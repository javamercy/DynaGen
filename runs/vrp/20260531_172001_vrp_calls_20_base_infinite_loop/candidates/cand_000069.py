import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    # Precompute segment costs for a given permutation
    def segment_cost(perm, i, j):
        # perm is list of customers (not including depot), 0-indexed
        if i > j:
            return 0.0
        cost = distance_matrix[0, perm[i]]
        for idx in range(i, j):
            cost += distance_matrix[perm[idx], perm[idx+1]]
        cost += distance_matrix[perm[j], 0]
        return cost

    def split_perm(perm):
        L = len(perm)
        K = truck_count
        # dp[k][i] = min max route length for first i customers with k routes
        dp = [[float('inf')] * (L+1) for _ in range(K+1)]
        dp[0][0] = 0.0
        # precompute segment costs for all i,j
        seg_cost = [[0.0]*L for _ in range(L)]
        for i in range(L):
            for j in range(i, L):
                seg_cost[i][j] = segment_cost(perm, i, j)
        # reconstruction
        prev = [[-1]*(L+1) for _ in range(K+1)]
        for k in range(1, K+1):
            for i in range(1, L+1):
                # try last segment from j to i-1
                best_val = float('inf')
                best_j = -1
                for j in range(0, i):
                    if dp[k-1][j] < float('inf'):
                        val = max(dp[k-1][j], seg_cost[j][i-1])
                        if val < best_val:
                            best_val = val
                            best_j = j
                dp[k][i] = best_val
                prev[k][i] = best_j
        # reconstruct routes
        routes = []
        k = K
        i = L
        while k > 0:
            j = prev[k][i]
            seg_cust = perm[j:i] if j < i else []
            route = [0] + seg_cust + [0]
            routes.insert(0, route)
            k -= 1
            i = j
        # fill empty routes if not enough trucks used (shouldn't happen)
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes, dp[K][L]

    best_routes = None
    best_max = float('inf')

    # Number of restarts and SA iterations
    max_restarts = max(1, n // 8)
    max_sa_iters = max(10, n * 20)

    for restart in range(max_restarts):
        # Random initial permutation
        perm = list(range(1, n))
        random.shuffle(perm)

        routes, cur_max = split_perm(perm)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = routes
            report_best_vrp(routes)

        # Simulated annealing
        temp = 100.0
        cooling = 0.99
        min_temp = 0.01
        for it in range(max_sa_iters):
            # Generate neighbor by random move
            move_type = random.randint(0,2)
            new_perm = perm[:]
            if move_type == 0:  # swap two customers
                i, j = random.sample(range(len(perm)), 2)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            elif move_type == 1:  # insert: move a customer to another position
                i = random.randrange(len(perm))
                cust = new_perm.pop(i)
                j = random.randrange(len(perm))
                new_perm.insert(j, cust)
            else:  # 2-opt: reverse a segment
                i, j = sorted(random.sample(range(len(perm)), 2))
                new_perm[i:j+1] = reversed(new_perm[i:j+1])

            new_routes, new_max = split_perm(new_perm)
            delta = new_max - cur_max
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                perm = new_perm
                cur_max = new_max
                routes = new_routes
                if cur_max < best_max:
                    best_max = cur_max
                    best_routes = routes
                    report_best_vrp(routes)
            temp *= cooling
            if temp < min_temp:
                break

        # Optional: simple hill climbing on final permutation
        improved = True
        while improved:
            improved = False
            for i in range(len(perm)):
                for j in range(i+1, len(perm)):
                    # try swap
                    new_perm = perm[:]
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_routes, new_max = split_perm(new_perm)
                    if new_max < cur_max - 1e-12:
                        perm = new_perm
                        cur_max = new_max
                        routes = new_routes
                        improved = True
                        if cur_max < best_max:
                            best_max = cur_max
                            best_routes = routes
                            report_best_vrp(routes)
                        break
                if improved:
                    break

    if best_routes is None:
        best_routes = [[0,0] for _ in range(truck_count)]
    return best_routes