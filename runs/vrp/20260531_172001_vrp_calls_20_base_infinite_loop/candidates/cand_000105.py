import numpy as np
import random
from math import exp

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def decode(perm):
        """Decode permutation into routes minimizing max route length using DP."""
        N = len(perm)
        if N == 0:
            return [[0,0] for _ in range(truck_count)]
        # Precompute segment distances
        depot_dist = [distance_matrix[0, cust] for cust in perm]
        Np = N
        seg = [[0.0]*(Np+1) for _ in range(Np)]  # seg[s][e] for s<e
        for s in range(Np):
            cur = depot_dist[s]
            for e in range(s+1, Np+1):
                if e > s+1:
                    cur += distance_matrix[perm[e-2], perm[e-1]]
                seg[s][e] = cur + depot_dist[e-1]
        # DP: dp[i][k] = min max for first i customers with exactly k routes
        INF = 1e100
        dp = [[INF]*(truck_count+1) for _ in range(N+1)]
        dp[0][0] = 0.0
        # Fill DP
        for i in range(1, N+1):
            for k in range(1, min(i, truck_count)+1):
                best = INF
                for j in range(k-1, i):  # at least one customer per route before
                    if dp[j][k-1] < INF:
                        cand = max(dp[j][k-1], seg[j][i])
                        if cand < best:
                            best = cand
                dp[i][k] = best
        # Find best max and k
        best_max = INF
        best_k = 1
        for k in range(1, truck_count+1):
            if dp[N][k] < best_max:
                best_max = dp[N][k]
                best_k = k
        # Reconstruct routes
        routes = [[0,0] for _ in range(truck_count)]
        i = N
        k = best_k
        while i > 0 and k > 0:
            for j in range(k-1, i):
                if dp[j][k-1] < INF and max(dp[j][k-1], seg[j][i]) == dp[i][k]:
                    # route from j to i (j customers before, then customers j..i-1)
                    route = [0] + perm[j:i] + [0]
                    # assign to first available truck (or later we can assign to any)
                    for t in range(truck_count):
                        if len(routes[t]) == 2:
                            routes[t] = route
                            break
                    i = j
                    k -= 1
                    break
        # Fill remaining trucks with empty routes
        for t in range(truck_count):
            if len(routes[t]) == 2 and routes[t][0]==0 and routes[t][1]==0:
                pass  # already empty
            # ensure all routes are correct
        return routes, best_max

    # Build initial permutation from regret construction (like parent)
    def initial_permutation():
        routes = [[0,0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                insert_info = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0]+1e9, best[1]+1e9, -1, -1)
                regret = second[0] - best[0]
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        # Concatenate routes to form permutation
        perm = []
        for r in routes:
            perm.extend(r[1:-1])
        return perm, routes

    perm, best_routes = initial_permutation()
    best_routes, best_max = decode(perm)
    report_best_vrp(best_routes)

    current_perm = perm[:]
    current_routes, current_max = decode(current_perm)

    max_iter = n * truck_count * 10
    initial_temp = current_max
    cooling_rate = 0.99
    stagnation_limit = max(10, n // 2)
    last_improvement = 0

    for it in range(max_iter):
        temp = initial_temp * (cooling_rate ** it)
        if temp < 1e-12:
            temp = 1e-12

        # Choose neighborhood
        neigh = random.choice(['swap', 'insert', 'invert'])
        new_perm = current_perm[:]
        if neigh == 'swap':
            i, j = random.sample(range(len(new_perm)), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        elif neigh == 'insert':
            i, j = random.sample(range(len(new_perm)), 2)
            cust = new_perm.pop(i)
            new_perm.insert(j, cust)
        else:  # invert
            i, j = sorted(random.sample(range(len(new_perm)), 2))
            new_perm[i:j+1] = reversed(new_perm[i:j+1])

        new_routes, new_max = decode(new_perm)
        delta = new_max - current_max
        if delta <= 0 or random.random() < exp(-delta / temp):
            current_perm = new_perm
            current_routes = new_routes
            current_max = new_max
            if current_max < best_max - 1e-9:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                report_best_vrp(best_routes)
                last_improvement = it
        # Check stagnation
        if it - last_improvement > stagnation_limit:
            # Ruin-recreate perturbation
            m = len(current_perm)
            remove_count = max(1, int(m * 0.2))
            indices = random.sample(range(m), remove_count)
            removed = [current_perm[i] for i in sorted(indices, reverse=True)]
            for i in sorted(indices, reverse=True):
                current_perm.pop(i)
            random.shuffle(removed)
            for cust in removed:
                pos = random.randint(0, len(current_perm))
                current_perm.insert(pos, cust)
            # Evaluate
            current_routes, current_max = decode(current_perm)
            if current_max < best_max - 1e-9:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                report_best_vrp(best_routes)
            last_improvement = it

    return best_routes