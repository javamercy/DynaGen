import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    num_ants = 10
    max_iter = 10
    alpha = 1.0
    beta = 2.0
    rho_global = 0.1
    rho_local = 0.1

    # Heuristic matrix (1/distance)
    heuristic = 1.0 / (distance_matrix + 1e-10)
    # Initial pheromone
    nonzero = distance_matrix[distance_matrix > 0]
    if len(nonzero) == 0:
        tau0 = 1.0
    else:
        tau0 = 1.0 / (n * np.mean(nonzero))
    tau = np.full((n, n), tau0)

    best_global_max = float('inf')
    best_global_routes = None
    best_global_perm = None

    def compute_segment_length(perm, start, end):
        if start >= end:
            return 0.0
        custs = perm[start:end]
        length = distance_matrix[0, custs[0]]
        for k in range(len(custs)-1):
            length += distance_matrix[custs[k], custs[k+1]]
        length += distance_matrix[custs[-1], 0]
        return length

    def split_permutation(perm):
        m = len(perm)
        if m == 0:
            return 0.0, [[0, 0] for _ in range(truck_count)]
        K = truck_count
        dp = [[float('inf')] * (K + 1) for _ in range(m + 1)]
        prev = [[-1] * (K + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(1, m + 1):
            for t in range(1, min(i, K) + 1):
                best = float('inf')
                best_j = -1
                # j must be at least t-1 to have enough customers for previous routes
                for j in range(t-1, i):
                    seg_len = compute_segment_length(perm, j, i)
                    candidate = max(dp[j][t-1], seg_len)
                    if candidate < best:
                        best = candidate
                        best_j = j
                dp[i][t] = best
                prev[i][t] = best_j
        # Find best t
        best_max = float('inf')
        best_t = -1
        for t in range(1, K + 1):
            if dp[m][t] < best_max:
                best_max = dp[m][t]
                best_t = t
        # Reconstruct routes
        routes = []
        i = m
        t = best_t
        while t > 0:
            j = prev[i][t]
            route_customers = perm[j:i]
            route = [0] + route_customers + [0]
            routes.append(route)
            i = j
            t -= 1
        while len(routes) < K:
            routes.append([0, 0])
        routes.reverse()
        return best_max, routes

    for iteration in range(max_iter):
        for ant in range(num_ants):
            # Construct permutation
            unvisited = set(range(1, n))
            current = 0
            perm = []
            while unvisited:
                # Compute probabilities
                probs = []
                custs_list = list(unvisited)
                for cust in custs_list:
                    prob = (tau[current, cust] ** alpha) * (heuristic[current, cust] ** beta)
                    probs.append(prob)
                total = sum(probs)
                if total > 1e-12:
                    probs = [p / total for p in probs]
                    chosen = random.choices(custs_list, weights=probs, k=1)[0]
                else:
                    chosen = random.choice(custs_list)
                perm.append(chosen)
                # Local pheromone update
                tau[current, chosen] = (1 - rho_local) * tau[current, chosen] + rho_local * tau0
                unvisited.remove(chosen)
                current = chosen
            # Evaluate
            max_len, routes = split_permutation(perm)
            if max_len < best_global_max - 1e-12:
                best_global_max = max_len
                best_global_routes = routes
                best_global_perm = perm[:]
                report_best_vrp(routes)
        # Global pheromone update
        tau *= (1 - rho_global)
        if best_global_perm is not None:
            seq = [0] + best_global_perm + [0]
            deposit = 1.0 / (best_global_max + 1e-10)
            for k in range(len(seq)-1):
                i, j = seq[k], seq[k+1]
                tau[i, j] += rho_global * deposit

    if best_global_routes is None:
        # Fallback: sequential assignment
        routes = [[0, 0] for _ in range(truck_count)]
        idx = 0
        for cust in range(1, n):
            routes[idx].insert(-1, cust)
            idx = (idx + 1) % truck_count
        best_global_routes = routes
    return best_global_routes