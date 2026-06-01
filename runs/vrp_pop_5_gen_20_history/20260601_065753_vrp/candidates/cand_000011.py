import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    perm = customers[:]

    def decode(perm):
        n_cust = len(perm)
        if n_cust == 0:
            return [[0, 0]] * truck_count
        cum = [0.0]
        for idx in range(n_cust - 1):
            cum.append(cum[-1] + distance_matrix[perm[idx], perm[idx + 1]])
        depot_dist = distance_matrix[0, :]

        def segment_cost(i, j):
            # segment from index i to j inclusive
            internal = cum[j] - cum[i]
            return internal + depot_dist[perm[i]] + depot_dist[perm[j]]

        low = 0.0
        high = sum(depot_dist) * 2
        while high - low > 1e-6:
            mid = (low + high) / 2
            dp = [float('inf')] * (n_cust + 1)
            dp[0] = 0
            for i in range(1, n_cust + 1):
                for j in range(i - 1, -1, -1):
                    cost = segment_cost(j, i - 1)
                    if cost <= mid and dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
            if dp[n_cust] <= truck_count:
                high = mid
            else:
                low = mid
        target = high
        dp = [float('inf')] * (n_cust + 1)
        prev = [-1] * (n_cust + 1)
        dp[0] = 0
        for i in range(1, n_cust + 1):
            for j in range(i - 1, -1, -1):
                cost = segment_cost(j, i - 1)
                if cost <= target and dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j
        routes = []
        i = n_cust
        while i > 0:
            j = prev[i]
            seg = perm[j:i]
            routes.append([0] + seg + [0])
            i = j
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def evaluate(routes):
        max_len = 0.0
        for route in routes:
            total = 0.0
            for k in range(len(route) - 1):
                total += distance_matrix[route[k], route[k + 1]]
            if total > max_len:
                max_len = total
        return max_len

    def report_best_vrp(routes):
        pass

    routes = decode(perm)
    best_max = evaluate(routes)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    improved = True
    for _ in range(2 * n):
        if not improved:
            break
        improved = False
        # Swap neighborhood
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                new_perm = perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_routes = decode(new_perm)
                new_max = evaluate(new_routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in new_routes]
                    perm = new_perm
                    improved = True
                    report_best_vrp(best_routes)
                    break
            if improved:
                break
        if improved:
            continue
        # Insert neighborhood
        for i in range(len(perm)):
            cust = perm[i]
            for j in range(len(perm)):
                if i == j:
                    continue
                new_perm = perm[:i] + perm[i+1:]
                new_perm.insert(j, cust)
                new_routes = decode(new_perm)
                new_max = evaluate(new_routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in new_routes]
                    perm = new_perm
                    improved = True
                    report_best_vrp(best_routes)
                    break
            if improved:
                break
    return best_routes