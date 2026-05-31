import numpy as np

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
            routes[i - 1] = [0, i, 0]
        return routes

    # Helper functions
    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i + 1]]
        return d

    # Build initial permutation via nearest neighbor from depot
    perm = []
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: (distance_matrix[current, x], x))
        perm.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    # 2-opt improvement on the tour (including depot)
    def two_opt_tour(tour):
        # tour is list of customers in order (without depot)
        # convert to full route [0] + tour + [0]
        best = [0] + tour[:] + [0]
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
        return best[1:-1]  # return customer list
    perm = two_opt_tour(perm)

    # Precompute distances for DP
    m = len(perm)
    start_to_depot = np.array([distance_matrix[0, c] for c in perm])
    end_to_depot = np.array([distance_matrix[c, 0] for c in perm])
    cum_inter = np.zeros(m + 1)
    for i in range(1, m):
        cum_inter[i] = cum_inter[i-1] + distance_matrix[perm[i-1], perm[i]]
    cum_inter[m] = cum_inter[m-1]  # not used
    def seg_dist(l, r):
        if l > r:
            return 0.0
        return start_to_depot[l] + (cum_inter[r] - cum_inter[l]) + end_to_depot[r]

    # DP for optimal split
    def split_perm(perm):
        m = len(perm)
        K = truck_count
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
        # Reconstruct routes
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
        return routes

    # Intra-route 2-opt
    def intra_opt(route):
        if len(route) <= 3:
            return route
        best = route[:]
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
        return best

    # Initial solution
    routes = split_perm(perm)
    for idx in range(truck_count):
        routes[idx] = intra_opt(routes[idx])
    best_routes = [r[:] for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)

    # Local search on permutation
    max_iter = customer_count * truck_count
    for _ in range(max_iter):
        improved = False
        for i in range(m):
            for j in range(i + 1, m):
                new_perm = perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_routes = split_perm(new_perm)
                for idx in range(truck_count):
                    new_routes[idx] = intra_opt(new_routes[idx])
                new_max = max(route_dist(r) for r in new_routes)
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
    return best_routes