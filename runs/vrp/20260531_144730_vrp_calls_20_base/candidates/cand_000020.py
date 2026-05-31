import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, 0] for _ in range(truck_count)]
        for idx, cust in enumerate(customers):
            routes[idx] = [0, cust, 0]
        report_best_vrp(routes)
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
    tour.append(0)  # close the tour
    giant_tour = tour[1:-1]  # customers in order
    n_c = len(giant_tour)

    # 2. Precompute segment distances: seg[i][j] = cost of route from depot to giant_tour[i..j] and back
    seg = [[0.0] * n_c for _ in range(n_c)]
    for i in range(n_c):
        d = distance_matrix[0, giant_tour[i]]
        seg[i][i] = d + distance_matrix[giant_tour[i], 0]
        for j in range(i+1, n_c):
            d += distance_matrix[giant_tour[j-1], giant_tour[j]]
            seg[i][j] = d + distance_matrix[giant_tour[j], 0]

    # 3. DP to split into exactly K non-empty routes (K = min(truck_count, n_c))
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

    report_best_vrp(routes)
    return routes