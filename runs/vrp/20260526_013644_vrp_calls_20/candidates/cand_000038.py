import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # If enough trucks, assign each customer to separate route
    if truck_count >= n - 1:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    customers = list(range(1, n))
    m = len(customers)

    def tour_distance(perm):
        d = distance_matrix[0, perm[0]]
        for i in range(len(perm)-1):
            d += distance_matrix[perm[i], perm[i+1]]
        d += distance_matrix[perm[-1], 0]
        return d

    def evaluate_max(perm):
        # DP to split perm into truck_count routes minimizing max route distance
        seg = [[0.0]*m for _ in range(m)]
        for i in range(m):
            cum = distance_matrix[0, perm[i]]
            for j in range(i, m):
                if j > i:
                    cum += distance_matrix[perm[j-1], perm[j]]
                seg[i][j] = cum + distance_matrix[perm[j], 0]
        INF = float('inf')
        dp = [[INF]*(truck_count+1) for _ in range(m+1)]
        dp[0][0] = 0
        prev = [[None]*(truck_count+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for k in range(1, min(i, truck_count)+1):
                for j in range(1, i+1):
                    cost = seg[j-1][i-1]
                    if dp[j-1][k-1] != INF:
                        cand = max(dp[j-1][k-1], cost)
                        if cand < dp[i][k]:
                            dp[i][k] = cand
                            prev[i][k] = j-1
        best = dp[m][truck_count]
        routes = []
        if best != INF:
            i = m
            k = truck_count
            segments = []
            while k > 0:
                j = prev[i][k]
                segments.append((j, i-1))
                i = j
                k -= 1
            segments.reverse()
            for (start, end) in segments:
                if start > end:
                    routes.append([0, 0])
                else:
                    route = [0] + perm[start:end+1] + [0]
                    routes.append(route)
            while len(routes) < truck_count:
                routes.append([0, 0])
        return best, routes

    def improve_tour(perm, max_iter=50):
        best_perm = perm[:]
        best_max, best_routes = evaluate_max(perm)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for i in range(len(best_perm)):
                for j in range(i+2, len(best_perm)):
                    new_perm = best_perm[:i] + best_perm[i:j+1][::-1] + best_perm[j+1:]
                    cand_max, cand_routes = evaluate_max(new_perm)
                    if cand_max < best_max:
                        best_max = cand_max
                        best_perm = new_perm
                        best_routes = cand_routes
                        improved = True
        return best_perm, best_max, best_routes

    # Generate initial permutations
    # 1. nearest neighbor from depot
    perm1 = []
    remaining = set(customers)
    current = 0
    while remaining:
        next_cust = min(remaining, key=lambda c: (distance_matrix[current, c], c))
        perm1.append(next_cust)
        remaining.remove(next_cust)
        current = next_cust
    # 2. reverse of perm1
    perm2 = perm1[::-1]
    # 3. natural order
    perm3 = sorted(customers)
    # 4. reverse natural
    perm4 = perm3[::-1]

    best_global_max = float('inf')
    best_global_routes = None

    for perm in [perm1, perm2, perm3, perm4]:
        improved_perm, max_dist, routes = improve_tour(perm, max_iter=50)
        if max_dist < best_global_max:
            best_global_max = max_dist
            best_global_routes = routes
            try:
                report_best_vrp(best_global_routes)
            except NameError:
                pass

    # Fallback if best_global_routes is None (should not happen)
    if best_global_routes is None:
        # simple construction
        routes = [[0, i, 0] for i in range(1, min(n, truck_count+1))]
        while len(routes) < truck_count:
            routes.append([0, 0])
        best_global_routes = routes

    return best_global_routes