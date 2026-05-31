import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = []
        for i in customers:
            routes.append([0, i, 0])
        for _ in range(truck_count - len(customers)):
            routes.append([0, 0])
        return routes

    # Step 1: Build nearest neighbor giant tour (starting from 0, end at 0)
    unvisited = set(customers)
    tour = [0]
    current = 0
    while unvisited:
        # Find nearest unvisited customer, tie-break by smaller index
        best_cust = None
        best_dist = float('inf')
        for c in unvisited:
            d = distance_matrix[current, c]
            if d < best_dist or (d == best_dist and (best_cust is None or c < best_cust)):
                best_dist = d
                best_cust = c
        tour.append(best_cust)
        unvisited.remove(best_cust)
        current = best_cust
    tour.append(0)
    # permutation of customers (order of first visit)
    perm = tour[1:-1]  # list of customers in order of visitation
    
    # Step 2: Split permutation into truck_count routes via DP minimizing max route length
    m = len(perm)
    # Precompute segment costs: seg_cost[i][j] = cost of a route covering perm[i:j] (i inclusive, j exclusive)
    seg_cost = [[0.0]*(m+1) for _ in range(m)]
    for i in range(m):
        # segment from i to i: empty route (cost 0? but not used)
        for j in range(i+1, m+1):
            # route: 0 -> perm[i] -> ... -> perm[j-1] -> 0
            cost = distance_matrix[0, perm[i]]
            for k in range(i, j-1):
                cost += distance_matrix[perm[k], perm[k+1]]
            cost += distance_matrix[perm[j-1], 0]
            seg_cost[i][j] = cost
    # DP: dp[k][j] = min max distance for first j customers (perm[0:j]) using exactly k routes
    INF = 1e100
    dp = [[INF]*(m+1) for _ in range(truck_count+1)]
    split = [[-1]*(m+1) for _ in range(truck_count+1)]
    dp[0][0] = 0.0
    for k in range(1, truck_count+1):
        for j in range(k, m+1):  # need at least k customers
            # try last segment from i to j
            for i in range(k-1, j):
                prev_max = dp[k-1][i]
                seg = seg_cost[i][j]
                cand = max(prev_max, seg)
                if cand < dp[k][j] or (cand == dp[k][j] and (seg < seg_cost[split[k][j]][j] if split[k][j]!=-1 else False)):
                    dp[k][j] = cand
                    split[k][j] = i
    # Reconstruct routes from split
    routes = []
    j = m
    for k in range(truck_count, 0, -1):
        i = split[k][j]
        # route covers perm[i:j]
        if i == j:
            routes.append([0,0])
        else:
            route = [0] + perm[i:j] + [0]
            routes.append(route)
        j = i
    # Routes are in reverse order (last split first), reverse to have first route first (any order works)
    routes = routes[::-1]
    # Ensure exactly truck_count routes (if some split yields empty, we already added [0,0])
    while len(routes) < truck_count:
        routes.append([0,0])
    
    # Helper functions
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def total_max():
        return max(route_distance(r) for r in routes)

    current_max = total_max()
    report_best_vrp(routes)

    # Step 3: Intra-route 2-opt on each route
    for r_idx in range(truck_count):
        route = routes[r_idx]
        if len(route) <= 3:
            continue
        improved = True
        max_iter = len(route) * 10
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_distance(new_route)
                    old_len = route_distance(route)
                    if new_len < old_len - 1e-12:
                        route[:] = new_route
                        new_max = total_max()
                        if new_max < current_max - 1e-12:
                            current_max = new_max
                            report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
    
    # Step 4: Permutation-level local search: try swapping each pair of customers in the perm and re-split
    # We'll work on a copy of perm
    best_perm = perm[:]
    best_routes = [r[:] for r in routes]
    best_max = current_max
    # iterate over all pairs once
    for i in range(m):
        for j in range(i+1, m):
            new_perm = best_perm[:]
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            # Re-split new_perm
            # compute segment costs for new_perm
            new_seg_cost = [[0.0]*(m+1) for _ in range(m)]
            for ii in range(m):
                for jj in range(ii+1, m+1):
                    cost = distance_matrix[0, new_perm[ii]]
                    for k in range(ii, jj-1):
                        cost += distance_matrix[new_perm[k], new_perm[k+1]]
                    cost += distance_matrix[new_perm[jj-1], 0]
                    new_seg_cost[ii][jj] = cost
            # DP
            dp2 = [[INF]*(m+1) for _ in range(truck_count+1)]
            split2 = [[-1]*(m+1) for _ in range(truck_count+1)]
            dp2[0][0] = 0.0
            for k in range(1, truck_count+1):
                for jj in range(k, m+1):
                    for ii in range(k-1, jj):
                        prev = dp2[k-1][ii]
                        seg = new_seg_cost[ii][jj]
                        cand = max(prev, seg)
                        if cand < dp2[k][jj] or (cand == dp2[k][jj] and (seg < new_seg_cost[split2[k][jj]][jj] if split2[k][jj]!=-1 else False)):
                            dp2[k][jj] = cand
                            split2[k][jj] = ii
            # reconstruct
            new_routes = []
            jj = m
            for k in range(truck_count, 0, -1):
                ii = split2[k][jj]
                if ii == jj:
                    new_routes.append([0,0])
                else:
                    route = [0] + new_perm[ii:jj] + [0]
                    new_routes.append(route)
                jj = ii
            new_routes = new_routes[::-1]
            while len(new_routes) < truck_count:
                new_routes.append([0,0])
            new_max = max(route_distance(r) for r in new_routes)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_perm = new_perm[:]
                best_routes = [r[:] for r in new_routes]
                current_max = best_max
                report_best_vrp(best_routes)
    
    # Return best found routes
    return best_routes