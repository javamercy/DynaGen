import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        # each customer gets its own truck
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
    # Add empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])

    # Helper functions
    def route_dist(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for a, b in zip(route[:-1], route[1:]):
            d += distance_matrix[a, b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)

    # 4. Ruin and recreate improvement
    max_iter = n_c * 5
    for _ in range(max_iter):
        old_max = max_dist(routes)
        # Select customers with highest contribution (sum of incident edge distances)
        contribs = []
        for ridx, route in enumerate(routes):
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                c = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos+1]]
                contribs.append((-c, cust, ridx, pos))  # negative for descending
        contribs.sort()
        ruin_size = max(1, n_c // 10)
        to_ruin = [t[1] for t in contribs[:ruin_size]]

        # Save state before ruin
        saved_routes = [r[:] for r in routes]

        # Remove customers from routes
        for cust in to_ruin:
            for ridx, route in enumerate(routes):
                if cust in route:
                    new_route = [x for x in route if x != cust]
                    if len(new_route) == 1:  # should not happen as depot at ends
                        new_route = [0, 0]
                    elif new_route[0] != 0 or new_route[-1] != 0:
                        # ensure depot at ends (in case of odd removals)
                        if new_route[0] != 0:
                            new_route = [0] + new_route
                        if new_route[-1] != 0:
                            new_route = new_route + [0]
                        # if length becomes 2 and is [cust,0] etc, fix
                        if len(new_route) == 2 and new_route[0] != 0:
                            new_route = [0, cust, 0]
                    routes[ridx] = new_route
                    break

        # Regret-2 insertion for removed customers
        unrouted = set(to_ruin)
        current_routes = [r[:] for r in routes]
        while unrouted:
            best_cust = None
            best_regret = -1.0
            best_insert = (None, None)  # (route_idx, pos)
            for cust in sorted(unrouted):  # deterministic tie-breaking by index
                # Evaluate insertion in each route
                costs = []
                for ri, route in enumerate(current_routes):
                    best_cost = INF
                    best_pos = -1
                    # possible insertion positions: after first depot, before last depot
                    for pos in range(1, len(route)):
                        # Simulate insertion
                        new_route = route[:pos] + [cust] + route[pos:]
                        # Compute new max distance quickly (only this route changed)
                        old_d = route_dist(route)
                        new_d = route_dist(new_route)
                        # But max may change only if this route's new distance exceeds old max
                        # However, to be accurate, we need to evaluate all routes? Actually only this route's distance changed, so new max is max(old_max_excluding_this_route, new_route_dist). But we don't have old_max_excluding_this_route easily. For simplicity we compute full max_dist, but that costs O(n). Since n small, it's okay.
                        temp_routes = current_routes[:]
                        temp_routes[ri] = new_route
                        new_max = max_dist(temp_routes)
                        if new_max < best_cost:
                            best_cost = new_max
                            best_pos = pos
                    costs.append((best_cost, best_pos, ri))
                costs.sort(key=lambda x: (x[0], x[2]))  # tie-breaking by route index
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0.0
                if regret > best_regret - 1e-12:
                    best_regret = regret
                    best_cust = cust
                    best_insert = (costs[0][2], costs[0][1])  # route idx, pos
            # Insert best_cust
            ri, pos = best_insert
            route = current_routes[ri]
            if len(route) == 2 and route[0] == 0 and route[1] == 0:
                current_routes[ri] = [0, best_cust, 0]
            else:
                current_routes[ri] = route[:pos] + [best_cust] + route[pos:]
            unrouted.remove(best_cust)

        new_max = max_dist(current_routes)
        if new_max < old_max - 1e-9:
            routes = current_routes
            if new_max < max_dist(best_routes) - 1e-9:
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        else:
            # revert to saved state
            routes = saved_routes

    return best_routes