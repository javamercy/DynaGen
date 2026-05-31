import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    n_cust = len(customers)
    if truck_count >= n_cust:
        routes = [[0, 0] for _ in range(truck_count)]
        for i, cust in enumerate(customers):
            routes[i] = [0, cust, 0]
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
    tour.append(0)
    giant_tour = tour[1:-1]
    n_c = len(giant_tour)

    # 2. Precompute segment distances
    seg = [[0.0] * n_c for _ in range(n_c)]
    for i in range(n_c):
        d = distance_matrix[0, giant_tour[i]]
        seg[i][i] = d + distance_matrix[giant_tour[i], 0]
        for j in range(i+1, n_c):
            d += distance_matrix[giant_tour[j-1], giant_tour[j]]
            seg[i][j] = d + distance_matrix[giant_tour[j], 0]

    # 3. DP split into exactly K non-empty routes (K = min(truck_count, n_c))
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

    def route_dist(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for a, b in zip(route[:-1], route[1:]):
            d += distance_matrix[a, b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def two_opt(route):
        best = route[:]
        improved = True
        max_passes = 5  # limit number of passes
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_dist(new_route) < route_dist(best):
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)

    # 4. Ruin and recreate improvement (reduced iterations to n)
    max_iter = n_c
    for _ in range(max_iter):
        old_max = max_dist(routes)
        # Select customers with highest contribution
        contribs = []
        for ridx, route in enumerate(routes):
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                c = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos+1]]
                contribs.append((-c, cust, ridx, pos))
        contribs.sort()
        ruin_size = max(1, n_c // 10)
        to_ruin = [t[1] for t in contribs[:ruin_size]]

        saved_routes = [r[:] for r in routes]

        for cust in to_ruin:
            for ridx, route in enumerate(routes):
                if cust in route:
                    new_route = [x for x in route if x != cust]
                    if len(new_route) < 2 or new_route[0] != 0:
                        new_route = [0] + new_route
                    if new_route[-1] != 0:
                        new_route = new_route + [0]
                    if len(new_route) == 2:
                        new_route = [0, 0]
                    routes[ridx] = new_route
                    break

        unrouted = set(to_ruin)
        current_routes = [r[:] for r in routes]
        while unrouted:
            best_cust = None
            best_regret = -1.0
            best_insert = (None, None)
            for cust in sorted(unrouted):
                costs = []
                for ri, route in enumerate(current_routes):
                    best_cost = INF
                    best_pos = -1
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        temp_routes = current_routes[:]
                        temp_routes[ri] = new_route
                        new_max = max_dist(temp_routes)
                        if new_max < best_cost:
                            best_cost = new_max
                            best_pos = pos
                    costs.append((best_cost, best_pos, ri))
                costs.sort(key=lambda x: (x[0], x[2]))
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0.0
                if regret > best_regret + 1e-12:
                    best_regret = regret
                    best_cust = cust
                    best_insert = (costs[0][2], costs[0][1])
            ri, pos = best_insert
            route = current_routes[ri]
            if len(route) == 2:
                current_routes[ri] = [0, best_cust, 0]
            else:
                current_routes[ri] = route[:pos] + [best_cust] + route[pos:]
            unrouted.remove(best_cust)

        new_max = max_dist(current_routes)
        if new_max < old_max - 1e-9:
            routes = current_routes
            for idx in range(truck_count):
                routes[idx] = two_opt(routes[idx])
            if new_max < max_dist(best_routes) - 1e-9:
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        else:
            routes = saved_routes

    # 5. Relocate and exchange improvement (reduced iterations to n//2)
    max_iter2 = max(1, n_c // 2)
    for _ in range(max_iter2):
        moved = False
        # Relocate
        for from_route in range(truck_count):
            for to_route in range(truck_count):
                if from_route == to_route or len(routes[from_route]) <= 2:
                    continue
                for cust_pos in range(1, len(routes[from_route])-1):
                    for insert_pos in range(1, len(routes[to_route])):
                        new_routes = [r[:] for r in routes]
                        cust = new_routes[from_route].pop(cust_pos)
                        new_routes[to_route].insert(insert_pos, cust)
                        new_max = max_dist(new_routes)
                        if new_max < max_dist(best_routes) - 1e-9:
                            routes = new_routes
                            for idx in range(truck_count):
                                routes[idx] = two_opt(routes[idx])
                            if max_dist(routes) < max_dist(best_routes) - 1e-9:
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break
            if moved:
                break
        if moved:
            continue
        # Exchange
        for r1, r2 in combinations(range(truck_count), 2):
            if len(routes[r1]) <= 2 or len(routes[r2]) <= 2:
                continue
            for p1 in range(1, len(routes[r1])-1):
                for p2 in range(1, len(routes[r2])-1):
                    new_routes = [r[:] for r in routes]
                    new_routes[r1][p1], new_routes[r2][p2] = new_routes[r2][p2], new_routes[r1][p1]
                    new_max = max_dist(new_routes)
                    if new_max < max_dist(best_routes) - 1e-9:
                        routes = new_routes
                        for idx in range(truck_count):
                            routes[idx] = two_opt(routes[idx])
                        if max_dist(routes) < max_dist(best_routes) - 1e-9:
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break

    return best_routes