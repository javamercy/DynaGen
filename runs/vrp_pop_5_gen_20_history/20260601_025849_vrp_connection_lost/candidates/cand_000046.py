import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    customers = list(range(1, n))
    m = n - 1

    if m == 0:
        return [[0, 0] for _ in range(truck_count)]

    # 1. Build giant tour using nearest neighbor heuristic
    tour = [0]
    current = 0
    unvisited = set(customers)
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    tour.append(0)
    seq = tour[1:-1]  # sequence of customers in order

    # 2. Precompute segment costs: seg_cost[i][j] for i<=j
    seg_cost = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            d = dist[0][seq[i]] + dist[seq[j]][0]
            for k in range(i, j):
                d += dist[seq[k]][seq[k+1]]
            seg_cost[i][j] = d

    # 3. Dynamic programming for min-max split into exactly k non-empty routes
    k_max = min(truck_count, m)
    INF = float('inf')
    dp = [[INF] * (k_max + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    # parent tracking for reconstruction
    parent = [[-1] * (k_max + 1) for _ in range(m + 1)]

    for p in range(1, m + 1):
        for k in range(1, min(k_max, p) + 1):
            best = INF
            best_t = -1
            for t in range(0, p):
                if dp[t][k-1] == INF:
                    continue
                candidate = max(dp[t][k-1], seg_cost[t][p-1])
                if candidate < best:
                    best = candidate
                    best_t = t
            dp[p][k] = best
            parent[p][k] = best_t

    # Reconstruct partition
    partition = []
    p = m
    k = k_max
    while k > 0:
        t = parent[p][k]
        partition.append((t, p-1))  # segment covers customers seq[t:p] (inclusive)
        p = t
        k -= 1
    partition.reverse()

    # Build routes from partition
    routes = []
    for start, end in partition:
        route = [0] + seq[start:end+1] + [0]
        routes.append(route)
    # Add empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])

    # Helper functions
    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    # Report initial solution
    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)

    # 4. Improvement: relocate, swap, 2-opt
    max_iter = n * truck_count
    for _ in range(max_iter):
        improved = False
        current_routes = [list(r) for r in routes]
        current_max = max_dist(current_routes)

        # Relocate
        for r_idx in range(truck_count):
            if len(current_routes[r_idx]) <= 3:
                continue
            for pos in range(1, len(current_routes[r_idx]) - 1):
                cust = current_routes[r_idx][pos]
                for o_idx in range(truck_count):
                    if o_idx == r_idx:
                        continue
                    for o_pos in range(1, len(current_routes[o_idx])):
                        new_self = current_routes[r_idx][:pos] + current_routes[r_idx][pos+1:]
                        new_other = current_routes[o_idx][:o_pos] + [cust] + current_routes[o_idx][o_pos:]
                        new_routes = [list(r) for r in current_routes]
                        new_routes[r_idx] = new_self
                        new_routes[o_idx] = new_other
                        new_max = max_dist(new_routes)
                        if new_max < best_max - 1e-12:
                            best_max = new_max
                            best_routes = new_routes
                            routes = new_routes
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Swap
        for r_idx in range(truck_count):
            if len(current_routes[r_idx]) <= 3:
                continue
            for pos1 in range(1, len(current_routes[r_idx]) - 1):
                for o_idx in range(r_idx + 1, truck_count):
                    if len(current_routes[o_idx]) <= 3:
                        continue
                    for pos2 in range(1, len(current_routes[o_idx]) - 1):
                        cust1 = current_routes[r_idx][pos1]
                        cust2 = current_routes[o_idx][pos2]
                        new_route1 = list(current_routes[r_idx])
                        new_route2 = list(current_routes[o_idx])
                        new_route1[pos1] = cust2
                        new_route2[pos2] = cust1
                        new_routes = [list(r) for r in current_routes]
                        new_routes[r_idx] = new_route1
                        new_routes[o_idx] = new_route2
                        new_max = max_dist(new_routes)
                        if new_max < best_max - 1e-12:
                            best_max = new_max
                            best_routes = new_routes
                            routes = new_routes
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # 2-opt
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i+1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_routes = [list(r) for r in current_routes]
                    new_routes[r_idx] = new_route
                    new_max = max_dist(new_routes)
                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        best_routes = new_routes
                        routes = new_routes
                        improved = True
                        report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    # Final routes: ensure exactly truck_count and format
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append(route)
    return final_routes