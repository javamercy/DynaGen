import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Initial solution: TSP tour (nearest neighbor) + DP split (minimax)
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current][v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    # Segment distances for DP
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0][tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r - 2]][tour[r - 1]]
            if r == l + 1:
                seg_dist[l][r] = distance_matrix[0][tour[l]] + distance_matrix[tour[l]][0]
            else:
                seg_dist[l][r] = acc + distance_matrix[tour[r - 1]][0]

    dp = [[math.inf] * (truck_count + 1) for _ in range(m + 1)]
    choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        for t in range(1, min(i, truck_count) + 1):
            best_val = math.inf
            best_j = -1
            for j in range(t - 1, i):
                if dp[j][t - 1] < math.inf:
                    cand = max(dp[j][t - 1], seg_dist[j][i])
                    if cand < best_val or (cand == best_val and j < best_j):
                        best_val = cand
                        best_j = j
            dp[i][t] = best_val
            choice[i][t] = best_j

    routes = []
    i = m
    t = truck_count
    while t > 0:
        j = choice[i][t]
        seg = tour[j:i]
        routes.append([0] + seg + [0])
        i = j
        t -= 1
    routes.reverse()
    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    current = copy_routes(routes)
    current_max = max_route_dist(current)
    best = copy_routes(current)
    best_max = current_max
    report_best_vrp(best)

    # Improvement parameters
    n_cust = m
    max_iter = min(2000, n_cust * 100)
    if max_iter < 10:
        max_iter = 10
    initial_temp = 0.1 * current_max
    final_temp = 0.001
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max_iter)
    temp = initial_temp

    for it in range(max_iter):
        # Destroy: worst removal (deterministic detour cost, tie-break by customer index)
        q = max(1, n_cust // 10)
        detour = {}
        for route in current:
            for p in range(1, len(route)-1):
                c = route[p]
                prev = route[p-1]
                nxt = route[p+1]
                det = distance_matrix[prev][c] + distance_matrix[c][nxt] - distance_matrix[prev][nxt]
                detour[c] = det
        # Sort by descending detour, then by customer index ascending for tie-breaking
        sorted_cust = sorted(detour.items(), key=lambda x: (-x[1], x[0]))
        removed = [c for c, _ in sorted_cust[:q]]
        new_routes = copy_routes(current)
        for c in removed:
            for route in new_routes:
                if c in route:
                    route.remove(c)
                    break
        # Repair: regret-2 insertion
        # Insert customers one by one, selecting the one with max regret
        while removed:
            best_c = -1
            best_regret = -1
            best_ri = -1
            best_pos = -1
            best_max_val = math.inf
            for c in removed:
                first = (math.inf, -1, -1)
                second = (math.inf, -1, -1)
                for ri, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [c] + route[pos:]
                        new_dist = route_dist(new_route)
                        other_dists = [route_dist(r) for ri2, r in enumerate(new_routes) if ri2 != ri]
                        new_max = max(new_dist, *other_dists)
                        # Tie-break: lower new_max, then lower ri, then lower pos
                        if new_max < first[0] or (new_max == first[0] and ri < first[1]) or (new_max == first[0] and ri == first[1] and pos < first[2]):
                            second = first
                            first = (new_max, ri, pos)
                        elif new_max < second[0] or (new_max == second[0] and ri < second[1]) or (new_max == second[0] and ri == second[1] and pos < second[2]):
                            second = (new_max, ri, pos)
                if first[0] == math.inf:
                    continue
                regret = second[0] - first[0]
                # Tie-break: higher regret, then lower c
                if regret > best_regret or (regret == best_regret and c < best_c):
                    best_regret = regret
                    best_c = c
                    best_ri = first[1]
                    best_pos = first[2]
                    best_max_val = first[0]
            if best_c == -1:
                break
            new_routes[best_ri].insert(best_pos, best_c)
            removed.remove(best_c)

        new_max = max_route_dist(new_routes)
        delta = new_max - current_max
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_routes
            current_max = new_max
            if new_max < best_max:
                best = copy_routes(new_routes)
                best_max = new_max
                report_best_vrp(best)
        temp *= cooling_rate

    return best