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

    # --- Initial solution: TSP tour + DP split (minimax) ---
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

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    current = copy_routes(routes)
    current_max = compute_max(current)
    best = copy_routes(current)
    best_max = current_max
    report_best_vrp(best)

    # --- ALNS parameters ---
    max_iter = 2000
    initial_temp = 0.2 * current_max
    final_temp = 0.001
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max_iter)
    temp = initial_temp
    n_cust = m

    for it in range(max_iter):
        # Dynamic removal fraction: linearly from 30% to 5%
        removal_frac = 0.3 - (0.3 - 0.05) * it / max_iter
        q = max(1, int(n_cust * removal_frac))
        removed = []
        new_routes = copy_routes(current)
        all_cust = [c for route in new_routes for c in route if c != 0]
        random.shuffle(all_cust)
        for c in all_cust[:q]:
            for route in new_routes:
                if c in route:
                    route.remove(c)
                    removed.append(c)
                    break

        # Repair: greedy insertion minimizing resulting max distance
        random.shuffle(removed)
        for c in removed:
            best_inc = math.inf
            best_ri = -1
            best_pos = -1
            for ri, route in enumerate(new_routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [c] + route[pos:]
                    new_dist = route_dist(new_route)
                    other_dists = [route_dist(r) for ri2, r in enumerate(new_routes) if ri2 != ri]
                    new_max = max(new_dist, *other_dists)
                    if new_max < best_inc or (new_max == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                        best_inc = new_max
                        best_ri = ri
                        best_pos = pos
            new_routes[best_ri].insert(best_pos, c)

        # Evaluate new solution
        new_max = compute_max(new_routes)
        delta = new_max - current_max

        # Accept using simulated annealing
        accepted = False
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_routes
            current_max = new_max
            accepted = True
            if new_max < best_max:
                best = copy_routes(new_routes)
                best_max = new_max
                report_best_vrp(best)

        # Cool temperature
        temp *= cooling_rate

    return best