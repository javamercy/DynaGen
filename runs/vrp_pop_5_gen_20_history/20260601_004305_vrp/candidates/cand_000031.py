import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        for _ in range(truck_count - m):
            routes.append([0, 0])
        return routes

    # Set fixed seed for deterministic randomness
    random.seed(0)

    # --- Two-phase construction: TSP tour + DP split (from TwoPhase2Opt) ---
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current, v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    # Precompute segment distances
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0, tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r-2], tour[r-1]]
            if r == l + 1:
                route_dist = distance_matrix[0, tour[l]] + distance_matrix[tour[l], 0]
            else:
                route_dist = acc + distance_matrix[tour[r-1], 0]
            seg_dist[l][r] = route_dist

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
        total = 0
        for a in range(len(route) - 1):
            total += distance_matrix[route[a], route[a + 1]]
        return total

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def compute_max():
        maxd = 0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    best_max = compute_max()
    best_routes = [list(r) for r in routes]
    total_best = total_dist(best_routes)
    report_best_vrp(best_routes)

    # --- Simulated Annealing ---
    current_max = best_max
    current_total = total_dist(routes)
    T = current_max * 0.1
    max_iter = min(10000, n * n * truck_count)
    cooling_factor = 0.999

    for it in range(max_iter):
        # choose move type (0 = intra 2-opt, 1 = inter relocate)
        if random.random() < 0.5:
            # intra-route 2-opt
            feasible_routes = [idx for idx, r in enumerate(routes) if len(r) > 3]
            if not feasible_routes:
                continue
            ri = random.choice(feasible_routes)
            route = routes[ri]
            L = len(route)
            i = random.randint(1, L - 3)
            j = random.randint(i + 1, L - 2)
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            # compute new max and total
            old_max_route = route_dist(route)
            new_max_route = route_dist(new_route)
            # update distances for affected route
            new_max = current_max
            new_total = current_total - old_max_route + new_max_route
            # compute new overall max
            if new_max_route > new_max:
                new_max = new_max_route
            elif old_max_route == current_max and new_max_route < current_max:
                # might need to recompute max from scratch
                new_max = max(route_dist(r) if idx != ri else new_max_route for idx, r in enumerate(routes))
            # else unchanged
            # acceptance
            delta = new_max - current_max
            if delta < 0 or math.exp(-delta / T) > random.random():
                # accept move
                routes[ri] = new_route
                current_max = new_max
                current_total = new_total
                if new_max < best_max or (new_max == best_max and new_total < total_best):
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    total_best = new_total
                    report_best_vrp(routes)
        else:
            # inter-route relocate
            feasible_routes = [idx for idx, r in enumerate(routes) if len(r) > 2]
            if len(feasible_routes) < 2:
                continue
            src_idx = random.choice(feasible_routes)
            dst_idx = random.choice([idx for idx in range(truck_count) if idx != src_idx])
            src_route = routes[src_idx]
            L_src = len(src_route)
            # choose customer position (1 to L_src-2)
            cust_pos = random.randint(1, L_src - 2)
            cust = src_route[cust_pos]
            new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
            # choose insertion position in dst route (1 to len(dst_route)-1)
            dst_route = routes[dst_idx]
            L_dst = len(dst_route)
            ins_pos = random.randint(1, L_dst - 1)
            new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
            # compute new distances
            old_src_dist = route_dist(src_route)
            new_src_dist = route_dist(new_src)
            old_dst_dist = route_dist(dst_route)
            new_dst_dist = route_dist(new_dst)
            new_total = current_total - old_src_dist - old_dst_dist + new_src_dist + new_dst_dist
            # compute new max
            new_max = current_max
            if new_src_dist > new_max:
                new_max = new_src_dist
            if new_dst_dist > new_max:
                new_max = new_dst_dist
            if old_src_dist == current_max and new_src_dist < current_max:
                # recompute max
                new_max = max(route_dist(r) for idx, r in enumerate(routes) if idx not in (src_idx, dst_idx))
                new_max = max(new_max, new_src_dist, new_dst_dist)
            if old_dst_dist == current_max and new_dst_dist < current_max:
                new_max = max(route_dist(r) for idx, r in enumerate(routes) if idx not in (src_idx, dst_idx))
                new_max = max(new_max, new_src_dist, new_dst_dist)
            delta = new_max - current_max
            if delta < 0 or math.exp(-delta / T) > random.random():
                routes[src_idx] = new_src
                routes[dst_idx] = new_dst
                current_max = new_max
                current_total = new_total
                if new_max < best_max or (new_max == best_max and new_total < total_best):
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    total_best = new_total
                    report_best_vrp(routes)

        # cooling
        T = max(T * cooling_factor, 1e-12)

    return best_routes