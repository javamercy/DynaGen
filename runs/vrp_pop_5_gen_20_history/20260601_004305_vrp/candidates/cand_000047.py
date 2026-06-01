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

    random.seed(42)

    def route_dist(route):
        d = 0
        for a in range(len(route)-1):
            d += distance_matrix[route[a], route[a+1]]
        return d

    def compute_max_total(routes):
        maxd = 0
        total = 0
        for r in routes:
            d = route_dist(r)
            total += d
            if d > maxd:
                maxd = d
        return maxd, total

    # Nearest neighbor TSP tour
    def build_tour():
        tour = []
        visited = [False]*n
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
        return tour

    # DP split with tie-breaking by total distance
    def split_tour(tour):
        k = truck_count
        seg_dist = [[0]*(m+1) for _ in range(m)]
        for l in range(m):
            acc = distance_matrix[0, tour[l]]
            for r in range(l+1, m+1):
                if r > l+1:
                    acc += distance_matrix[tour[r-2], tour[r-1]]
                if r == l+1:
                    route_d = distance_matrix[0, tour[l]] + distance_matrix[tour[l], 0]
                else:
                    route_d = acc + distance_matrix[tour[r-1], 0]
                seg_dist[l][r] = route_d

        INF = math.inf
        dp = [[INF]*(k+1) for _ in range(m+1)]
        total_dp = [[0]*(k+1) for _ in range(m+1)]
        choice = [[-1]*(k+1) for _ in range(m+1)]
        dp[0][0] = 0
        total_dp[0][0] = 0
        for i in range(1, m+1):
            for t in range(1, min(i, k)+1):
                best = INF
                best_total = INF
                best_j = -1
                for j in range(t-1, i):
                    if dp[j][t-1] < INF:
                        cand_max = max(dp[j][t-1], seg_dist[j][i])
                        cand_total = total_dp[j][t-1] + seg_dist[j][i]
                        if cand_max < best or (cand_max == best and cand_total < best_total):
                            best = cand_max
                            best_total = cand_total
                            best_j = j
                dp[i][t] = best
                total_dp[i][t] = best_total
                choice[i][t] = best_j

        routes = []
        i = m
        t = k
        while t > 0:
            j = choice[i][t]
            seg = tour[j:i]
            routes.append([0] + seg + [0])
            i = j
            t -= 1
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Perturb tour by random swaps
    def perturb_tour(tour, swaps=5):
        tour = tour[:]
        for _ in range(swaps):
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    best_routes = None
    best_max = math.inf
    best_total = math.inf

    for restart in range(3):
        if restart == 0:
            tour = build_tour()
        else:
            tour = perturb_tour(build_tour(), swaps=5)
        routes = split_tour(tour)
        current_max, current_total = compute_max_total(routes)
        if current_max < best_max or (current_max == best_max and current_total < best_total):
            best_max = current_max
            best_total = current_total
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

        # Adaptive SA parameters
        T = current_max * 0.1
        cooling_factor = 0.999
        max_iter = min(5000, n * truck_count * 10)
        move_types = [0, 1, 2]  # 0: 2-opt, 1: relocate, 2: swap
        move_idx = 0

        for it in range(max_iter):
            move = move_types[move_idx]
            improved = False
            if move == 0:  # intra-route 2-opt
                feasible = [ri for ri, r in enumerate(routes) if len(r) > 3]
                if feasible:
                    ri = random.choice(feasible)
                    route = routes[ri]
                    L = len(route)
                    i = random.randint(1, L-3)
                    j = random.randint(i+1, L-2)
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_max, new_total = compute_max_total(routes[:ri] + [new_route] + routes[ri+1:])
                    delta = new_max - current_max
                    if delta < 0 or (delta == 0 and new_total < current_total) or (delta > 0 and random.random() < math.exp(-delta/T)):
                        routes[ri] = new_route
                        current_max = new_max
                        current_total = new_total
                        if current_max < best_max or (current_max == best_max and current_total < best_total):
                            best_max = current_max
                            best_total = current_total
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        improved = True
            elif move == 1:  # inter-route relocate
                feasible_src = [ri for ri, r in enumerate(routes) if len(r) > 2]
                if len(feasible_src) >= 1:
                    src_idx = random.choice(feasible_src)
                    dst_idx = random.choice([i for i in range(truck_count) if i != src_idx])
                    src_route = routes[src_idx]
                    L_src = len(src_route)
                    cust_pos = random.randint(1, L_src-2)
                    cust = src_route[cust_pos]
                    new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
                    dst_route = routes[dst_idx]
                    ins_pos = random.randint(1, len(dst_route)-1)
                    new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                    new_routes = routes[:]
                    new_routes[src_idx] = new_src
                    new_routes[dst_idx] = new_dst
                    new_max, new_total = compute_max_total(new_routes)
                    delta = new_max - current_max
                    if delta < 0 or (delta == 0 and new_total < current_total) or (delta > 0 and random.random() < math.exp(-delta/T)):
                        routes[src_idx] = new_src
                        routes[dst_idx] = new_dst
                        current_max = new_max
                        current_total = new_total
                        if current_max < best_max or (current_max == best_max and current_total < best_total):
                            best_max = current_max
                            best_total = current_total
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        improved = True
            else:  # move == 2: swap customers between routes
                # pick two customers from different routes
                routes_with_cust = [(ri, pos, cust) for ri, r in enumerate(routes) for pos, cust in enumerate(r) if cust != 0]
                if len(routes_with_cust) >= 2:
                    pair = random.sample(routes_with_cust, 2)
                    (ri1, pos1, cust1), (ri2, pos2, cust2) = pair
                    if ri1 != ri2:
                        # swap
                        new_routes = [list(r) for r in routes]
                        new_routes[ri1][pos1] = cust2
                        new_routes[ri2][pos2] = cust1
                        new_max, new_total = compute_max_total(new_routes)
                        delta = new_max - current_max
                        if delta < 0 or (delta == 0 and new_total < current_total) or (delta > 0 and random.random() < math.exp(-delta/T)):
                            routes = new_routes
                            current_max = new_max
                            current_total = new_total
                            if current_max < best_max or (current_max == best_max and current_total < best_total):
                                best_max = current_max
                                best_total = current_total
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                            improved = True

            # Update move index: reset if improved, else advance
            if improved:
                move_idx = 0
            else:
                move_idx = (move_idx + 1) % len(move_types)

            # Cool temperature
            T = max(T * cooling_factor, 1e-12)

    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes