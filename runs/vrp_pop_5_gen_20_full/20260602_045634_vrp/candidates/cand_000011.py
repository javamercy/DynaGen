import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    num_customers = n - 1
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_dist(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    # TSP construction
    def nearest_neighbor_tsp(start_cust):
        unvisited = set(customers)
        tour = [0, start_cust]
        unvisited.remove(start_cust)
        current = start_cust
        while unvisited:
            next_c = min(unvisited, key=lambda c: (distance_matrix[current][c], c))
            tour.append(next_c)
            unvisited.remove(next_c)
            current = next_c
        tour.append(0)
        return tour

    def two_opt_tsp(tour):
        improved = True
        max_iter = 100
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            for i in range(1, len(tour)-2):
                for j in range(i+1, len(tour)-1):
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    new_dist = route_dist(new_tour)
                    if new_dist < route_dist(tour):
                        tour = new_tour
                        improved = True
        return tour

    # Build giant TSP tour
    depot_dists = [distance_matrix[0][c] for c in customers]
    sorted_cust = sorted(customers, key=lambda c: (depot_dists[customers.index(c)], c), reverse=True)
    num_starts = min(5, num_customers)
    best_tour = None
    best_dist = float('inf')
    for start in sorted_cust[:num_starts]:
        tour = nearest_neighbor_tsp(start)
        tour = two_opt_tsp(tour)
        d = route_dist(tour)
        if d < best_dist:
            best_dist = d
            best_tour = tour
    tour = best_tour
    report_best_vrp([tour])  # report initial giant tour? Not exactly VRP routes, but we'll report later

    # Split tour into routes using DP
    cust_seq = tour[1:-1]
    m = len(cust_seq)
    direct_from = [distance_matrix[0][c] for c in cust_seq]
    direct_to = [distance_matrix[c][0] for c in cust_seq]
    interior_prefix = [0.0]
    for i in range(m-1):
        interior_prefix.append(interior_prefix[-1] + distance_matrix[cust_seq[i]][cust_seq[i+1]])

    def segment_cost(l, r):
        return direct_from[l] + (interior_prefix[r] - interior_prefix[l]) + direct_to[r]

    INF = float('inf')
    dp = [[INF]*(truck_count+1) for _ in range(m+1)]
    split = [[0]*(truck_count+1) for _ in range(m+1)]
    dp[0][0] = 0.0
    for k in range(1, truck_count+1):
        for i in range(k, m+1):
            for j in range(k-1, i):
                cost = segment_cost(j, i-1)
                candidate = max(dp[j][k-1], cost)
                if candidate < dp[i][k]:
                    dp[i][k] = candidate
                    split[i][k] = j
    # Reconstruct routes
    routes = []
    i = m
    k = truck_count
    while k > 0:
        j = split[i][k]
        seg = cust_seq[j:i]
        route = [0] + seg + [0]
        routes.append(route)
        i = j
        k -= 1
    routes.reverse()
    while len(routes) < truck_count:
        routes.append([0,0])
    report_best_vrp(routes)

    # Intra-route 2-opt on each route
    for idx in range(len(routes)):
        route = routes[idx]
        if len(route) <= 2:
            continue
        improved = True
        max_iter = 100
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route):
                        route = new_route
                        improved = True
        routes[idx] = route
    report_best_vrp(routes)

    # Inter-route improvement
    current_max = max(route_dist(r) for r in routes)
    improved = True
    max_iter = num_customers * truck_count
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        dists = [route_dist(r) for r in routes]
        max_dist = max(dists)
        if max_dist == 0:
            break
        longest_idx = dists.index(max_dist)
        min_dist = min(d for d in dists)
        shortest_idx = dists.index(min_dist)
        best_new_max = max_dist
        best_move = None
        # Relocate from longest to shortest
        for cust in routes[longest_idx][1:-1]:
            new_long = [0] + [c for c in routes[longest_idx][1:-1] if c != cust] + [0]
            dist_long = route_dist(new_long)
            best_insert_dist = float('inf')
            best_pos = None
            for pos in range(1, len(routes[shortest_idx])):
                new_short = routes[shortest_idx][:pos] + [cust] + routes[shortest_idx][pos:]
                dist_short = route_dist(new_short)
                if dist_short < best_insert_dist:
                    best_insert_dist = dist_short
                    best_pos = pos
            new_max = max(dist_long, best_insert_dist, max(d for i,d in enumerate(dists) if i not in [longest_idx, shortest_idx]))
            if new_max < best_new_max:
                best_new_max = new_max
                best_move = ('relocate', longest_idx, shortest_idx, cust, best_pos)
        # Swap between longest and shortest (if both have at least one customer)
        if len(routes[longest_idx]) > 2 and len(routes[shortest_idx]) > 2:
            for cust_long in routes[longest_idx][1:-1]:
                for cust_short in routes[shortest_idx][1:-1]:
                    new_long = [0] + [c if c != cust_long else cust_short for c in routes[longest_idx][1:-1]] + [0]
                    new_short = [0] + [c if c != cust_short else cust_long for c in routes[shortest_idx][1:-1]] + [0]
                    dist_long = route_dist(new_long)
                    dist_short = route_dist(new_short)
                    new_max = max(dist_long, dist_short, max(d for i,d in enumerate(dists) if i not in [longest_idx, shortest_idx]))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = ('swap', longest_idx, shortest_idx, cust_long, cust_short)
        if best_move is not None:
            if best_move[0] == 'relocate':
                _, li, si, cust, pos = best_move
                new_long = [0] + [c for c in routes[li][1:-1] if c != cust] + [0]
                new_short = routes[si][:pos] + [cust] + routes[si][pos:]
                routes[li] = new_long
                routes[si] = new_short
            else:
                _, li, si, cl, cs = best_move
                new_long = [0] + [c if c != cl else cs for c in routes[li][1:-1]] + [0]
                new_short = [0] + [c if c != cs else cl for c in routes[si][1:-1]] + [0]
                routes[li] = new_long
                routes[si] = new_short
            current_max = max(route_dist(r) for r in routes)
            report_best_vrp(routes)
            improved = True
    # Ensure exactly truck_count routes (should already be)
    return routes