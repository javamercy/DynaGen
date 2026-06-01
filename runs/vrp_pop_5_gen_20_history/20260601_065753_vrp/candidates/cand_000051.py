import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = len(customers)
    
    def compute_route_length(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def cost(cust_list, start, end):
        """Distance of route covering customers from start to end inclusive."""
        route = [0] + cust_list[start:end+1] + [0]
        return compute_route_length(route)
    
    def split_tour(tour):
        """tour: list of customer indices in order. Return (routes, max_distance)."""
        if m == 0:
            routes = [[0,0] for _ in range(truck_count)]
            return routes, 0.0
        # Precompute segment costs
        seg_cost = [[0.0]*m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                seg_cost[i][j] = cost(tour, i, j)
        # DP: f[i][k] = min max distance for first i customers (0..i-1) with k trucks
        # i from 0..m, k from 0..truck_count
        INF = float('inf')
        f = [[INF]*(truck_count+1) for _ in range(m+1)]
        f[0][0] = 0.0
        # For tie-breaking, also store split points
        # We'll store the best max distance, and if tied, prefer lower total cost
        # We'll store f_total[i][k] for total distance to break ties
        f_total = [[0.0]*(truck_count+1) for _ in range(m+1)]
        split_point = [[-1]*(truck_count+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for k in range(1, min(i, truck_count)+1):
                best_max = INF
                best_total = INF
                best_j = -1
                for j in range(k-1, i):  # last segment from j to i-1
                    # f[j][k-1] must be finite
                    if f[j][k-1] == INF:
                        continue
                    seg = seg_cost[j][i-1]
                    cur_max = max(f[j][k-1], seg)
                    cur_total = f_total[j][k-1] + seg
                    if cur_max < best_max or (cur_max == best_max and cur_total < best_total):
                        best_max = cur_max
                        best_total = cur_total
                        best_j = j
                if best_j != -1:
                    f[i][k] = best_max
                    f_total[i][k] = best_total
                    split_point[i][k] = best_j
        # Backtrack to get segments
        routes = []
        i = m
        k = truck_count
        segments = []
        while i > 0 and k > 0:
            j = split_point[i][k]
            if j == -1:
                # fallback: equal split
                segment_len = i // k
                j = i - segment_len
                if j < 0:
                    j = 0
            segments.append((j, i-1))
            i = j
            k -= 1
        segments.reverse()
        # Build routes from segments
        routes = []
        for (s, e) in segments:
            route = [0] + tour[s:e+1] + [0]
            routes.append(route)
        # Fill remaining routes if fewer than truck_count (should not happen if m >= truck_count-1? But for empty case, handle)
        while len(routes) < truck_count:
            routes.append([0,0])
        max_dist = max(compute_route_length(r) for r in routes)
        return routes, max_dist
    
    # Build initial giant tour using nearest neighbor
    tour = []
    visited = set([0])
    current = 0
    for _ in range(m):
        best_dist = float('inf')
        best_cust = -1
        for cust in customers:
            if cust not in visited:
                d = distance_matrix[current][cust]
                if d < best_dist or (d == best_dist and cust < best_cust):
                    best_dist = d
                    best_cust = cust
        if best_cust == -1:
            break
        tour.append(best_cust)
        visited.add(best_cust)
        current = best_cust
    # If any customers left (shouldn't), add them
    for cust in customers:
        if cust not in visited:
            tour.append(cust)
    
    routes, best_max = split_tour(tour)
    best_routes = [list(r) for r in routes]
    
    def report_best_vrp(routes_to_report):
        nonlocal best_max, best_routes
        current_max = max(compute_route_length(r) for r in routes_to_report)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in routes_to_report]
    
    report_best_vrp(routes)
    
    # VNS: local search with 2-opt on tour
    def apply_2opt(tour, i, j):
        # Reverse segment from i to j-1 (0-indexed)
        new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
        return new_tour
    
    def shake(tour, intensity=1):
        # Randomly swap intensity pairs of customers
        new_tour = tour[:]
        for _ in range(intensity):
            i = random.randrange(m)
            j = random.randrange(m)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour
    
    max_iterations = min(100, (n + truck_count) * 5)
    n_iter = 0
    while n_iter < max_iterations:
        improved = False
        # Neighborhood: 2-opt
        for i in range(m-1):
            for j in range(i+2, m+1):  # j exclusive
                new_tour = apply_2opt(tour, i, j)
                new_routes, new_max = split_tour(new_tour)
                if new_max < best_max - 1e-9:
                    tour = new_tour
                    routes = new_routes
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(routes)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Shake: random swap of 1 to 3 pairs
            shake_intensity = random.randint(1, min(3, m//2))
            tour = shake(tour, shake_intensity)
            # Local search again
            n_iter += 1
            # Check if shake improves (optional, but we'll do reporting in LS)
            new_routes, _ = split_tour(tour)
            report_best_vrp(new_routes)
        else:
            n_iter = 0  # reset counter if improved
    
    return best_routes