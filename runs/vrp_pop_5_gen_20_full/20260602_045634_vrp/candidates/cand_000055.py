import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Build TSP tour using nearest neighbor from depot
    def build_tsp_tour():
        visited = [False]*n
        visited[0] = True
        tour = [0]
        current = 0
        for _ in range(n-1):
            nearest = None
            nearest_dist = float('inf')
            for v in range(1, n):
                if not visited[v] and distance_matrix[current, v] < nearest_dist:
                    nearest_dist = distance_matrix[current, v]
                    nearest = v
            tour.append(nearest)
            visited[nearest] = True
            current = nearest
        tour.append(0)  # return to depot
        return tour
    
    tour = build_tsp_tour()
    # Customer sequence (excluding depot at ends)
    seq = tour[1:-1]
    m = len(seq)
    
    # Precompute prefix sums of distances along tour (between consecutive customers)
    pref = [0.0] * (m+1)
    for i in range(1, m):
        pref[i] = pref[i-1] + distance_matrix[seq[i-1], seq[i]]
    pref[m] = pref[m-1] + distance_matrix[seq[m-1], 0]  # not used but for completeness
    
    # Precompute segment distances seg_dist[j][i] for segment j..i-1
    INF = float('inf')
    seg_dist = [[0.0]*m for _ in range(m)]  # seg_dist[start][end] for end>start
    for start in range(m):
        for end in range(start, m):
            dist_start = distance_matrix[0, seq[start]]
            dist_end = distance_matrix[seq[end], 0]
            dist_along = pref[end] - pref[start]  # careful: pref[end] sum up to end-1? Let's use pref properly
            # Actually pref[i] = sum_{k=0}^{i-1} dist(seq[k], seq[k+1])
            # So sum from start to end-1 = pref[end] - pref[start]
            seg_dist[start][end] = dist_start + (pref[end] - pref[start]) + dist_end
    
    # DP: dp[k][i] = min max for first i customers (0..i-1) split into k routes
    dp = [[INF]*(m+1) for _ in range(truck_count+1)]
    backtrack = [[-1]*(m+1) for _ in range(truck_count+1)]
    dp[0][0] = 0.0
    for k in range(1, truck_count+1):
        for i in range(0, m+1):
            best = INF
            best_j = -1
            # j is number of customers assigned to first k-1 routes
            for j in range(0, i+1):
                if dp[k-1][j] == INF:
                    continue
                if j == i:
                    seg = 0.0  # empty route
                else:
                    seg = seg_dist[j][i-1]
                candidate = max(dp[k-1][j], seg)
                if candidate < best or (candidate == best and candidate < best):
                    best = candidate
                    best_j = j
            dp[k][i] = best
            backtrack[k][i] = best_j
    
    # Reconstruct routes
    routes = []
    i = m
    k = truck_count
    while k > 0:
        j = backtrack[k][i]
        if j == i:
            route = [0, 0]
        else:
            route = [0] + seq[j:i] + [0]
        routes.append(route)
        i = j
        k -= 1
    routes.reverse()
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    report_best_vrp(best_routes)
    
    # Local search similar to parent but with tie-breaking on total distance
    max_iter = (n - 1) * truck_count * 5
    for _ in range(max_iter):
        improved = False
        # Identify max route
        max_dist = 0
        max_idx = 0
        for i, r in enumerate(routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                max_idx = i
        route_max = routes[max_idx]
        
        # Relocate from max route to another
        if len(route_max) > 2:
            for pos in range(1, len(route_max)-1):
                cust = route_max[pos]
                new_max_route = route_max[:pos] + route_max[pos+1:]
                new_max_dist = route_distance(new_max_route)
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for other_pos in range(1, len(other_route)):
                        new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                        new_other_dist = route_distance(new_other)
                        other_max = max(route_distance(routes[i]) for i in range(truck_count) if i not in (max_idx, other_idx))
                        new_overall_max = max(new_max_dist, new_other_dist, other_max)
                        current_total = sum(route_distance(r) for r in routes)
                        new_total = new_max_dist + new_other_dist + sum(route_distance(routes[i]) for i in range(truck_count) if i not in (max_idx, other_idx))
                        if new_overall_max < best_max or (new_overall_max == best_max and new_total < current_total):
                            routes[max_idx] = new_max_route
                            routes[other_idx] = new_other
                            best_max = new_overall_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if improved:
            continue
        
        # Swap between max route and another
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_route = routes[other_idx]
            if len(route_max) <= 2 or len(other_route) <= 2:
                continue
            for i in range(1, len(route_max)-1):
                for j in range(1, len(other_route)-1):
                    cust_i = route_max[i]
                    cust_j = other_route[j]
                    new_max_route = route_max[:i] + [cust_j] + route_max[i+1:]
                    new_other_route = other_route[:j] + [cust_i] + other_route[j+1:]
                    d1 = route_distance(new_max_route)
                    d2 = route_distance(new_other_route)
                    other_max = max(route_distance(routes[k]) for k in range(truck_count) if k not in (max_idx, other_idx))
                    new_overall_max = max(d1, d2, other_max)
                    current_total = sum(route_distance(r) for r in routes)
                    new_total = d1 + d2 + sum(route_distance(routes[k]) for k in range(truck_count) if k not in (max_idx, other_idx))
                    if new_overall_max < best_max or (new_overall_max == best_max and new_total < current_total):
                        routes[max_idx] = new_max_route
                        routes[other_idx] = new_other_route
                        best_max = new_overall_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # 2-opt on max route
        if len(route_max) > 3:
            for i in range(1, len(route_max)-2):
                for j in range(i+1, len(route_max)-1):
                    new_route = route_max[:i] + route_max[i:j+1][::-1] + route_max[j+1:]
                    new_dist = route_distance(new_route)
                    other_max = max(route_distance(routes[k]) for k in range(truck_count) if k != max_idx)
                    new_overall_max = max(new_dist, other_max)
                    current_total = sum(route_distance(r) for r in routes)
                    new_total = new_dist + sum(route_distance(routes[k]) for k in range(truck_count) if k != max_idx)
                    if new_overall_max < best_max or (new_overall_max == best_max and new_total < current_total):
                        routes[max_idx] = new_route
                        best_max = new_overall_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
        if improved:
            continue
        else:
            break  # no improving move found, exit
    
    return best_routes

# The code ends here. In practice, the function must be defined as above.
# Note: The helper function report_best_vrp is assumed to be provided externally.