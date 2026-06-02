import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    L_total = len(customers)
    best_routes = None
    best_max = float('inf')
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    # Build giant tour via nearest neighbor starting from customer 1
    start_customer = 1
    giant_path = [start_customer]
    remaining = set(customers)
    remaining.remove(start_customer)
    current = start_customer
    while remaining:
        nearest = None
        min_dist = float('inf')
        for c in remaining:
            d = distance_matrix[current][c]
            if d < min_dist or (d == min_dist and c < nearest):
                min_dist = d
                nearest = c
        giant_path.append(nearest)
        remaining.remove(nearest)
        current = nearest
    L = len(giant_path)
    
    # Handle case where L < truck_count
    if L < truck_count:
        routes = []
        idx = 0
        for t in range(truck_count):
            if idx < L:
                route = [0, giant_path[idx], 0]
                idx += 1
            else:
                route = [0, 0]
            routes.append(route)
        best_routes = routes
        best_max = max(route_distance(r) for r in routes)
        try: report_best_vrp(best_routes)
        except: pass
        return best_routes
    
    # DP split to minimize max route distance
    dist0_to = [distance_matrix[0][c] for c in giant_path]
    dist_to0 = [distance_matrix[c][0] for c in giant_path]
    edge = [distance_matrix[giant_path[i]][giant_path[i+1]] for i in range(L-1)]
    pref = [0]*L
    for i in range(1, L):
        pref[i] = pref[i-1] + edge[i-1]
    
    def seg_cost(i, j):
        return dist0_to[i] + (pref[j] - pref[i]) + dist_to0[j]
    
    INF = 1e18
    dp = [[INF]*(L+1) for _ in range(truck_count+1)]
    prev = [[-1]*(L+1) for _ in range(truck_count+1)]
    dp[0][0] = 0
    for k in range(1, truck_count+1):
        for i in range(k, L+1):
            best_val = INF
            best_j = -1
            for j in range(k-1, i):
                if dp[k-1][j] < INF:
                    seg = seg_cost(j, i-1)
                    candidate = max(dp[k-1][j], seg)
                    if candidate < best_val:
                        best_val = candidate
                        best_j = j
            dp[k][i] = best_val
            prev[k][i] = best_j
    
    # Fallback if no DP solution (should not happen)
    if dp[truck_count][L] == INF:
        # Just assign each customer to a separate truck
        routes = []
        for i in range(truck_count):
            if i < L:
                route = [0, giant_path[i], 0]
            else:
                route = [0, 0]
            routes.append(route)
        best_routes = routes
        best_max = max(route_distance(r) for r in routes)
        try: report_best_vrp(best_routes)
        except: pass
        return best_routes
    
    # Reconstruct split points
    split_points = []
    k = truck_count
    i = L
    while k > 0 and i > 0:
        j = prev[k][i]
        split_points.append(j)
        i = j
        k -= 1
    split_points.reverse()
    split_points.append(L)
    
    routes = []
    for r in range(truck_count):
        start_idx = split_points[r]
        end_idx = split_points[r+1]
        rc = giant_path[start_idx:end_idx]
        if not rc:
            route = [0, 0]
        else:
            route = [0] + rc + [0]
        routes.append(route)
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    best_routes = routes
    best_max = max(route_distance(r) for r in routes)
    try: report_best_vrp(best_routes)
    except: pass
    
    # Improvement loop
    max_iter = n * 2
    for _ in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = best_routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    if new < old - 1e-10:
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        other_max = max(route_distance(best_routes[k]) for k in range(truck_count) if k != r_idx)
                        new_max = max(new_dist, other_max)
                        if new_max < best_max - 1e-10:
                            best_routes[r_idx] = new_route
                            best_max = new_max
                            improved = True
                            try: report_best_vrp(best_routes)
                            except: pass
                            break
                if improved:
                    break
            if improved:
                continue
        if improved:
            continue
        # Inter-route relocate from longest route
        max_dist = 0
        max_idx = 0
        for i, r in enumerate(best_routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                max_idx = i
        route_from = best_routes[max_idx]
        if len(route_from) > 2:
            for cust in route_from[1:-1]:
                for r_to_idx in range(truck_count):
                    if r_to_idx == max_idx:
                        continue
                    route_to = best_routes[r_to_idx]
                    for pos in range(1, len(route_to)):
                        new_route_from = [0] + [x for x in route_from[1:-1] if x != cust] + [0]
                        if len(new_route_from) == 1:
                            new_route_from = [0, 0]
                        new_route_to = route_to[:pos] + [cust] + route_to[pos:]
                        new_dist_from = route_distance(new_route_from)
                        new_dist_to = route_distance(new_route_to)
                        other_max = 0.0
                        for k in range(truck_count):
                            if k != max_idx and k != r_to_idx:
                                dk = route_distance(best_routes[k])
                                if dk > other_max:
                                    other_max = dk
                        new_max = max(new_dist_from, new_dist_to, other_max)
                        if new_max < best_max - 1e-10:
                            best_routes[max_idx] = new_route_from
                            best_routes[r_to_idx] = new_route_to
                            best_max = new_max
                            improved = True
                            try: report_best_vrp(best_routes)
                            except: pass
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return best_routes