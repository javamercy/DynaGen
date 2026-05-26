import numpy as np
import random
import math

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def two_opt(route, dm):
    improved = True
    best_route = route[:]
    best_dist = route_distance(best_route, dm)
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)-1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_distance(new_route, dm)
                if new_dist < best_dist - 1e-12:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route

def split_giant_tour(customers, dm, truck_count):
    m = len(customers)
    if m == 0:
        return [[0,0]] * truck_count, 0.0
    # Precompute segment cost: seg_cost[l][r] for l <= r
    seg_cost = [[0.0]*m for _ in range(m)]
    for l in range(m):
        route = [0, customers[l]]
        for r in range(l, m):
            if r > l:
                route.append(customers[r])
            # compute cost from l to r inclusive
            cost = dm[0, customers[l]] + sum(dm[route[i], route[i+1]] for i in range(len(route)-1)) + dm[customers[r], 0]
            seg_cost[l][r] = cost
    # DP: dp_max[i][k] = min max for first i customers (indices 0..i-1) with k routes
    INF = float('inf')
    dp_max = [[INF]*(truck_count+1) for _ in range(m+1)]
    split = [[-1]*(truck_count+1) for _ in range(m+1)]
    dp_max[0][0] = 0.0
    for i in range(1, m+1):
        for k in range(1, min(i, truck_count)+1):
            best = INF
            best_j = -1
            for j in range(k-1, i):
                if dp_max[j][k-1] < INF:
                    cand = max(dp_max[j][k-1], seg_cost[j][i-1])
                    if cand < best:
                        best = cand
                        best_j = j
            dp_max[i][k] = best
            split[i][k] = best_j
    # Find best k (1..truck_count) with minimal max
    best_max = INF
    best_k = 1
    for k in range(1, truck_count+1):
        if dp_max[m][k] < best_max:
            best_max = dp_max[m][k]
            best_k = k
    # Reconstruct routes
    routes = []
    i = m
    k = best_k
    while k > 0:
        j = split[i][k]
        # segment from j to i-1
        seg_custs = customers[j:i]
        if seg_custs:
            route = [0] + seg_custs + [0]
        else:
            route = [0,0]
        routes.append(route)
        i = j
        k -= 1
    # Reverse because we built from end
    routes.reverse()
    # Pad with empty routes
    while len(routes) < truck_count:
        routes.append([0,0])
    return routes, best_max

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Build giant tour via nearest neighbor (deterministic: smallest index ties)
    unvisited = set(customers)
    tour = []
    current = 0
    while unvisited:
        best = None
        best_dist = float('inf')
        for c in sorted(unvisited):
            d = distance_matrix[current, c]
            if d < best_dist:
                best_dist = d
                best = c
        tour.append(best)
        unvisited.remove(best)
        current = best
    
    # Split tour into routes
    routes, best_max = split_giant_tour(tour, distance_matrix, truck_count)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)
    
    # Apply intra-route 2-opt on all routes
    for idx in range(truck_count):
        if len(routes[idx]) > 3:
            routes[idx] = two_opt(routes[idx], distance_matrix)
    dists = [route_distance(r, distance_matrix) for r in routes]
    curr_max = max(dists)
    if curr_max < best_max - 1e-12:
        best_max = curr_max
        best_routes = [list(r) for r in routes]
        report_best_vrp(best_routes)
    
    # Local search: relocate & swap to reduce max
    max_iter = n * truck_count * 2
    for _ in range(max_iter):
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_dist = max(dists)
        improved = False
        # Relocate from longest route (first by index if tie)
        longest_idx = -1
        longest_dist = -1
        for idx, d in enumerate(dists):
            if d > longest_dist + 1e-12:
                longest_dist = d
                longest_idx = idx
        # Try relocating each customer in longest route to other routes
        if len(routes[longest_idx]) > 2:
            for pos in range(1, len(routes[longest_idx])-1):
                cust = routes[longest_idx][pos]
                new_long_route = routes[longest_idx][:pos] + routes[longest_idx][pos+1:]
                new_long_dist = route_distance(new_long_route, distance_matrix)
                for other_idx in range(truck_count):
                    if other_idx == longest_idx:
                        continue
                    for ins in range(1, len(routes[other_idx])):
                        new_other_route = routes[other_idx][:ins] + [cust] + routes[other_idx][ins:]
                        new_other_dist = route_distance(new_other_route, distance_matrix)
                        new_max = max(new_long_dist, new_other_dist, [dists[i] for i in range(truck_count) if i not in (longest_idx, other_idx)] + [0]) 
                        # above handles empty list
                        if new_max < max_dist - 1e-12:
                            routes[longest_idx] = new_long_route
                            routes[other_idx] = new_other_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            # Try swap between longest and another route
            for other_idx in range(truck_count):
                if other_idx == longest_idx or len(routes[other_idx]) <= 2:
                    continue
                for pos_long in range(1, len(routes[longest_idx])-1):
                    cust_a = routes[longest_idx][pos_long]
                    for pos_other in range(1, len(routes[other_idx])-1):
                        cust_b = routes[other_idx][pos_other]
                        new_long_route = routes[longest_idx][:pos_long] + [cust_b] + routes[longest_idx][pos_long+1:]
                        new_other_route = routes[other_idx][:pos_other] + [cust_a] + routes[other_idx][pos_other+1:]
                        new_long_dist = route_distance(new_long_route, distance_matrix)
                        new_other_dist = route_distance(new_other_route, distance_matrix)
                        new_max = max(new_long_dist, new_other_dist, [dists[i] for i in range(truck_count) if i not in (longest_idx, other_idx)] + [0])
                        if new_max < max_dist - 1e-12:
                            routes[longest_idx] = new_long_route
                            routes[other_idx] = new_other_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
        # Update best
        dists = [route_distance(r, distance_matrix) for r in routes]
        curr_max = max(dists)
        if curr_max < best_max - 1e-12:
            best_max = curr_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    
    # Perturbation: random 2-opt on giant tour, re-split, and improve
    random.seed(0)  # deterministic randomness
    for restart in range(3):
        # Perturb giant tour by random 2-opt
        if len(tour) >= 2:
            i = random.randint(0, len(tour)-1)
            j = random.randint(0, len(tour)-1)
            if i > j:
                i, j = j, i
            tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        new_routes, new_max = split_giant_tour(tour, distance_matrix, truck_count)
        # Apply 2-opt on each route
        for idx in range(truck_count):
            if len(new_routes[idx]) > 3:
                new_routes[idx] = two_opt(new_routes[idx], distance_matrix)
        # Local search on new_routes
        dists = [route_distance(r, distance_matrix) for r in new_routes]
        for _ in range(max_iter):
            max_dist = max(dists)
            improved = False
            longest_idx = -1
            longest_dist = -1
            for idx, d in enumerate(dists):
                if d > longest_dist + 1e-12:
                    longest_dist = d
                    longest_idx = idx
            if len(new_routes[longest_idx]) > 2:
                for pos in range(1, len(new_routes[longest_idx])-1):
                    cust = new_routes[longest_idx][pos]
                    new_long_route = new_routes[longest_idx][:pos] + new_routes[longest_idx][pos+1:]
                    new_long_dist = route_distance(new_long_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == longest_idx:
                            continue
                        for ins in range(1, len(new_routes[other_idx])):
                            new_other_route = new_routes[other_idx][:ins] + [cust] + new_routes[other_idx][ins:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_max = max(new_long_dist, new_other_dist, [dists[i] for i in range(truck_count) if i not in (longest_idx, other_idx)] + [0])
                            if new_max < max_dist - 1e-12:
                                new_routes[longest_idx] = new_long_route
                                new_routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                for other_idx in range(truck_count):
                    if other_idx == longest_idx or len(new_routes[other_idx]) <= 2:
                        continue
                    for pos_long in range(1, len(new_routes[longest_idx])-1):
                        cust_a = new_routes[longest_idx][pos_long]
                        for pos_other in range(1, len(new_routes[other_idx])-1):
                            cust_b = new_routes[other_idx][pos_other]
                            new_long_route = new_routes[longest_idx][:pos_long] + [cust_b] + new_routes[longest_idx][pos_long+1:]
                            new_other_route = new_routes[other_idx][:pos_other] + [cust_a] + new_routes[other_idx][pos_other+1:]
                            new_long_dist = route_distance(new_long_route, distance_matrix)
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_max = max(new_long_dist, new_other_dist, [dists[i] for i in range(truck_count) if i not in (longest_idx, other_idx)] + [0])
                            if new_max < max_dist - 1e-12:
                                new_routes[longest_idx] = new_long_route
                                new_routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
            dists = [route_distance(r, distance_matrix) for r in new_routes]
        # Update best
        curr_max = max(dists)
        if curr_max < best_max - 1e-12:
            best_max = curr_max
            best_routes = [list(r) for r in new_routes]
            report_best_vrp(best_routes)
    
    # Ensure exactly truck_count routes (should already be)
    if len(best_routes) != truck_count:
        while len(best_routes) < truck_count:
            best_routes.append([0,0])
        while len(best_routes) > truck_count:
            # merge? shouldn't happen
            pass
    report_best_vrp(best_routes)
    return best_routes