import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    
    # Nearest neighbor giant tour
    unvisited = set(customers)
    current = 0
    tour = []
    while unvisited:
        next_cust = min(unvisited, key=lambda c: distance_matrix[current, c])
        tour.append(next_cust)
        unvisited.remove(next_cust)
        current = next_cust
    # perm is the order of customers in the tour
    perm = tour  # length m
    
    # Precompute distances for segment cost
    # dist_depot_to_first[i] = dist(0, perm[i])
    d_first = np.array([distance_matrix[0, c] for c in perm])
    # dist_last_to_depot[i] = dist(perm[i], 0)
    d_last = np.array([distance_matrix[c, 0] for c in perm])
    # between distances
    d_between = np.array([distance_matrix[perm[i], perm[i+1]] for i in range(m-1)])
    
    # DP: minimize max route distance for first i customers (0-indexed) with k routes
    # dp[k][i] = minimal max distance
    INF = 1e18
    # dp will be list of arrays; we store for all k up to truck_count
    dp = [np.full(m, INF) for _ in range(truck_count+1)]
    # For reconstruction
    pre = [[-1]*m for _ in range(truck_count+1)]  # pre[k][i] = j (last split point)
    
    # Compute segment cost from i to j inclusive
    # segment_cost(i,j) = d_first[i] + sum_{k=i}^{j-1} d_between[k] + d_last[j]
    # Precompute prefix sums of d_between for quick segment cost
    prefix = np.zeros(m)
    for i in range(1, m):
        prefix[i] = prefix[i-1] + d_between[i-1]
    def seg_cost(i, j):
        # i <= j
        return d_first[i] + (prefix[j] - prefix[i]) + d_last[j]
    
    # Base: 1 route
    for i in range(m):
        dp[1][i] = seg_cost(0, i)
        pre[1][i] = -1
    
    # Fill DP for k >= 2
    for k in range(2, truck_count+1):
        for i in range(k-1, m):  # need at least k customers for k routes (each route at least 1 cust)
            best = INF
            best_j = -1
            for j in range(k-2, i):  # j is the last index of previous route
                # previous best max for first j customers with k-1 routes
                prev_max = dp[k-1][j]
                if prev_max == INF:
                    continue
                cur_cost = seg_cost(j+1, i)
                new_max = max(prev_max, cur_cost)
                if new_max < best:
                    best = new_max
                    best_j = j
            dp[k][i] = best
            pre[k][i] = best_j
    
    # Find best number of routes k_opt (<= truck_count) that minimizes max distance
    best_max = INF
    best_k = truck_count
    for k in range(1, truck_count+1):
        if dp[k][m-1] < best_max:
            best_max = dp[k][m-1]
            best_k = k
    
    # Reconstruct routes for best_k
    routes = []
    idx = m-1
    k = best_k
    while k > 0:
        j = pre[k][idx]  # start index of this segment is j+1
        start = j+1 if j != -1 else 0
        # route from start to idx
        segment = [0] + [perm[i] for i in range(start, idx+1)] + [0]
        routes.insert(0, segment)
        idx = j
        k -= 1
    
    # Add empty routes if fewer than truck_count
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # Compute route lengths
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    route_lengths = [route_dist(r) for r in routes]
    current_max = max(route_lengths)
    
    # Report initial
    report_best_vrp(routes)
    
    # Local search improvement (bounded passes)
    max_passes = min(100, m * truck_count)
    for _ in range(max_passes):
        improved = False
        
        # Intra-route 2-opt
        for ri in range(truck_count):
            route = routes[ri]
            if len(route) <= 3:
                continue
            best_improve = False
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                    if new < old:
                        new_len = route_lengths[ri] - old + new
                        other_max = max(route_lengths[:ri] + route_lengths[ri+1:])
                        new_max = max(new_len, other_max)
                        if new_max < current_max:
                            # Apply move
                            route[a:b+1] = reversed(route[a:b+1])
                            route_lengths[ri] = new_len
                            current_max = new_max
                            report_best_vrp(routes)
                            best_improve = True
                            break
                if best_improve:
                    break
            if best_improve:
                improved = True
                break
        if improved:
            continue
        
        # Inter-route relocate
        for r_from in range(truck_count):
            route_from = routes[r_from]
            if len(route_from) <= 2:
                continue
            for idx in range(1, len(route_from)-1):
                cust = route_from[idx]
                # cost to remove cust from route_from
                prev_f = route_from[idx-1]
                next_f = route_from[idx+1]
                cost_remove = distance_matrix[prev_f, cust] + distance_matrix[cust, next_f] - distance_matrix[prev_f, next_f]
                new_len_from = route_lengths[r_from] - cost_remove
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = routes[r_to]
                    # find best insertion
                    best_inc = INF
                    best_pos = -1
                    for pos in range(1, len(route_to)):
                        prev_t = route_to[pos-1]
                        next_t = route_to[pos]
                        inc = distance_matrix[prev_t, cust] + distance_matrix[cust, next_t] - distance_matrix[prev_t, next_t]
                        if inc < best_inc:
                            best_inc = inc
                            best_pos = pos
                    new_len_to = route_lengths[r_to] + best_inc
                    other_max = max(route_lengths[i] for i in range(truck_count) if i not in (r_from, r_to))
                    new_max = max(new_len_from, new_len_to, other_max)
                    if new_max < current_max:
                        # apply
                        route_from.pop(idx)
                        route_to.insert(best_pos, cust)
                        route_lengths[r_from] = new_len_from
                        route_lengths[r_to] = new_len_to
                        current_max = new_max
                        report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Inter-route swap
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                prev1 = route1[idx1-1]
                next1 = route1[idx1+1]
                cost_rem1 = distance_matrix[prev1, cust1] + distance_matrix[cust1, next1] - distance_matrix[prev1, next1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        cust2 = route2[idx2]
                        prev2 = route2[idx2-1]
                        next2 = route2[idx2+1]
                        cost_rem2 = distance_matrix[prev2, cust2] + distance_matrix[cust2, next2] - distance_matrix[prev2, next2]
                        # insert cust2 into route1 at idx1
                        add1 = distance_matrix[prev1, cust2] + distance_matrix[cust2, next1] - distance_matrix[prev1, next1]
                        new_len1 = route_lengths[r1] - cost_rem1 + add1
                        # insert cust1 into route2 at idx2
                        add2 = distance_matrix[prev2, cust1] + distance_matrix[cust1, next2] - distance_matrix[prev2, next2]
                        new_len2 = route_lengths[r2] - cost_rem2 + add2
                        other_max = max(route_lengths[i] for i in range(truck_count) if i not in (r1, r2))
                        new_max = max(new_len1, new_len2, other_max)
                        if new_max < current_max:
                            # apply swap
                            del route1[idx1]
                            del route2[idx2]
                            route1.insert(idx1, cust2)
                            route2.insert(idx2, cust1)
                            route_lengths[r1] = new_len1
                            route_lengths[r2] = new_len2
                            current_max = new_max
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return routes