import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    max_restarts = min(50, 2 * n)
    
    best_routes = None
    best_max = float('inf')
    
    for restart in range(max_restarts):
        # Select seeds: farthest customers from depot, one per route
        # Sort all customers by distance to depot descending
        dist_to_depot = [(distance_matrix[0, i], i) for i in customers]
        dist_to_depot.sort(reverse=True)
        seeds = [dist_to_depot[i][1] for i in range(min(truck_count, len(customers)))]
        # If more trucks than customers, some routes stay empty
        # Initialize routes with seeds
        routes = [[0, s, 0] for s in seeds]
        # Add empty routes for remaining trucks
        for _ in range(truck_count - len(seeds)):
            routes.append([0, 0])
        route_lengths = []
        for route in routes:
            if len(route) > 2:
                length = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            else:
                length = 0.0
            route_lengths.append(length)
        
        # Remaining customers (excluding seeds)
        remaining = [c for c in customers if c not in seeds]
        random.shuffle(remaining)
        
        # Greedy insertion to minimize max distance
        for c in remaining:
            best_route = -1
            best_pos = -1
            best_new_max = float('inf')
            for r_idx, route in enumerate(routes):
                # Consider all insertion positions (including after last before depot)
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len = route_lengths[r_idx] + increase
                    new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route = r_idx
                        best_pos = pos
            # Insert
            route = routes[best_route]
            route.insert(best_pos, c)
            route_lengths[best_route] = best_new_max  # approximate; recompute exactly later
            # Recompute exactly for accuracy
            route_lengths[best_route] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            report_best_vrp(routes)
        
        # Improvement loop: intra 2-opt and inter relocate/swap
        improved = True
        max_iter = 10 * n * truck_count
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old:
                            route[i:j+1] = reversed(route[i:j+1])
                            route_lengths[r_idx] -= old - new
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Inter-route relocate
            for r_from in range(truck_count):
                route_from = routes[r_from]
                if len(route_from) <= 2:
                    continue
                for idx_c in range(1, len(route_from)-1):
                    c = route_from[idx_c]
                    prev = route_from[idx_c-1]
                    nxt = route_from[idx_c+1]
                    cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len_from = route_lengths[r_from] - cost_remove
                    for r_to in range(truck_count):
                        if r_to == r_from:
                            continue
                        route_to = routes[r_to]
                        for pos in range(1, len(route_to)):
                            prev_to = route_to[pos-1]
                            nxt_to = route_to[pos]
                            cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                            new_len_to = route_lengths[r_to] + cost_insert
                            new_max = max(route_lengths[:r_from] + [new_len_from] + route_lengths[r_from+1:r_to] + [new_len_to] + route_lengths[r_to+1:])
                            current_max = max(route_lengths)
                            if new_max < current_max:
                                # Perform move
                                route_from.pop(idx_c)
                                route_lengths[r_from] = new_len_from
                                route_to.insert(pos, c)
                                route_lengths[r_to] = new_len_to
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
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
                    c1 = route1[idx1]
                    prev1 = route1[idx1-1]
                    nxt1 = route1[idx1+1]
                    cost_remove1 = distance_matrix[prev1, c1] + distance_matrix[c1, nxt1] - distance_matrix[prev1, nxt1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            c2 = route2[idx2]
                            prev2 = route2[idx2-1]
                            nxt2 = route2[idx2+1]
                            cost_remove2 = distance_matrix[prev2, c2] + distance_matrix[c2, nxt2] - distance_matrix[prev2, nxt2]
                            # Insert c2 into route1 at idx1
                            cost_insert1 = distance_matrix[prev1, c2] + distance_matrix[c2, nxt1] - distance_matrix[prev1, nxt1]
                            new_len1 = route_lengths[r1] - cost_remove1 + cost_insert1
                            # Insert c1 into route2 at idx2
                            cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, nxt2] - distance_matrix[prev2, nxt2]
                            new_len2 = route_lengths[r2] - cost_remove2 + cost_insert2
                            new_max = max(route_lengths[:r1] + [new_len1] + route_lengths[r1+1:r2] + [new_len2] + route_lengths[r2+1:])
                            current_max = max(route_lengths)
                            if new_max < current_max:
                                # Perform swap
                                del route1[idx1]
                                del route2[idx2]
                                route1.insert(idx1, c2)
                                route2.insert(idx2, c1)
                                route_lengths[r1] = new_len1
                                route_lengths[r2] = new_len2
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Evaluate current solution
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
    
    if best_routes is None:
        best_routes = routes  # fallback
    # Ensure exactly truck_count routes, each starting and ending at 0
    # Already guaranteed
    return best_routes