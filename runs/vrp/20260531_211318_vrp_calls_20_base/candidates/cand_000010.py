import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unrouted = set(range(1, n))
    
    # Greedy insertion (same as parent)
    while unrouted:
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_cost = float('inf')
        for cust in sorted(unrouted):
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if cost < best_cost or (cost == best_cost and (cust < best_customer or (cust == best_customer and r_idx < best_route_idx))):
                        best_cost = cost
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
        routes[best_route_idx].insert(best_pos, best_customer)
        unrouted.remove(best_customer)
    
    def route_dist(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    
    dists = [route_dist(r) for r in routes]
    best_max = max(dists)
    best_routes = [r[:] for r in routes]
    # report_best_vrp would be called here
    
    # Improvement with adaptive schedule
    max_iterations = n * n
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Adaptive threshold: linear decay from 0 (equal) to strict negative
        # In first half, allow non-worsening; in second half, require strict improvement
        if iteration <= max_iterations // 2:
            threshold = 1e-9  # accept if new_max <= old_max
        else:
            threshold = -1e-9  # accept only if new_max < old_max
        
        # Intra-route 2-opt
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    old_dist = route_dist(route)
                    old_max_temp = max(route_dist(r) for r in routes)
                    # compute new max if we apply the change
                    new_routes = [r[:] for r in routes]
                    new_routes[r_idx] = new_route
                    new_max_temp = max(route_dist(r) for r in new_routes)
                    if new_max_temp < old_max_temp + threshold:
                        # accept
                        routes[r_idx] = new_route
                        improved = True
                        if new_max_temp < best_max:
                            best_max = new_max_temp
                            best_routes = [r[:] for r in routes]
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Inter-route relocate (from max route)
        max_idx = max(range(truck_count), key=lambda i: route_dist(routes[i]))
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            continue
        for cust_pos in range(1, len(max_route)-1):
            cust = max_route[cust_pos]
            temp_route = max_route[:cust_pos] + max_route[cust_pos+1:]
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                for pos in range(1, len(routes[r_idx])):
                    new_route_other = routes[r_idx][:pos] + [cust] + routes[r_idx][pos:]
                    old_dists = [route_dist(r) for r in routes]
                    old_max_dist = max(old_dists)
                    new_max_dist = max(
                        route_dist(temp_route),
                        route_dist(new_route_other),
                        max(route_dist(r) for i,r in enumerate(routes) if i not in (max_idx, r_idx))
                    )
                    if new_max_dist < old_max_dist + threshold:
                        routes[max_idx] = temp_route
                        routes[r_idx] = new_route_other
                        improved = True
                        if new_max_dist < best_max:
                            best_max = new_max_dist
                            best_routes = [r[:] for r in routes]
                        break
                if improved:
                    break
            if improved:
                break
    
    # report best found
    # report_best_vrp(best_routes)
    return best_routes