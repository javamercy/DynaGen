import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unrouted = set(range(1, n))
    
    # Greedy insertion
    while unrouted:
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_cost = float('inf')
        for cust in sorted(unrouted):  # deterministic order
            for r_idx, route in enumerate(routes):
                # insertion positions: from 1 to len(route)-1 (since ends are depot)
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if cost < best_cost or (cost == best_cost and (cust < best_customer or (cust == best_customer and r_idx < best_route_idx))):
                        best_cost = cost
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
        # Insert best customer
        routes[best_route_idx].insert(best_pos, best_customer)
        unrouted.remove(best_customer)
    
    # Helper to compute route distance
    def route_dist(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    
    # Compute initial max distance and report
    dists = [route_dist(r) for r in routes]
    best_max = max(dists)
    best_routes = [r[:] for r in routes]
    # report_best_vrp would be called here but we store for later
    
    # Improvement loops
    improved = True
    iteration = 0
    max_iterations = n * n  # bounded
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Intra-route 2-opt for each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dist(route):
                        # accept if max improves or stays same? We'll accept if new max <= old max
                        old_max = max(route_dist(r) for r in routes)
                        routes[r_idx] = new_route
                        new_max = max(route_dist(r) for r in routes)
                        if new_max > old_max:
                            # revert
                            routes[r_idx] = route
                        else:
                            improved = True
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [r[:] for r in routes]
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route relocate: move customer from max route to other routes
        # Identify route with maximum distance
        max_idx = max(range(truck_count), key=lambda i: route_dist(routes[i]))
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            continue
        # Try removing each customer from max_route and insert into another route
        for cust_pos in range(1, len(max_route)-1):
            cust = max_route[cust_pos]
            # remove customer from max_route temporarily
            temp_route = max_route[:cust_pos] + max_route[cust_pos+1:]
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                for pos in range(1, len(routes[r_idx])):
                    new_route_other = routes[r_idx][:pos] + [cust] + routes[r_idx][pos:]
                    # compute new distances
                    old_dists = [route_dist(r) for r in routes]
                    old_max_dist = max(old_dists)
                    new_max_dist = max(
                        route_dist(temp_route),
                        route_dist(new_route_other),
                        max(route_dist(r) for i,r in enumerate(routes) if i not in (max_idx, r_idx))
                    )
                    if new_max_dist < old_max_dist - 1e-9:  # strict improvement
                        # accept
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