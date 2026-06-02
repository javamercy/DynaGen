import numpy as np
import math
from itertools import combinations

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # initialize routes as lists of [0,0]
    routes = [[0, 0] for _ in range(truck_count)]
    # store distance for each route
    route_dist = [0.0 for _ in range(truck_count)]
    
    def route_length(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def compute_max():
        return max(route_dist)
    
    # insert customers 1..n-1
    customers = list(range(1, n))
    # deterministic: keep original order? We'll sort to be deterministic
    # but any fixed order works; we'll use increasing customer index
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            # consider insertion positions from 1 to len(route)-1 (since depot at ends)
            for pos in range(1, len(route)):
                # compute new distance if inserting cust at pos
                old_len = route_dist[r]
                # remove edge (route[pos-1], route[pos]) and add two new edges
                prev = route[pos-1]
                next_ = route[pos]
                added = distance_matrix[prev, cust] + distance_matrix[cust, next_]
                removed = distance_matrix[prev, next_]
                new_dist = old_len + added - removed
                # compute new max distance across all routes
                new_max = max(new_dist, max(route_dist[:r] + route_dist[r+1:], default=0.0))
                if new_max < best_max or (new_max == best_max and (r < best_route_idx or (r == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r
                    best_pos = pos
        # perform insertion
        route = routes[best_route_idx]
        prev = route[best_pos-1]
        next_ = route[best_pos]
        route_dist[best_route_idx] += distance_matrix[prev, cust] + distance_matrix[cust, next_] - distance_matrix[prev, next_]
        route.insert(best_pos, cust)
    
    # call report_best_vrp for initial solution
    report_best_vrp(routes)
    
    # local search: bounded iterations
    max_iter = n * 2
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        current_max = compute_max()
        # inter-route relocate
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1)-1):  # customer positions (exclude depots)
                cust = route1[i]
                # remove cust from its route
                # compute new route1 length after removal
                prev_rem = route1[i-1]
                next_rem = route1[i+1]
                removed_cost = distance_matrix[prev_rem, cust] + distance_matrix[cust, next_rem] - distance_matrix[prev_rem, next_rem]
                new_dist_r1 = route_dist[r1] - removed_cost
                for r2 in range(truck_count):
                    route2 = routes[r2]
                    # consider insertion in same route? But that would be intra-route, handled by 2-opt
                    # So only different routes for inter-route
                    if r2 == r1:
                        continue
                    for pos in range(1, len(route2)):
                        # compute cost of inserting cust into route2 at pos
                        prev_ins = route2[pos-1]
                        next_ins = route2[pos]
                        added_cost = distance_matrix[prev_ins, cust] + distance_matrix[cust, next_ins] - distance_matrix[prev_ins, next_ins]
                        new_dist_r2 = route_dist[r2] + added_cost
                        new_max = max(new_dist_r1, new_dist_r2, max(route_dist[:r1] + route_dist[r1+1:r2] + route_dist[r2+1:], default=0.0))
                        if new_max < current_max - 1e-9:
                            # perform move
                            # remove from r1
                            r1_dist = route1.pop(i)
                            route_dist[r1] = new_dist_r1
                            # insert into r2
                            route2.insert(pos, cust)
                            route_dist[r2] = new_dist_r2
                            improved = True
                            current_max = new_max
                            report_best_vrp(routes)
                            # break out of loops? We'll continue with updated structures
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            iteration += 1
            continue
        # intra-route 2-opt
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment from i to j (inclusive)
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_length(new_route)
                    if new_len < route_dist[r] - 1e-9:
                        # compute new max
                        new_max = max(new_len, max(route_dist[:r] + route_dist[r+1:], default=0.0))
                        if new_max < current_max - 1e-9:
                            route[:] = new_route
                            route_dist[r] = new_len
                            improved = True
                            current_max = new_max
                            report_best_vrp(routes)
                            break
                if improved:
                    break
            if improved:
                break
        iteration += 1
    return routes