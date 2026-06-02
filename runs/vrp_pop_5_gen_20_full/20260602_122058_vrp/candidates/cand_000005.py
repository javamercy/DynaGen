import numpy as np
import heapq
import itertools
import collections
import math

def route_distance(route, dist):
    if len(route) < 2:
        return 0.0
    return sum(dist[route[i], route[i+1]] for i in range(len(route)-1))

def max_route_distance(routes, dist):
    return max(route_distance(r, dist) for r in routes)

def best_insert(customer, route, dist):
    best_delta = float('inf')
    best_pos = -1
    for pos in range(1, len(route)):  # between route[pos-1] and route[pos]
        prev_node = route[pos-1]
        next_node = route[pos]
        delta = dist[prev_node, customer] + dist[customer, next_node] - dist[prev_node, next_node]
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
    return best_delta, best_pos

def construct_solution(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # sort customers by distance from depot descending
    customers.sort(key=lambda c: -distance_matrix[0, c])
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_new_max = float('inf')
        for i, route in enumerate(routes):
            delta, pos = best_insert(cust, route, distance_matrix)
            new_route_dist = route_distance(route, distance_matrix) + delta
            # Compute new max if we assign this customer to this route
            current_max = max_route_distance(routes, distance_matrix)
            new_max = max(current_max, new_route_dist)
            # For routes other than i, distances unchanged; but consider that route i's distance changes
            other_max = max(route_distance(r, distance_matrix) for j, r in enumerate(routes) if j != i)
            new_max2 = max(other_max, new_route_dist)
            if new_max2 < best_new_max:
                best_new_max = new_max2
                best_route_idx = i
                best_pos = pos
            elif new_max2 == best_new_max:
                # tie-breaking: smaller route index
                if i < best_route_idx:
                    best_route_idx = i
                    best_pos = pos
        # Insert into best route at best_pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
    return routes

def improve_solution(routes, distance_matrix):
    # Local search: relocate, swap, 2-opt, cross-route 2-opt*
    n = distance_matrix.shape[0]
    truck_count = len(routes)
    best_routes = [r[:] for r in routes]
    best_max = max_route_distance(best_routes, distance_matrix)
    
    improved = True
    max_iter = 50  # bounded
    for _ in range(max_iter):
        improved = False
        # Relocate move
        for from_route_idx in range(truck_count):
            route_from = best_routes[from_route_idx]
            if len(route_from) <= 2:  # only depot
                continue
            for cust_pos in range(1, len(route_from)-1):  # exclude depots
                cust = route_from[cust_pos]
                # evaluate insertion in all other routes (including same route?) 
                # Relocate to different route only to avoid trivial same position
                for to_route_idx in range(truck_count):
                    if to_route_idx == from_route_idx:
                        continue
                    route_to = best_routes[to_route_idx]
                    delta, to_pos = best_insert(cust, route_to, distance_matrix)
                    # Remove cust from route_from
                    new_route_from = route_from[:cust_pos] + route_from[cust_pos+1:]
                    new_dist_from = route_distance(new_route_from, distance_matrix)
                    new_dist_to = route_distance(route_to, distance_matrix) + delta
                    # compute new max excluding old routes
                    other_max = 0.0
                    for k, r in enumerate(best_routes):
                        if k == from_route_idx or k == to_route_idx:
                            continue
                        other_max = max(other_max, route_distance(r, distance_matrix))
                    new_max = max(other_max, new_dist_from, new_dist_to)
                    if new_max < best_max:
                        # apply move
                        new_routes = [r[:] for idx, r in enumerate(best_routes)]
                        new_routes[from_route_idx] = new_route_from
                        new_routes[to_route_idx] = route_to[:to_pos] + [cust] + route_to[to_pos:]
                        best_max = new_max
                        best_routes = new_routes
                        improved = True
                        # report
                        from problem_solver import report_best_vrp
                        report_best_vrp(best_routes)
                        break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Swap move: exchange two customers from different routes
        for i in range(truck_count):
            route_i = best_routes[i]
            if len(route_i) <= 2:
                continue
            for pos_i in range(1, len(route_i)-1):
                cust_i = route_i[pos_i]
                for j in range(i+1, truck_count):
                    route_j = best_routes[j]
                    if len(route_j) <= 2:
                        continue
                    for pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[pos_j]
                        # compute new distances if swap
                        # Remove cust_i from route_i, insert cust_j at best position in route_i
                        new_route_i = route_i[:pos_i] + route_i[pos_i+1:]
                        delta_i, ins_i = best_insert(cust_j, new_route_i, distance_matrix)
                        new_dist_i = route_distance(new_route_i, distance_matrix) + delta_i
                        # Similarly for route_j
                        new_route_j = route_j[:pos_j] + route_j[pos_j+1:]
                        delta_j, ins_j = best_insert(cust_i, new_route_j, distance_matrix)
                        new_dist_j = route_distance(new_route_j, distance_matrix) + delta_j
                        # compute new max
                        other_max = 0.0
                        for k, r in enumerate(best_routes):
                            if k == i or k == j:
                                continue
                            other_max = max(other_max, route_distance(r, distance_matrix))
                        new_max = max(other_max, new_dist_i, new_dist_j)
                        if new_max < best_max:
                            new_routes = [r[:] for idx, r in enumerate(best_routes)]
                            # apply insertions
                            new_routes[i] = new_route_i[:ins_i] + [cust_j] + new_route_i[ins_i:]
                            new_routes[j] = new_route_j[:ins_j] + [cust_i] + new_route_j[ins_j:]
                            best_max = new_max
                            best_routes = new_routes
                            improved = True
                            from problem_solver import report_best_vrp
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # 2-opt within routes
        for r_idx in range(truck_count):
            route = best_routes[r_idx]
            if len(route) <= 3:  # at least 2 customers between depots
                continue
            best_delta = 0
            best_i = best_j = -1
            for i in range(1, len(route)-1):
                for j in range(i+1, len(route)-1):
                    # nodes: i-1,i,i+1 and j,j+1
                    prev_i = route[i-1]
                    curr_i = route[i]
                    curr_j = route[j]
                    next_j = route[j+1]
                    old_edges = distance_matrix[prev_i, curr_i] + distance_matrix[curr_j, next_j]
                    new_edges = distance_matrix[prev_i, curr_j] + distance_matrix[curr_i, next_j]
                    delta = new_edges - old_edges
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < 0:
                new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                # check if same as before? Actually the reversal is correct for 2-opt
                new_routes = [r[:] for idx, r in enumerate(best_routes)]
                new_routes[r_idx] = new_route
                new_max = max_route_distance(new_routes, distance_matrix)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = new_routes
                    improved = True
                    from problem_solver import report_best_vrp
                    report_best_vrp(best_routes)
        if improved:
            continue
        
        # Cross-route 2-opt*: consider two routes, cut each at a point, reconnect crosswise
        for i in range(truck_count):
            route_i = best_routes[i]
            if len(route_i) <= 2:
                continue
            for j in range(i+1, truck_count):
                route_j = best_routes[j]
                if len(route_j) <= 2:
                    continue
                # cut after node i_cut (0..len-2) and after node j_cut
                for cut_i in range(0, len(route_i)-1):
                    for cut_j in range(0, len(route_j)-1):
                        # new routes: route_i[:cut_i+1] + route_j[cut_j+1:] and route_j[:cut_j+1] + route_i[cut_i+1:]
                        new_i = route_i[:cut_i+1] + route_j[cut_j+1:]
                        new_j = route_j[:cut_j+1] + route_i[cut_i+1:]
                        # must start and end at depot: check if first and last are 0
                        # Actually, cut_i and cut_j are positions after nodes; depot is at start and end
                        # Ensure depots: route_i[0] and route_i[-1] are 0; same for route_j
                        # After cut, first part always starts with 0; last part must end with 0? 
                        # Since we join first part of i with second part of j, the second part includes the depot at end? 
                        # route_j[cut_j+1:] includes the last element which is 0 (depot) if cut_j+1 <= len-1. 
                        # So new_i will start with 0 and end with 0. ok.
                        # But we need to ensure no duplicate customers: the sets of customers in each part are disjoint because original routes are disjoint.
                        # Check lengths
                        if len(new_i) < 2 or len(new_j) < 2:
                            continue
                        # compute distances
                        dist_i = route_distance(new_i, distance_matrix)
                        dist_j = route_distance(new_j, distance_matrix)
                        other_max = 0.0
                        for k, r in enumerate(best_routes):
                            if k == i or k == j:
                                continue
                            other_max = max(other_max, route_distance(r, distance_matrix))
                        new_max = max(other_max, dist_i, dist_j)
                        if new_max < best_max:
                            new_routes = [r[:] for idx, r in enumerate(best_routes)]
                            new_routes[i] = new_i
                            new_routes[j] = new_j
                            best_max = new_max
                            best_routes = new_routes
                            improved = True
                            from problem_solver import report_best_vrp
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_routes

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    # initial construction
    routes = construct_solution(distance_matrix, truck_count)
    from problem_solver import report_best_vrp
    report_best_vrp(routes)
    # improvement
    better_routes = improve_solution(routes, distance_matrix)
    return better_routes