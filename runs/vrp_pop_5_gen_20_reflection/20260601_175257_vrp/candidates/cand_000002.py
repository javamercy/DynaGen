import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # sort customers by distance from depot descending (tie: lower index first)
    customers.sort(key=lambda c: (-distance_matrix[0][c], c))
    
    # initialize routes: each route is list [0,0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count  # current total distance per route
    
    def route_length(route):
        # compute distance of a route (including depot at ends)
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    # insertion phase
    for c in customers:
        best_max = float('inf')
        best_marginal = float('inf')
        best_route = None
        best_pos = None
        
        for r in range(truck_count):
            route = routes[r]
            # positions where we can insert: between any two nodes, also at end before last 0? Actually route ends with 0, so valid insert positions are indices 1 ... len(route)-1 (since we insert before the final 0)
            # But we want to insert after the starting 0 and before the ending 0, so positions from 1 to len(route)-1 inclusive (inserting at len(route)-1 means inserting just before the final 0)
            for pos in range(1, len(route)):
                # current segment: route[pos-1] -> route[pos]
                # after insertion: route[pos-1] -> c -> route[pos]
                increase = distance_matrix[route[pos-1]][c] + distance_matrix[c][route[pos]] - distance_matrix[route[pos-1]][route[pos]]
                new_dist = route_distances[r] + increase
                # compute new max distance across all routes
                new_max = new_dist
                for rr in range(truck_count):
                    if rr != r:
                        new_max = max(new_max, route_distances[rr])
                if new_max < best_max or (new_max == best_max and increase < best_marginal):
                    best_max = new_max
                    best_marginal = increase
                    best_route = r
                    best_pos = pos
                # tie on both: first encountered (smaller r, smaller pos) automatically due to iteration order
        # insert
        route = routes[best_route]
        route.insert(best_pos, c)
        route_distances[best_route] += best_marginal
    
    best_routes = [route[:] for route in routes]
    best_max = max(route_distances)
    # report initial solution
    # report_best_vrp is external; we call it when we have a complete feasible solution
    # We'll assume it exists; if not, we can wrap in try/except but spec says call it
    # We'll define a dummy function to avoid NameError in testing? Actually in harness it's defined.
    # But to be safe, we'll call it only if it's defined.
    if callable(globals().get('report_best_vrp', None)):
        report_best_vrp(best_routes)
    
    # improvement phase: deterministic local search
    max_iterations = 100 * (n + truck_count)
    for iteration in range(max_iterations):
        improved = False
        # 2-opt intra-route for each route
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_length(new_route)
                    if new_dist < route_distances[r]:
                        # check if max improves
                        new_max = new_dist
                        for rr in range(truck_count):
                            if rr != r:
                                new_max = max(new_max, route_distances[rr])
                        if new_max < best_max:
                            routes[r] = new_route
                            route_distances[r] = new_dist
                            best_max = new_max
                            best_routes = [route[:] for route in routes]
                            if callable(globals().get('report_best_vrp', None)):
                                report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # relocate: move a customer from one route to another
        for r1 in range(truck_count):
            if len(routes[r1]) <= 2:
                continue
            for idx in range(1, len(routes[r1])-1):
                customer = routes[r1][idx]
                # remove customer from r1
                new_route1 = routes[r1][:idx] + routes[r1][idx+1:]
                new_dist1 = route_length(new_route1)
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route2 = routes[r2]
                    for pos in range(1, len(route2)):
                        new_route2 = route2[:pos] + [customer] + route2[pos:]
                        new_dist2 = route_length(new_route2)
                        # compute new max
                        new_max = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                new_max = max(new_max, route_distances[rr])
                        if new_max < best_max:
                            routes[r1] = new_route1
                            route_distances[r1] = new_dist1
                            routes[r2] = new_route2
                            route_distances[r2] = new_dist2
                            best_max = new_max
                            best_routes = [route[:] for route in routes]
                            if callable(globals().get('report_best_vrp', None)):
                                report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # swap: exchange customers between two routes
        for r1 in range(truck_count):
            if len(routes[r1]) <= 2:
                continue
            for idx1 in range(1, len(routes[r1])-1):
                cust1 = routes[r1][idx1]
                for r2 in range(r1+1, truck_count):
                    if len(routes[r2]) <= 2:
                        continue
                    for idx2 in range(1, len(routes[r2])-1):
                        cust2 = routes[r2][idx2]
                        # swap
                        new_route1 = routes[r1][:idx1] + [cust2] + routes[r1][idx1+1:]
                        new_route2 = routes[r2][:idx2] + [cust1] + routes[r2][idx2+1:]
                        new_dist1 = route_length(new_route1)
                        new_dist2 = route_length(new_route2)
                        new_max = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                new_max = max(new_max, route_distances[rr])
                        if new_max < best_max:
                            routes[r1] = new_route1
                            route_distances[r1] = new_dist1
                            routes[r2] = new_route2
                            route_distances[r2] = new_dist2
                            best_max = new_max
                            best_routes = [route[:] for route in routes]
                            if callable(globals().get('report_best_vrp', None)):
                                report_best_vrp(best_routes)
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
    return best_routes