import numpy as np
import math
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n - 1:
        routes = []
        for i in customers:
            routes.append([0, i, 0])
        for _ in range(truck_count - len(customers)):
            routes.append([0, 0])
        # report initial feasible solution
        report_best_vrp(routes)
        return routes
    
    # Deterministic seed selection: choose customers farthest from depot, tie-break by index
    dist_from_depot = [(distance_matrix[0][i], i) for i in range(1, n)]
    dist_from_depot.sort(key=lambda x: (-x[0], x[1]))
    seeds = [x[1] for x in dist_from_depot[:truck_count]]
    
    # Cluster assignment: each customer to nearest seed
    clusters = {seed: [] for seed in seeds}
    for cust in customers:
        best_seed = min(seeds, key=lambda s: (distance_matrix[cust][s], s))
        clusters[best_seed].append(cust)
    
    def compute_route_distance(route):
        if len(route) == 2:
            return 0.0
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    def nearest_neighbor_route(nodes):
        if not nodes:
            return [0, 0]
        route = [0]
        remaining = set(nodes)
        current = 0
        while remaining:
            next_node = min(remaining, key=lambda x: distance_matrix[current][x])
            route.append(next_node)
            remaining.remove(next_node)
            current = next_node
        route.append(0)
        return route
    
    def two_opt_improve(route):
        if len(route) <= 4:
            return route
        improved = True
        max_iter = len(route) * 10
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            for i in range(1, len(route)-3):
                for j in range(i+1, len(route)-2):
                    if j-i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = compute_route_distance(new_route)
                    old_dist = compute_route_distance(route)
                    if new_dist < old_dist:
                        route = new_route
                        improved = True
        return route
    
    # Build initial routes
    routes = []
    for seed in seeds:
        clist = clusters[seed]
        route = nearest_neighbor_route(clist)
        route = two_opt_improve(route)
        routes.append(route)
    
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    def get_max_distance(routes):
        return max(compute_route_distance(r) for r in routes)
    
    best_routes = [route[:] for route in routes]
    best_max = get_max_distance(routes)
    # report initial best
    report_best_vrp(best_routes)
    
    # Improvement: move customers between routes to reduce max distance
    n_cust = n - 1
    max_iter = min(100, n_cust * truck_count)
    for _ in range(max_iter):
        # Find route with max distance
        current_max = get_max_distance(routes)
        max_idx = max(range(truck_count), key=lambda i: compute_route_distance(routes[i]))
        # Try moving a customer from max route to another route
        improved = False
        route_to_move = routes[max_idx]
        if len(route_to_move) <= 3:
            break  # route has only depot, no customers to move
        customers_in_route = route_to_move[1:-1]
        for cust in customers_in_route:
            if improved:
                break
            for target_idx in range(truck_count):
                if target_idx == max_idx:
                    continue
                target_route = routes[target_idx]
                if len(target_route) == 2:
                    new_target = [0, cust, 0]
                    increase = 2 * distance_matrix[0][cust]
                else:
                    best_increase = float('inf')
                    best_new_target = None
                    for pos in range(1, len(target_route)):
                        a = target_route[pos-1]
                        b = target_route[pos]
                        new_dist = distance_matrix[a][cust] + distance_matrix[cust][b] - distance_matrix[a][b]
                        if new_dist < best_increase:
                            best_increase = new_dist
                            best_new_target = target_route[:pos] + [cust] + target_route[pos:]
                    new_target = best_new_target
                    increase = best_increase
                # Remove cust from original route
                new_orig = [0] + [c for c in route_to_move[1:-1] if c != cust] + [0]
                if len(new_orig) == 2:
                    new_orig_dist = 0
                else:
                    new_orig_dist = compute_route_distance(new_orig)
                # Compute new max if move applied
                temp_routes = [route[:] for route in routes]
                temp_routes[max_idx] = new_orig
                temp_routes[target_idx] = new_target
                new_max = get_max_distance(temp_routes)
                if new_max < current_max - 1e-9:
                    # Accept move
                    routes[max_idx] = new_orig
                    routes[target_idx] = new_target
                    current_max = new_max
                    improved = True
                    if new_max < best_max - 1e-9:
                        best_max = new_max
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(best_routes)
                    break
        if not improved:
            break
    
    # Final ensure routes start and end at 0
    for i, r in enumerate(best_routes):
        if r[0] != 0 or r[-1] != 0:
            best_routes[i] = [0] + r[1:-1] + [0] if len(r)>2 else [0,0]
    return best_routes