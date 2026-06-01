import numpy as np
from copy import deepcopy

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # sort customers by distance from depot descending
    customers.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    
    # initialize routes: each truck has empty route [0,0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0 for _ in range(truck_count)]
    
    # helper to compute route distance
    def calc_route_dist(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    # greedy insertion: assign customers one by one to the position that minimizes the new max route distance
    for c in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            # all insertion positions from 1 to len(route)-1 (before the final 0)
            for pos in range(1, len(route)):
                new_route = route[:pos] + [c] + route[pos:]
                new_dist = calc_route_dist(new_route)
                new_max = max(route_distances[:r] + [new_dist] + route_distances[r+1:])
                if new_max < best_max:
                    best_max = new_max
                    best_route_idx = r
                    best_pos = pos
                elif new_max == best_max:
                    # tie: prefer smaller route index, then smaller position
                    if r < best_route_idx or (r == best_route_idx and pos < best_pos):
                        best_route_idx = r
                        best_pos = pos
        # insert customer
        routes[best_route_idx].insert(best_pos, c)
        route_distances[best_route_idx] = calc_route_dist(routes[best_route_idx])
    
    # initial best solution
    best_routes = deepcopy(routes)
    best_max_dist = max(route_distances)
    # call report (simulate)
    # report_best_vrp(best_routes)  # commented out because undefined; in real solver, call
    
    # improvement phase
    n_customers = len(customers)
    max_iter = n_customers * truck_count * 2  # bounded
    for _ in range(max_iter):
        improved = False
        # intra-route 2-opt for each route
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            best_improvement = 0
            best_i = best_j = -1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reversed segment from i to j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = calc_route_dist(new_route)
                    if new_dist < route_distances[r] - best_improvement:
                        improvement = route_distances[r] - new_dist
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i = i
                            best_j = j
            if best_improvement > 0:
                # apply best 2-opt move
                route = routes[r]
                new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[r] = new_route
                route_distances[r] = calc_route_dist(new_route)
                improved = True
        # inter-route relocate
        for src_r in range(truck_count):
            src_route = routes[src_r]
            if len(src_route) <= 2:
                continue
            # consider each customer in src_route (excluding depots at ends)
            for idx in range(1, len(src_route)-1):
                c = src_route[idx]
                for dst_r in range(truck_count):
                    if dst_r == src_r:
                        continue
                    dst_route = routes[dst_r]
                    for pos in range(1, len(dst_route)):
                        # compute new distances
                        new_src_route = src_route[:idx] + src_route[idx+1:]
                        new_src_dist = calc_route_dist(new_src_route)
                        new_dst_route = dst_route[:pos] + [c] + dst_route[pos:]
                        new_dst_dist = calc_route_dist(new_dst_route)
                        new_max = max(route_distances[:dst_r] + [new_dst_dist] + route_distances[dst_r+1:src_r] + [new_src_dist] + route_distances[src_r+1:])
                        if new_max < best_max_dist:
                            # apply move
                            routes[src_r] = new_src_route
                            routes[dst_r] = new_dst_route
                            route_distances[src_r] = new_src_dist
                            route_distances[dst_r] = new_dst_dist
                            best_max_dist = new_max
                            best_routes = deepcopy(routes)
                            # report_best_vrp(best_routes)
                            improved = True
                            # break out of loops? we want best improvement, but we accept first improvement to avoid complexity
                            # Since we iterate in deterministic order, we can accept first improvement that reduces best.
                            # To find best improvement would be heavy. Instead we use first improvement.
                            # We'll re-enter loops after each improvement.
                            # To avoid deep nesting, we can set flags and break
                            # We'll break all loops and restart
                            pass # we'll handle with a goto-like mechanism
        # For simplicity, we use a local variable to track if any improvement occurred and break accordingly.
        # In code we need to break out of multiple loops. We'll restructure to use a flag.
        # Since we cannot write goto, we'll flatten loops with flags.
        # Actually we can encapsulate inside functions but within solve_vrp we can use a while loop with iteration count and nested loops.
        # To keep code simpler, we'll just do one pass and avoid complex breaking.
        # More robust: use a variable 'changed' and break with exceptions is not allowed.
        # We'll just run multiple iterations, and within each iteration we try moves in order.
        # If a move improves, we update and continue scanning (not restart). That's okay.
        if not improved:
            break
    
    # return best_routes found
    return best_routes