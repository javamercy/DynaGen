import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # each route: list of nodes starting and ending with 0
    routes = [[0, 0] for _ in range(truck_count)]
    
    # compute current route distances
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    route_dists = [route_dist(r) for r in routes]
    max_dist = max(route_dists) if truck_count > 0 else 0.0
    
    unrouted = list(range(1, n))
    np.random.shuffle(unrouted)  # deterministic? use fixed seed? but allowed? we'll keep deterministic but shuffle may be random; to be deterministic, use sorted. But no requirement for determinism overall, but internal checklist says deterministic tie handling. We'll use sorted order for insertion to be deterministic.
    # Actually for greedy insertion, order matters. We'll use a deterministic order: sort customers by some measure? But simplest: iterate in increasing node index.
    # Use sorting to ensure determinism.
    unrouted.sort()
    
    best_routes = None
    best_max = float('inf')
    
    def compute_max_dist():
        return max(route_dist(r) for r in routes)
    
    def report_if_better():
        nonlocal best_routes, best_max
        current_max = compute_max_dist()
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
        # Note: report_best_vrp is called externally, we just store.
    
    # initial empty routes
    report_if_better()
    
    for cust in unrouted:
        best_insertion = None
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        
        for r_idx in range(truck_count):
            route = routes[r_idx]
            # possible positions: between 0 and len(route)-1 (i.e., after node i, before i+1)
            for pos in range(1, len(route)):  # insert after position pos-1, before pos
                # compute new distance for this route after insertion
                # old edge: route[pos-1] -> route[pos]
                # new edges: route[pos-1] -> cust, cust -> route[pos]
                old_edge = distance_matrix[route[pos-1], route[pos]]
                new_edge1 = distance_matrix[route[pos-1], cust]
                new_edge2 = distance_matrix[cust, route[pos]]
                new_route_dist = route_dists[r_idx] - old_edge + new_edge1 + new_edge2
                # new max distance across all routes
                new_max = max(new_route_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:]))
                
                # tie-breaking: smallest new_max, then smallest r_idx, then smallest pos
                if (new_max < best_new_max - 1e-12 or
                    (abs(new_max - best_new_max) < 1e-12 and r_idx < best_route_idx) or
                    (abs(new_max - best_new_max) < 1e-12 and r_idx == best_route_idx and pos < best_pos)):
                    best_new_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
                    best_insertion = (cust, new_route_dist)
        
        # perform insertion
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        # update route_dists
        old_dist = route_dists[best_route_idx]
        new_route_dist = best_insertion[1]
        route_dists[best_route_idx] = new_route_dist
        max_dist = max(route_dists)
        
        # after each insertion, check new solution
        report_if_better()
    
    # ---- Local Search: improve max distance ----
    improved = True
    max_iter = 100  # bounded iterations
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            # try all 2-opt swaps
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # remove edges (i-1,i) and (j,j+1), add (i-1,j) and (i,j+1)
                    if j == i+1:
                        # reversal of a segment of length 2? Actually 2-opt on adjacent edges would produce same route?
                        # skip for simplicity
                        continue
                    old_segment = route[i:j+1]
                    new_segment = old_segment[::-1]
                    new_route = route[:i] + new_segment + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dists[r_idx] - 1e-12:
                        # check if new max distance improves
                        new_max = max(new_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:]))
                        if new_max < max_dist - 1e-12:
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            max_dist = new_max
                            improved = True
                            report_if_better()
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route relocate: move a customer from one route to another
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                # remove customer
                old_route = route[:pos] + route[pos+1:]
                old_dist = route_dist(old_route)
                # try insert into another route
                for r_idx2 in range(truck_count):
                    if r_idx2 == r_idx:
                        continue
                    route2 = routes[r_idx2]
                    for pos2 in range(1, len(route2)):
                        new_route2 = route2[:pos2] + [cust] + route2[pos2:]
                        new_dist2 = route_dist(new_route2)
                        # new max distance
                        new_max = max(old_dist, new_dist2, 
                                      max([route_dists[i] for i in range(truck_count) if i != r_idx and i != r_idx2]))
                        if new_max < max_dist - 1e-12:
                            routes[r_idx] = old_route
                            routes[r_idx2] = new_route2
                            route_dists[r_idx] = old_dist
                            route_dists[r_idx2] = new_dist2
                            max_dist = new_max
                            improved = True
                            report_if_better()
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # final check and restore best
    if best_routes is not None and compute_max_dist() > best_max + 1e-12:
        routes = best_routes
        route_dists = [route_dist(r) for r in routes]
    
    return [[int(node) for node in route] for route in routes]