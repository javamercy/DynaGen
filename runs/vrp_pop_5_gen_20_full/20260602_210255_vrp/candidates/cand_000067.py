import numpy as np
from collections import defaultdict
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    unvisited = set(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_dist = [0.0 for _ in range(truck_count)]
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Construction: cheapest insertion into route with smallest current distance
    while unvisited:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_inc = float('inf')
        for cust in sorted(unvisited):
            # Evaluate insertion into each route, pick one with smallest resulting max distance
            best_for_cust_inc = float('inf')
            best_for_cust_route = None
            best_for_cust_pos = None
            for r_idx, route in enumerate(routes):
                # Check positions
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                    # Simulate new route distance
                    new_dist = route_dist[r_idx] + inc
                    # We want to keep max route distance low, so we consider the impact
                    # For construction, we can use a composite: encourage balancing
                    # Here: choose insertion that minimizes the resulting max distance
                    current_max = max(route_dist)
                    candidate_max = max(current_max, new_dist)  # other routes unchanged
                    if candidate_max < best_for_cust_inc or (abs(candidate_max - best_for_cust_inc) < 1e-9 and new_dist < route_dist[best_for_cust_route]):
                        best_for_cust_inc = candidate_max
                        best_for_cust_route = r_idx
                        best_for_cust_pos = pos
            if best_for_cust_inc < best_inc:
                best_inc = best_for_cust_inc
                best_cust = cust
                best_route_idx = best_for_cust_route
                best_pos = best_for_cust_pos
        if best_cust is not None:
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
            route_dist[best_route_idx] = route_distance(routes[best_route_idx])
    
    best_routes = [list(r) for r in routes]
    best_max_dist = max(route_dist)
    report_best_vrp(best_routes)
    
    def improve(routes, route_dist):
        # Local search with relocate and 2-opt*
        improved = True
        max_iter = n * truck_count
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # Sort routes by distance descending for priority
            order = sorted(range(truck_count), key=lambda i: -route_dist[i])
            for long_idx in order:
                if len(routes[long_idx]) <= 2:
                    continue
                route_long = routes[long_idx]
                # Relocate: try moving each customer to another route
                for pos in range(1, len(route_long)-1):
                    cust = route_long[pos]
                    # Remove cust
                    new_route_long = route_long[:pos] + route_long[pos+1:]
                    dist_long_new = route_distance(new_route_long)
                    for short_idx in range(truck_count):
                        if short_idx == long_idx:
                            continue
                        route_short = routes[short_idx]
                        for p in range(1, len(route_short)):
                            inc = distance_matrix[route_short[p-1]][cust] + distance_matrix[cust][route_short[p]] - distance_matrix[route_short[p-1]][route_short[p]]
                            new_route_short = route_short[:p] + [cust] + route_short[p:]
                            dist_short_new = route_distance(new_route_short)
                            new_max = max(dist_long_new, dist_short_new, max(d for i,d in enumerate(route_dist) if i not in (long_idx, short_idx)))
                            if new_max < best_max_dist - 1e-9:
                                routes[long_idx] = new_route_long
                                routes[short_idx] = new_route_short
                                route_dist[long_idx] = dist_long_new
                                route_dist[short_idx] = dist_short_new
                                best_max_dist = new_max
                                best_routes[:] = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
                # 2-opt*: exchange suffixes between two routes
                if not improved:
                    for i in range(truck_count):
                        for j in range(i+1, truck_count):
                            route_i = routes[i]
                            route_j = routes[j]
                            if len(route_i) <= 2 or len(route_j) <= 2:
                                continue
                            # Consider all breakpoints
                            for pos_i in range(1, len(route_i)-1):
                                for pos_j in range(1, len(route_j)-1):
                                    # New route i: route_i[:pos_i] + route_j[pos_j:]
                                    # New route j: route_j[:pos_j] + route_i[pos_i:]
                                    new_route_i = route_i[:pos_i] + route_j[pos_j:]
                                    new_route_j = route_j[:pos_j] + route_i[pos_i:]
                                    # Ensure depot at ends
                                    if new_route_i[-1] != 0 or new_route_j[-1] != 0:
                                        continue
                                    dist_i_new = route_distance(new_route_i)
                                    dist_j_new = route_distance(new_route_j)
                                    new_max = max(dist_i_new, dist_j_new, max(d for idx,d in enumerate(route_dist) if idx not in (i,j)))
                                    if new_max < best_max_dist - 1e-9:
                                        routes[i] = new_route_i
                                        routes[j] = new_route_j
                                        route_dist[i] = dist_i_new
                                        route_dist[j] = dist_j_new
                                        best_max_dist = new_max
                                        best_routes[:] = [list(r) for r in routes]
                                        report_best_vrp(best_routes)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    if improved:
                        break
        return routes, route_dist, best_routes, best_max_dist
    
    # First round of improvement
    routes, route_dist, best_routes, best_max_dist = improve(routes, route_dist)
    
    # Perturbation: double-bridge move on the longest route
    def perturb(routes, route_dist):
        # Identify longest route
        max_dist = max(route_dist)
        long_idx = [i for i,d in enumerate(route_dist) if abs(d-max_dist) < 1e-9][0]
        route = routes[long_idx]
        if len(route) <= 4:
            return routes, route_dist  # cannot perturb
        # Choose four random breakpoints
        nodes = list(range(1, len(route)-1))
        if len(nodes) < 4:
            return routes, route_dist
        a, b, c, d = sorted(random.sample(nodes, 4))
        # Reorder: segment1 = route[1:a], segment2 = route[a:b], segment3 = route[b:c], segment4 = route[c:-1]
        # New route: depot + segment1 + segment4 + segment3 + segment2 + depot
        new_route = [0] + route[1:a] + route[c:-1] + route[b:c] + route[a:b] + [0]
        routes[long_idx] = new_route
        route_dist[long_idx] = route_distance(new_route)
        return routes, route_dist
    
    # Apply perturbation
    routes_perturbed = [list(r) for r in routes]
    route_dist_perturbed = list(route_dist)
    routes_perturbed, route_dist_perturbed = perturb(routes_perturbed, route_dist_perturbed)
    # Re-improve from perturbed solution
    routes_perturbed, route_dist_perturbed, best_routes_pert, best_max_pert = improve(routes_perturbed, route_dist_perturbed)
    # Keep best overall
    if best_max_pert < best_max_dist - 1e-9:
        best_routes = best_routes_pert
        best_max_dist = best_max_pert
        report_best_vrp(best_routes)
    
    return best_routes