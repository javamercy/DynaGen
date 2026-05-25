import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)
    
    # Initialize empty routes for each truck
    routes = [[depot, depot] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    
    # Helper to compute route distance
    def route_dist(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Update all route distances (after changes)
    def update_route_dists():
        for i, r in enumerate(routes):
            route_dists[i] = route_dist(r)
    update_route_dists()
    
    # Helper to compute new max distance if customer cust inserted into route r_idx at position pos
    def new_max_if_insert(cust, r_idx, pos):
        # Compute new distance for route r_idx
        route = routes[r_idx]
        # delta from inserting cust between route[pos-1] and route[pos]
        prev = route[pos-1]
        next_ = route[pos]
        delta = distance_matrix[prev, cust] + distance_matrix[cust, next_] - distance_matrix[prev, next_]
        new_dist = route_dists[r_idx] + delta
        # Compute new max among all routes
        candidate_max = new_dist
        for i in range(truck_count):
            if i == r_idx:
                continue
            if route_dists[i] > candidate_max:
                candidate_max = route_dists[i]
        return candidate_max, new_dist
    
    # Regret-based construction
    while unassigned:
        best_regret = -1e9
        best_cust = -1
        best_r_idx = -1
        best_pos = -1
        best_new_dist = -1
        best_cost_for_cust = None  # store the best cost (new max) for tie-breaking
        
        for cust in unassigned:
            costs = []  # list of (new_max, route_idx, position, new_dist_of_route)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                # Try every insertion position (between depot and depot)
                for pos in range(1, len(route)):
                    new_max, new_dist = new_max_if_insert(cust, r_idx, pos)
                    costs.append((new_max, r_idx, pos, new_dist))
            # Sort by new_max ascending
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                # Only one feasible insertion overall? Actually there are truck_count routes, each has len(route)-1 positions.
                # But to be safe, if only one option then regret is large
                regret = costs[0][0] * 2  # large regret
                best_candidate = costs[0]
            else:
                best = costs[0]
                second_best = costs[1]
                regret = second_best[0] - best[0]
                best_candidate = best
            # Tie-breaking: higher regret better; if equal, prefer higher best cost (the actual new max value) to prioritize less promising customers?
            # Following candidate 000003: if regret same, choose larger best cost (meaning worse insertion point) to insert earlier?
            # We'll do that.
            tie_breaker = best_candidate[0]  # best cost (new max)
            if regret > best_regret or (regret == best_regret and tie_breaker > best_cost_for_cust if best_cost_for_cust is not None else True):
                best_regret = regret
                best_cust = cust
                best_r_idx = best_candidate[1]
                best_pos = best_candidate[2]
                best_new_dist = best_candidate[3]
                best_cost_for_cust = tie_breaker
            elif regret == best_regret and tie_breaker == best_cost_for_cust:
                # Further tie by smaller customer index
                if cust < best_cust:
                    best_cust = cust
                    best_r_idx = best_candidate[1]
                    best_pos = best_candidate[2]
                    best_new_dist = best_candidate[3]
        
        # Insert best customer
        routes[best_r_idx].insert(best_pos, best_cust)
        route_dists[best_r_idx] = best_new_dist
        unassigned.remove(best_cust)
    
    # Post-construction: compute current best
    current_max = max(route_dists)
    best_routes = [list(r) for r in routes]
    best_max = current_max
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    # Local search improvement
    max_iter = 2 * (n - 1)  # finite iterations
    for _ in range(max_iter):
        improved = False
        
        # Inter-route relocate: try moving each customer to another route
        for r_from in range(truck_count):
            if len(routes[r_from]) <= 2:
                continue
            # list of customers positions (excluding depots)
            cust_positions = list(range(1, len(routes[r_from])-1))
            for pos_from in cust_positions:
                cust = routes[r_from][pos_from]
                # Remove cust from r_from and compute new distance for that route
                new_route_from = routes[r_from][:pos_from] + routes[r_from][pos_from+1:]
                new_dist_from = route_dist(new_route_from)
                
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = routes[r_to]
                    # Try every insertion position in route_to
                    for pos_to in range(1, len(route_to)):
                        # Insert cust into route_to at pos_to
                        new_route_to = route_to[:pos_to] + [cust] + route_to[pos_to:]
                        new_dist_to = route_dist(new_route_to)
                        # Compute new max distance
                        cand_max = max(new_dist_from, new_dist_to)
                        for r in range(truck_count):
                            if r != r_from and r != r_to:
                                cand_max = max(cand_max, route_dists[r])
                        if cand_max < best_max:
                            best_max = cand_max
                            best_routes = [list(r) for r in routes]
                            best_routes[r_from] = new_route_from
                            best_routes[r_to] = new_route_to
                            # Update actual routes and distances (we accept immediately)
                            routes[r_from] = new_route_from
                            routes[r_to] = new_route_to
                            route_dists[r_from] = new_dist_from
                            route_dists[r_to] = new_dist_to
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Inter-route swap: swap customers between two routes
        for r1 in range(truck_count):
            if len(routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(routes[r1])-1):
                cust1 = routes[r1][pos1]
                for r2 in range(r1+1, truck_count):
                    if len(routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(routes[r2])-1):
                        cust2 = routes[r2][pos2]
                        # Swap customers
                        new_route1 = list(routes[r1])
                        new_route2 = list(routes[r2])
                        new_route1[pos1] = cust2
                        new_route2[pos2] = cust1
                        new_dist1 = route_dist(new_route1)
                        new_dist2 = route_dist(new_route2)
                        cand_max = max(new_dist1, new_dist2)
                        for r in range(truck_count):
                            if r != r1 and r != r2:
                                cand_max = max(cand_max, route_dists[r])
                        if cand_max < best_max:
                            best_max = cand_max
                            best_routes = [list(r) for r in routes]
                            best_routes[r1] = new_route1
                            best_routes[r2] = new_route2
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            route_dists[r1] = new_dist1
                            route_dists[r2] = new_dist2
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Intra-route 2-opt: try reversing segments within each route
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 4:
                continue
            best_imp = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dists[r_idx]:
                        # Compute new max if we replace this route
                        cand_max = max(new_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:]))
                        if cand_max < best_max:
                            best_max = cand_max
                            best_routes = [list(r) for r in routes]
                            best_routes[r_idx] = new_route
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            improved = True
                            best_imp = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                if best_imp:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Ensure exactly truck_count routes, each starting and ending at 0
    result = []
    for r in best_routes:
        if len(r) >= 2 and r[0] == 0 and r[-1] == 0:
            result.append(r)
        else:
            # Should not happen, but fix
            new_r = [0] + [c for c in r if c != 0] + [0]
            result.append(new_r)
    while len(result) < truck_count:
        result.append([0, 0])
    return result