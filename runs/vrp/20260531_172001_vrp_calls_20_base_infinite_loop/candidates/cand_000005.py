import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    # helper: compute route distance
    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    
    # helper: evaluate insertion cost and new route
    def evaluate_insertion(route, customer, pos):
        new_route = route[:pos] + [customer] + route[pos:]
        return route_distance(new_route) - route_distance(route), new_route
    
    # Construction: greedy insertion with balancing
    routes = [[depot, depot] for _ in range(truck_count)]
    # sort customers by distance from depot descending (deterministic)
    sorted_cust = sorted(customers, key=lambda c: distance_matrix[depot, c], reverse=True)
    for cust in sorted_cust:
        best_cost = float('inf')
        best_route_idx = None
        best_route = None
        best_pos = None
        for r_idx, route in enumerate(routes):
            # possible positions: between 0 and last 0, but avoid inserting after last 0? Actually positions 0 to len(route)-1 inclusive
            for pos in range(1, len(route)):  # insert after depot: pos from 1 to len(route)-1 (before final 0)
                cost, new_route = evaluate_insertion(route, cust, pos)
                # tie-breaking: prefer smaller current route total distance
                if cost < best_cost - 1e-9 or (abs(cost - best_cost) < 1e-9 and route_distance(route) < route_distance(routes[best_route_idx])):
                    best_cost = cost
                    best_route_idx = r_idx
                    best_route = new_route
                    best_pos = pos
        routes[best_route_idx] = best_route
    
    # initial best
    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    
    # helper to update best
    def update_best(routes):
        nonlocal best_routes, best_max
        current_max = max(route_distance(r) for r in routes)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            # Note: we assume report_best_vrp is defined somewhere (not in scope). The contract says to call it.
            # We'll define it as a dummy if not present, but we'll just call it.
            # Since it's not defined in the environment, we need to assume it's available (like from outer scope).
            # The contract explicitly says to call report_best_vrp. We'll call it.
            try:
                report_best_vrp(routes)
            except:
                pass
    
    update_best(routes)
    
    # Local search: first-improvement, bounded iterations
    max_iter = max(100, n * 5)
    iter_count = 0
    improved = True
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # Relocate: move a customer to a different position (same route or different)
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 2:  # only depot-depot
                continue
            for i in range(1, len(route)-1):  # positions of customers (excluding depots)
                cust = route[i]
                # remove customer from current route
                temp_route = route[:i] + route[i+1:]
                old_cost_r = route_distance(route)
                new_cost_r = route_distance(temp_route)
                # try inserting in every route including same
                for other_idx in range(truck_count):
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        new_route = other_route[:pos] + [cust] + other_route[pos:]
                        # compute new max distance if we apply this move
                        # careful: we need to compute full set after move
                        # To avoid recomputing all, we can compute delta
                        old_max = max(route_distance(r) for r in routes)
                        new_routes = routes[:]
                        new_routes[r_idx] = temp_route
                        new_routes[other_idx] = new_route
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < old_max - 1e-9:
                            routes = new_routes
                            improved = True
                            update_best(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Swap: exchange two customers
        for r1_idx in range(truck_count):
            route1 = routes[r1_idx]
            if len(route1) <= 2:
                continue
            for r2_idx in range(r1_idx, truck_count):
                route2 = routes[r2_idx]
                if len(route2) <= 2:
                    continue
                for i1 in range(1, len(route1)-1):
                    cust1 = route1[i1]
                    for i2 in range(1, len(route2)-1):
                        if r1_idx == r2_idx and i1 >= i2:
                            continue
                        cust2 = route2[i2]
                        # create new routes
                        new_route1 = route1[:]
                        new_route1[i1] = cust2
                        new_route2 = route2[:]
                        new_route2[i2] = cust1
                        new_routes = routes[:]
                        new_routes[r1_idx] = new_route1
                        new_routes[r2_idx] = new_route2
                        old_max = max(route_distance(r) for r in routes)
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < old_max - 1e-9:
                            routes = new_routes
                            improved = True
                            update_best(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-2opt: reverse segment within a route
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 4:  # at least 2 customers
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_routes = routes[:]
                    new_routes[r_idx] = new_route
                    old_max = max(route_distance(r) for r in routes)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < old_max - 1e-9:
                        routes = new_routes
                        improved = True
                        update_best(routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Cross-2opt: exchange suffixes between two routes
        for r1_idx in range(truck_count):
            route1 = routes[r1_idx]
            if len(route1) < 3:
                continue
            for r2_idx in range(r1_idx+1, truck_count):
                route2 = routes[r2_idx]
                if len(route2) < 3:
                    continue
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        # new route1: route1[:i] + route2[j:-1] + [depot]
                        # new route2: route2[:j] + route1[i:-1] + [depot]
                        new_route1 = route1[:i] + route2[j:-1] + [depot]
                        new_route2 = route2[:j] + route1[i:-1] + [depot]
                        new_routes = routes[:]
                        new_routes[r1_idx] = new_route1
                        new_routes[r2_idx] = new_route2
                        old_max = max(route_distance(r) for r in routes)
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < old_max - 1e-9:
                            routes = new_routes
                            improved = True
                            update_best(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_routes