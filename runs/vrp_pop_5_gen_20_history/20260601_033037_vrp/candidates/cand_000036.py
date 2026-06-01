import numpy as np
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_distance(route):
        total = 0.0
        if route:
            total += distance_matrix[0, route[0]] + distance_matrix[route[-1], 0]
            for i in range(len(route)-1):
                total += distance_matrix[route[i], route[i+1]]
        return total
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    best_routes = None
    best_max = float('inf')
    
    def report_best(routes):
        nonlocal best_routes, best_max
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [list(r) for r in routes]
    
    # Beam search parameters
    beam_width = 10
    # initial state: all routes empty, all customers unassigned
    initial_routes = [[] for _ in range(truck_count)]
    initial_unassigned = set(customers)
    beam = [(initial_routes, initial_unassigned)]
    
    for _ in range(len(customers)):
        candidates = []
        for routes, unassigned in beam:
            for cust in sorted(unassigned):
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    # find best insertion position (min delta)
                    if not route:
                        delta = distance_matrix[0, cust] + distance_matrix[cust, 0]
                        best_pos = 0
                    else:
                        best_delta = float('inf')
                        best_pos = None
                        for pos in range(len(route)+1):
                            if pos == 0:
                                prev = 0
                                nxt = route[0]
                            elif pos == len(route):
                                prev = route[-1]
                                nxt = 0
                            else:
                                prev = route[pos-1]
                                nxt = route[pos]
                            delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                            if delta < best_delta:
                                best_delta = delta
                                best_pos = pos
                    # create new state
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx].insert(best_pos, cust)
                    new_unassigned = set(unassigned)
                    new_unassigned.remove(cust)
                    new_max = max_distance(new_routes)
                    candidates.append((new_max, new_routes, new_unassigned, id(cust)))
        # sort by max distance, then by a tie-breaker (customer id to be deterministic)
        candidates.sort(key=lambda x: (x[0], x[3]))
        beam = [(r, u) for _, r, u, _ in candidates[:beam_width]]
        # early report if any state is complete (unassigned empty)
        for routes, unassigned in beam:
            if not unassigned:
                report_best(routes)
    
    final_routes = None
    if best_routes is not None:
        final_routes = best_routes
    else:
        final_routes = beam[0][0] if beam else initial_routes
    
    # Improvement: relocate from longest route
    for _ in range(n):
        dists = [route_distance(r) for r in final_routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = final_routes[max_idx]
        if not interior:
            break
        moved = False
        for cust in list(interior):
            if moved:
                break
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = final_routes[other_idx]
                best_pos = None
                best_delta = float('inf')
                if not other_route:
                    delta = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    best_delta = delta
                    best_pos = 0
                else:
                    for pos in range(len(other_route)+1):
                        if pos == 0:
                            prev = 0
                            nxt = other_route[0]
                        elif pos == len(other_route):
                            prev = other_route[-1]
                            nxt = 0
                        else:
                            prev = other_route[pos-1]
                            nxt = other_route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                # Tentative new routes
                new_routes = [list(r) for r in final_routes]
                new_routes[max_idx].remove(cust)
                new_routes[other_idx].insert(best_pos, cust)
                new_max = max_distance(new_routes)
                if new_max < best_max - 1e-12:
                    final_routes = new_routes
                    report_best(final_routes)
                    moved = True
                    break
        if not moved:
            break
    
    # Improvement: swap between longest route and another
    for _ in range(n):
        dists = [route_distance(r) for r in final_routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = final_routes[max_idx]
        if not interior:
            break
        swapped = False
        for cust_max in interior:
            if swapped:
                break
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_interior = final_routes[other_idx]
                if not other_interior:
                    continue
                for cust_other in other_interior:
                    new_routes = [list(r) for r in final_routes]
                    idx_max = new_routes[max_idx].index(cust_max)
                    idx_other = new_routes[other_idx].index(cust_other)
                    new_routes[max_idx][idx_max] = cust_other
                    new_routes[other_idx][idx_other] = cust_max
                    new_max = max_distance(new_routes)
                    if new_max < best_max - 1e-12:
                        final_routes = new_routes
                        report_best(final_routes)
                        swapped = True
                        break
                if swapped:
                    break
        if not swapped:
            break
    
    # Improvement: 2-opt on each route
    for i in range(truck_count):
        route = final_routes[i]
        if len(route) < 3:
            continue
        improved = True
        while improved:
            improved = False
            for a in range(len(route)-1):
                for b in range(a+2, len(route)+1):
                    if b - a < 2:
                        continue
                    new_route = route[:a] + route[a:b][::-1] + route[b:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        final_routes[i] = route
        report_best(final_routes)
    
    # Ensure best is stored
    if best_routes is not None:
        final_routes = best_routes
    
    # Build full routes with depots
    full_routes = []
    for r in final_routes:
        if len(r) == 0:
            full_routes.append([0, 0])
        else:
            full_routes.append([0] + r + [0])
    
    while len(full_routes) < truck_count:
        full_routes.append([0, 0])
    
    return full_routes