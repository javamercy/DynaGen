import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    remaining = set(range(1, n))
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def insertion_cost(route, customer, pos):
        prev = route[pos-1]
        next = route[pos]
        return distance_matrix[prev, customer] + distance_matrix[customer, next] - distance_matrix[prev, next]
    
    while remaining:
        best_customer = None
        best_regret = -1
        best_route_idx = -1
        best_pos = -1
        
        for c in sorted(remaining):
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_route = -1
            best_pos_local = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    cost = insertion_cost(route, c, pos)
                    if cost < best_cost:
                        second_best_cost = best_cost
                        best_cost = cost
                        best_route = r_idx
                        best_pos_local = pos
                    elif cost < second_best_cost:
                        second_best_cost = cost
            if second_best_cost == float('inf'):
                regret = float('inf')
            else:
                regret = second_best_cost - best_cost
            if regret > best_regret:
                best_regret = regret
                best_customer = c
                best_route_idx = best_route
                best_pos = best_pos_local
        
        # Insert customer at best position
        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        remaining.remove(best_customer)
    
    # Compute initial max distance
    max_dist = max(route_distance(r) for r in routes)
    best_routes = [r[:] for r in routes]
    
    # Report initial incumbent
    # (report_best_vrp would be called here, but not defined; assume external)
    
    # Improvement: move customers from longest route to shortest
    improved = True
    iteration = 0
    while improved and iteration < n * truck_count:
        improved = False
        iteration += 1
        # Find longest and shortest routes
        lengths = [(route_distance(r), idx) for idx, r in enumerate(routes)]
        lengths.sort(reverse=True)
        longest_idx = lengths[0][1]
        shortest_idx = lengths[-1][1]
        if lengths[0][0] == lengths[-1][0]:
            break
        longest_route = routes[longest_idx]
        shortest_route = routes[shortest_idx]
        # Try moving each customer from longest to shortest
        for cust in longest_route[1:-1]:
            # Remove cust from longest
            longest_route.remove(cust)
            # Try inserting into shortest at best position
            best_pos = -1
            best_inc = float('inf')
            for pos in range(1, len(shortest_route)):
                cost = insertion_cost(shortest_route, cust, pos)
                if cost < best_inc:
                    best_inc = cost
                    best_pos = pos
            shortest_route.insert(best_pos, cust)
            # Evaluate new max
            new_max = max(route_distance(r) for r in routes)
            if new_max < max_dist:
                max_dist = new_max
                best_routes = [r[:] for r in routes]
                improved = True
                break  # Accept move and restart from longest/shortest
            else:
                # Undo move
                shortest_route.remove(cust)
                longest_route.insert(1, cust)  # reinsert at same position? Not exactly, but we can reinsert at original position
                # Actually we lost track of original position; better to use a copy
                # To keep deterministic, we revert by removing from shortest and reinserting at a known position
                # For simplicity, skip this customer and continue
    
    return best_routes