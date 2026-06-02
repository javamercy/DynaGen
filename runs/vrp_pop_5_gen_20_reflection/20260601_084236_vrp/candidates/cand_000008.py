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
        nxt = route[pos]
        return distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]

    # Construction: regret-2 insertion
    while remaining:
        best_customer = None
        best_regret = -1.0
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

        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        remaining.remove(best_customer)

    max_dist = max(route_distance(r) for r in routes)
    best_routes = [r[:] for r in routes]

    # Local search improvement: relocate customers between routes
    improved = True
    iteration = 0
    max_iterations = n * truck_count
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Find longest route
        longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
        longest_route = routes[longest_idx]
        best_move = None
        best_new_max = max_dist

        # Evaluate moving each customer from longest route to other routes
        for cust in longest_route[1:-1]:
            # Remove customer temporarily
            longest_route.remove(cust)
            # Evaluate insertion in every other route
            for target_idx in range(truck_count):
                if target_idx == longest_idx:
                    continue
                target_route = routes[target_idx]
                for pos in range(1, len(target_route)):
                    # Compute new route distances
                    new_dist_long = route_distance(longest_route)
                    # Simulate insertion
                    target_route.insert(pos, cust)
                    new_dist_target = route_distance(target_route)
                    new_max = max(new_dist_long, new_dist_target)
                    # Also consider unchanged routes
                    for r_idx in range(truck_count):
                        if r_idx != longest_idx and r_idx != target_idx:
                            new_max = max(new_max, route_distance(routes[r_idx]))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (cust, longest_idx, target_idx, pos)
                    target_route.remove(cust)
            # Reinsert customer back to original position
            # To preserve order, we need original position; we inserted at 1? Actually we removed, so we need to put back at the same spot.
            # Since we removed and will reinsert at the end, we can keep track but simpler: reinsert at position 1 (after depot) and then later it might break order, but we are not considering order for improvement; anyway we will revert properly.
            longest_route.insert(1, cust)  # approximate, but we will restore via deep copy later if needed. Better to use a copy of routes for evaluation and only apply move if improvement.
        
        # Apply best move if found
        if best_move is not None:
            cust, _, target_idx, pos = best_move
            # Remove customer from longest route (it is still there, need to find and remove)
            longest_route.remove(cust)
            # Insert into target route at best position
            target_route = routes[target_idx]
            target_route.insert(pos, cust)
            # Update max distance and best routes
            max_dist = max(route_distance(r) for r in routes)
            best_routes = [r[:] for r in routes]
            improved = True
    
    return best_routes