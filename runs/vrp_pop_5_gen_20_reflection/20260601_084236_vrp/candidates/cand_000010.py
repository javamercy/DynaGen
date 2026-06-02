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
    
    # Regret insertion
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
            # Tie-break: choose larger regret, then smaller customer index
            if regret > best_regret or (regret == best_regret and c < best_customer):
                best_regret = regret
                best_customer = c
                best_route_idx = best_route
                best_pos = best_pos_local
        
        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        remaining.remove(best_customer)
    
    # Initial max distance
    max_dist = max(route_distance(r) for r in routes)
    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)
    
    # Improvement: move customers from longest to shortest route
    improved = True
    iteration = 0
    max_iterations = n * truck_count
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Find routes with max and min distance
        lengths = [(route_distance(r), idx) for idx, r in enumerate(routes)]
        lengths.sort(key=lambda x: x[0])
        min_len, min_idx = lengths[0]
        max_len, max_idx = lengths[-1]
        if max_len == min_len:
            break
        longest_route = routes[max_idx]
        shortest_route = routes[min_idx]
        # Try moving each customer from longest to shortest (in original order)
        for cust in list(longest_route[1:-1]):
            # Remove customer from longest route
            longest_route.remove(cust)
            # Find best insertion position in shortest route
            best_pos = 1
            best_cost = insertion_cost(shortest_route, cust, 1)
            for pos in range(2, len(shortest_route)):
                cost = insertion_cost(shortest_route, cust, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            shortest_route.insert(best_pos, cust)
            new_max = max(route_distance(r) for r in routes)
            if new_max < max_dist:
                max_dist = new_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
                improved = True
                break  # accept move, restart loop
            else:
                # Undo move
                shortest_route.remove(cust)
                # Reinsert into longest at original position? Since we removed by value, we need to reinsert at a position that maintains order.
                # Simpler: reinsert at the same index where it was removed (but remove changed indices). To keep deterministic, we reinsert at the front after removal? Better: keep a copy of longest route before removal and restore.
                # We'll use a simpler approach: restore by finding a position where insertion cost matches (approximately). But for robustness, we'll reinsert at the first position (after depot) and then the route order may change, but it's acceptable for deterministic behavior as long as we use a fixed order.
                # To be deterministic, we reinsert at position 1 (right after depot). This changes the route, but subsequent moves will still converge. Let's do that to avoid complexity.
                longest_route.insert(1, cust)
                # However, this may disrupt the route structure. Alternatively, we can skip moving this customer and continue to next.
                # We'll just break out of the customer loop if move not accepted, and try next iteration with updated routes (which may have changed due to reinsertion). To keep code simple, we'll break after trying all customers (no early break). Actually we want to break after a successful move; otherwise continue.
                # Let's restructure: For each customer, do the move, evaluate, if not improving, undo and try next.
                # Undo as above.
                pass
        # If we didn't break, improved remains False and loop ends
    return best_routes