import numpy as np
from typing import List

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    # Initialize routes: each route starts and ends at depot (0)
    routes = [[0, 0] for _ in range(truck_count)]
    unrouted = set(range(1, n))
    
    # Helper to compute distance of a route
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Insertion heuristic: insert all customers
    for customer in sorted(unrouted):  # deterministic order
        best_increase = float('inf')
        best_route_idx = None
        best_pos = None
        for r_idx, route in enumerate(routes):
            # Positions from 1 to len(route)-1 (after first depot, before last depot)
            for pos in range(1, len(route)):
                prev = route[pos-1]
                next_node = route[pos]
                increase = (distance_matrix[prev, customer] + 
                            distance_matrix[customer, next_node] - 
                            distance_matrix[prev, next_node])
                if increase < best_increase or (increase == best_increase and 
                    (best_route_idx is None or r_idx < best_route_idx)):
                    best_increase = increase
                    best_route_idx = r_idx
                    best_pos = pos
        # Insert customer into best position
        routes[best_route_idx].insert(best_pos, customer)
    
    # Compute initial max distance
    max_dist = max(route_distance(r) for r in routes)
    best_routes = [r[:] for r in routes]
    best_max = max_dist
    
    # Improvement: relocate moves from longest route
    max_iter = n * truck_count
    for _ in range(max_iter):
        # Find longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(truck_count), key=lambda i: dists[i])
        longest_route = routes[max_idx]
        # Try to move each customer (except depot) from longest route to other routes
        improved = False
        # iterate over a copy of internal nodes (1..len-1)
        for idx in range(1, len(longest_route)-1):
            customer = longest_route[idx]
            # Remove customer temporarily
            old_dist = dists[max_idx]
            new_route = longest_route[:idx] + longest_route[idx+1:]
            new_dist_no_customer = route_distance(new_route)
            # Try inserting into other routes or same route at different position? Only other routes to balance
            best_increase = float('inf')
            best_route_idx = None
            best_pos = None
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next_node = route[pos]
                    increase = (distance_matrix[prev, customer] + 
                                distance_matrix[customer, next_node] - 
                                distance_matrix[prev, next_node])
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = r_idx
                        best_pos = pos
            if best_route_idx is None:
                continue
            # Evaluate new max distance if we move
            new_longest_dist = new_dist_no_customer  # after removal
            new_target_route = routes[best_route_idx][:]
            new_target_route.insert(best_pos, customer)
            new_target_dist = route_distance(new_target_route)
            new_max = max(new_longest_dist, new_target_dist, 
                          max(dists[:max_idx] + dists[max_idx+1:best_route_idx] + 
                              dists[best_route_idx+1:]))
            if new_max < best_max:
                # Accept move
                routes[max_idx] = new_route
                routes[best_route_idx] = new_target_route
                # Update best
                best_max = new_max
                best_routes = [r[:] for r in routes]
                improved = True
                break  # restart after one successful move
        if not improved:
            break
    
    # If after improvement we have a better solution, report it
    # (already updated best_routes during improvement, but ensure we call report_best_vrp if available)
    # Note: report_best_vrp is expected to be defined externally; we call it if it exists
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    return best_routes