import numpy as np
import itertools

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count
    
    # Sequential insertion in index order
    for cust in customers:
        best_new_max = float('inf')
        best_new_route_dist = float('inf')
        best_route_idx = None
        best_pos = None
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                old_dist = route_distances[r_idx]
                removed = distance_matrix[route[pos-1], route[pos]]
                added = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]]
                new_dist = old_dist - removed + added
                # Compute new max
                other_max = max(route_distances[j] for j in range(truck_count) if j != r_idx) if truck_count > 1 else 0.0
                new_max = max(new_dist, other_max)
                if new_max < best_new_max or (new_max == best_new_max and new_dist < best_new_route_dist):
                    best_new_max = new_max
                    best_new_route_dist = new_dist
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx].insert(best_pos, cust)
        route_distances[best_route_idx] = best_new_route_dist
    
    best_routes = [route[:] for route in routes]
    best_max = max(route_distances)
    report_best_vrp(best_routes)
    
    # Lightweight local search: try to reduce max distance
    # Use a fixed number of iterations (e.g., n * truck_count)
    max_iter = n * truck_count
    for iteration in range(max_iter):
        improved = False
        # First, find the index of the route with maximum distance (if tie, smallest index)
        max_val = max(route_distances)
        if max_val == 0:
            break
        # Indices of routes with max distance (if multiple, pick first)
        max_idx = route_distances.index(max_val)
        # Intra-route 2-opt on the longest route
        route = routes[max_idx]
        best_route = route[:]
        best_dist = route_distances[max_idx]
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                # reverse segment i..j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                # compute new distance
                new_dist = 0.0
                for k in range(len(new_route)-1):
                    new_dist += distance_matrix[new_route[k], new_route[k+1]]
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_route = new_route
        if best_dist < route_distances[max_idx]:
            routes[max_idx] = best_route
            route_distances[max_idx] = best_dist
            new_max = max(route_distances)
            if new_max < best_max:
                best_max = new_max
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)
            improved = True
        
        # Inter-route relocate: try to move a customer from the longest route to another route
        # Iterate over customers in longest route (excluding start/end depots)
        for cust_pos in range(1, len(routes[max_idx]) - 1):
            cust = routes[max_idx][cust_pos]
            # Remove customer from current route temporarily
            temp_route = routes[max_idx][:cust_pos] + routes[max_idx][cust_pos+1:]
            temp_dist = 0.0
            for k in range(len(temp_route)-1):
                temp_dist += distance_matrix[temp_route[k], temp_route[k+1]]
            # For each other route, find best insertion position
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                other_route = routes[r_idx]
                for pos in range(1, len(other_route)):
                    old_other_dist = route_distances[r_idx]
                    removed = distance_matrix[other_route[pos-1], other_route[pos]]
                    added = distance_matrix[other_route[pos-1], cust] + distance_matrix[cust, other_route[pos]]
                    new_other_dist = old_other_dist - removed + added
                    new_max_candidate = max(temp_dist, new_other_dist, max(route_distances[j] for j in range(truck_count) if j != max_idx and j != r_idx))
                    if new_max_candidate < best_max:
                        # Apply move
                        routes[max_idx] = temp_route
                        route_distances[max_idx] = temp_dist
                        routes[r_idx].insert(pos, cust)
                        route_distances[r_idx] = new_other_dist
                        # Update best
                        new_max = max(route_distances)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [route[:] for route in routes]
                            report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        else:
            break  # no improvement in this iteration
    
    return best_routes