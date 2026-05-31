import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    customers = list(range(1, n))
    # If truck_count >= n, assign each customer to its own route and fill empty
    if truck_count >= n:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Step 1: Initialize each customer as a separate route
    routes = [[0, i, 0] for i in customers]
    # For each route, maintain first and last customer (for savings computation)
    first_cust = {i: i for i in customers}
    last_cust = {i: i for i in customers}
    route_map = {i: i for i in customers}  # map customer index to route id (the customer itself initially)
    next_route_id = max(customers) + 1

    # Precompute savings for all pairs (i,j) where i<j
    savings_list = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings_list.append((s, i, j))
    savings_list.sort(key=lambda x: (-x[0], x[1], x[2]))

    # Merging loop
    while len(routes) > truck_count:
        best_savings = None
        best_pair = None
        best_orientation = None  # 0: connect last of first route to first of second; 1: opposite
        # Iterate over savings in order
        for s, i, j in savings_list:
            # Check if i and j are in different routes and are endpoints (first or last)
            if route_map[i] != route_map[j]:
                rid_i = route_map[i]
                rid_j = route_map[j]
                # Find endpoints for each route
                # For route rid_i, first and last are known
                first_i = first_cust[rid_i]
                last_i = last_cust[rid_i]
                first_j = first_cust[rid_j]
                last_j = last_cust[rid_j]
                # Check if i is either first or last of its route and j is either first or last of its route
                # Actually we need to know if connecting last_i to first_j or last_j to first_i
                # Compute savings for both orientations
                # Orientation 0: connect last_i to first_j
                s0 = distance_matrix[0, first_i] + distance_matrix[0, last_j] - distance_matrix[last_i, first_j]
                # Orientation 1: connect last_j to first_i
                s1 = distance_matrix[0, first_j] + distance_matrix[0, last_i] - distance_matrix[last_j, first_i]
                if s0 >= s1:
                    if best_savings is None or s0 > best_savings:
                        best_savings = s0
                        best_pair = (rid_i, rid_j, 0)
                else:
                    if best_savings is None or s1 > best_savings:
                        best_savings = s1
                        best_pair = (rid_i, rid_j, 1)
        if best_pair is None:
            break  # no further merges possible
        rid_i, rid_j, orientation = best_pair
        # Find the actual route objects
        route_i = None
        route_j = None
        for r in routes:
            if r[1] == first_cust[rid_i] and r[-2] == last_cust[rid_i]:
                route_i = r
            if r[1] == first_cust[rid_j] and r[-2] == last_cust[rid_j]:
                route_j = r
        if route_i is None or route_j is None:
            continue
        # Merge
        if orientation == 0:
            new_route = route_i[:-1] + route_j[1:]
        else:
            new_route = route_j[:-1] + route_i[1:]
        # Update first and last for new route
        new_id = next_route_id
        next_route_id += 1
        first_cust[new_id] = new_route[1]
        last_cust[new_id] = new_route[-2]
        # Update route_map for customers in new_route (excluding depots)
        for idx in range(1, len(new_route)-1):
            route_map[new_route[idx]] = new_id
        # Remove old routes and add new
        routes.remove(route_i)
        routes.remove(route_j)
        routes.append(new_route)
        # Remove old entries from first_cust, last_cust (optional)
        del first_cust[rid_i]
        del last_cust[rid_i]
        del first_cust[rid_j]
        del last_cust[rid_j]
        # Update savings_list? Not needed since we break after one merge per iteration
        break  # we only do one merge per outer loop iteration

    # After construction, we should have exactly truck_count routes; pad with empty if needed
    while len(routes) < truck_count:
        routes.append([0, 0])

    # Step 2: Inter-route 2-opt* improvement
    current_max = max(route_length(r) for r in routes)
    improved = True
    iteration = 0
    max_iter = n * truck_count
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        lengths = [route_length(r) for r in routes]
        current_max = max(lengths)
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 2 or len(route_j) <= 2:
                    continue
                # Try all cuts
                for cut_i in range(1, len(route_i)-1):
                    for cut_j in range(1, len(route_j)-1):
                        # Create new routes
                        new_route_i = route_i[:cut_i+1] + route_j[cut_j+1:]
                        new_route_j = route_j[:cut_j+1] + route_i[cut_i+1:]
                        new_len_i = route_length(new_route_i)
                        new_len_j = route_length(new_route_j)
                        # Compute new max
                        other_indices = [k for k in range(truck_count) if k not in (i,j)]
                        other_lengths = [lengths[k] for k in other_indices]
                        new_max = max(new_len_i, new_len_j, *other_lengths)
                        if new_max < current_max:
                            # Update routes
                            routes[i] = new_route_i
                            routes[j] = new_route_j
                            current_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return routes