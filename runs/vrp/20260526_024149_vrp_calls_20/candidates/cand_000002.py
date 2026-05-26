import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize routes: each truck starts and ends at depot
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count
    # List of unrouted customers (nodes 1..n-1)
    unrouted = list(range(1, n))
    # Helper function to compute cost of inserting a customer at a position in a route
    def insertion_cost(route, pos, customer):
        prev = route[pos]
        next_node = route[pos+1]
        return distance_matrix[prev, customer] + distance_matrix[customer, next_node] - distance_matrix[prev, next_node]
    # Greedy insertion: for each customer, find cheapest insertion across all routes
    while unrouted:
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_customer = -1
        for customer in unrouted:
            for r_idx, route in enumerate(routes):
                # Insert at any position between 0 and len(route)-1
                for pos in range(len(route)-1):
                    cost = insertion_cost(route, pos, customer)
                    if cost < best_cost or (cost == best_cost and r_idx < best_route_idx):
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
                        best_customer = customer
        # Insert best customer
        route = routes[best_route_idx]
        route.insert(best_pos+1, best_customer)
        route_lengths[best_route_idx] += best_cost
        unrouted.remove(best_customer)
    # Compute max route distance
    max_dist = max(route_lengths)
    best_routes = [list(r) for r in routes]
    best_max = max_dist
    report_best_vrp(best_routes)
    # Local search: for each customer, try to relocate to another route to reduce max distance
    # Limit iterations to avoid infinite loops
    max_iter = 2 * n
    for _ in range(max_iter):
        improved = False
        for cust in range(1, n):
            # Find current route of cust
            curr_route_idx = -1
            curr_pos = -1
            for r_idx, route in enumerate(routes):
                if cust in route:
                    curr_route_idx = r_idx
                    curr_pos = route.index(cust)
                    break
            if curr_route_idx == -1:
                continue
            curr_route = routes[curr_route_idx]
            # Evaluate removing cust from current route
            prev_node = curr_route[curr_pos-1]
            next_node = curr_route[curr_pos+1]
            removal_saving = distance_matrix[prev_node, cust] + distance_matrix[cust, next_node] - distance_matrix[prev_node, next_node]
            new_curr_len = route_lengths[curr_route_idx] - removal_saving
            # Try inserting into other routes or same route at different position?
            # For simplicity, try all routes including same route at different positions
            for r_idx, route in enumerate(routes):
                if r_idx == curr_route_idx:
                    # Try relocating within same route to different position
                    # But that might not help max distance unless it reduces length?
                    # Skip to avoid cycles
                    continue
                for pos in range(len(route)-1):
                    if route[pos] == cust or route[pos+1] == cust:
                        continue  # Skip if already adjacent
                    insertion_cost_val = distance_matrix[route[pos], cust] + distance_matrix[cust, route[pos+1]] - distance_matrix[route[pos], route[pos+1]]
                    new_len = route_lengths[r_idx] + insertion_cost_val
                    # Compute new max distance
                    new_max = max(new_curr_len, new_len, max(route_lengths[:r_idx] + route_lengths[r_idx+1:] if r_idx != curr_route_idx else []))
                    # Also consider route_lengths of other routes unchanged except those two
                    # Simplify: compute max of all route lengths after move
                    # Build list of lengths
                    new_lengths = route_lengths.copy()
                    new_lengths[curr_route_idx] = new_curr_len
                    new_lengths[r_idx] = new_len
                    new_max = max(new_lengths)
                    if new_max < best_max - 1e-9:  # improvement
                        # Perform move
                        # Remove cust from current route
                        curr_route.pop(curr_pos)
                        route_lengths[curr_route_idx] = new_curr_len
                        # Insert into target route
                        route.insert(pos+1, cust)
                        route_lengths[r_idx] = new_len
                        # Update best
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        improved = True
                        report_best_vrp(best_routes)
                        break  # break pos loop
                if improved:
                    break  # break route loop
            if improved:
                break  # break cust loop
        if not improved:
            break
    return best_routes