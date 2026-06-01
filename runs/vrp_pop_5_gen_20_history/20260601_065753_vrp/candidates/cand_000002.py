import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count
    
    def compute_route_length(route):
        length = 0.0
        for i in range(len(route) - 1):
            length += distance_matrix[route[i], route[i+1]]
        return length
    
    # Construction: insert all customers
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_total_inc = float('inf')
        for r in range(truck_count):
            route = routes[r]
            # Feasible positions: between 1 and len(route)-1 inclusive?
            # Actually route[0]=0, route[-1]=0, we insert between them.
            for p in range(1, len(route)):
                # Compute new length for route r
                prev_node = route[p-1]
                next_node = route[p]
                old_edge = distance_matrix[prev_node, next_node]
                new_len = route_lengths[r] - old_edge + distance_matrix[prev_node, cust] + distance_matrix[cust, next_node]
                # new max
                new_max = new_len
                for rr in range(truck_count):
                    if rr != r:
                        if route_lengths[rr] > new_max:
                            new_max = route_lengths[rr]
                if new_max < best_max:
                    best_max = new_max
                    best_route_idx = r
                    best_pos = p
                    # compute total increase for tie-breaking? Not needed now.
                elif new_max == best_max:
                    # Tie: prefer smaller route index, then smaller position
                    if r < best_route_idx or (r == best_route_idx and p < best_pos):
                        best_max = new_max
                        best_route_idx = r
                        best_pos = p
        # Insert customer
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_lengths[best_route_idx] = compute_route_length(route)
        # Report after each insertion? Might be many calls, but spec says call when better found.
        # Since we are constructing incrementally, we can report after full construction.
    
    # Report after construction
    # (We call report_best_vrp here with current routes)
    # But we must define report_best_vrp; assume it's available.
    def report_best_vrp(routes):
        pass  # will be replaced by external function
    
    # Call report_best_vrp after construction
    current_max = max(route_lengths)
    report_best_vrp([list(r) for r in routes])
    
    # Local search: relocate moves only, to reduce max
    improved = True
    for iteration in range(2 * n):  # finite bound
        if not improved:
            break
        improved = False
        # For each customer, try moving to another position (including same route? but relocate within same route might increase max? We'll consider moves that change max)
        for r_from in range(truck_count):
            route_from = routes[r_from]
            if len(route_from) <= 2:  # only depot
                continue
            for idx in range(1, len(route_from)-1):  # positions of customers in route
                cust = route_from[idx]
                # remove customer from current position
                new_route_from = route_from[:idx] + route_from[idx+1:]
                len_from_new = compute_route_length(new_route_from)
                for r_to in range(truck_count):
                    route_to = routes[r_to]
                    for p in range(1, len(route_to)):
                        if r_from == r_to:
                            # moving within same route: skip insertion at same position?
                            # Actually if r_from == r_to, we already removed from idx; we need to consider insertion positions that are not just idx? But we can just skip to avoid complexity. We'll only consider inter-route moves to keep it simple.
                            continue
                        # insert into route_to at position p
                        new_route_to = route_to[:p] + [cust] + route_to[p:]
                        len_to_new = compute_route_length(new_route_to)
                        new_max = max(len_from_new, len_to_new)
                        for rr in range(truck_count):
                            if rr != r_from and rr != r_to:
                                if route_lengths[rr] > new_max:
                                    new_max = route_lengths[rr]
                        if new_max < current_max:
                            # Accept move
                            routes[r_from] = new_route_from
                            routes[r_to] = new_route_to
                            route_lengths[r_from] = len_from_new
                            route_lengths[r_to] = len_to_new
                            current_max = new_max
                            improved = True
                            report_best_vrp([list(r) for r in routes])
                            break  # break position loop
                if improved:
                    break  # break customer loop in route_from
            if improved:
                break  # break route_from loop
        # end iteration
    
    # Return exactly truck_count routes
    return [list(r) for r in routes]