import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Build giant tour via deterministic nearest neighbor starting from customer 1
    tour = []
    unvisited = set(customers)
    if not unvisited:
        # No customers, return empty routes
        return [[0, 0] for _ in range(truck_count)]
    # Start at customer 1 (if exists)
    current = 0
    while unvisited:
        # Find nearest unvisited customer
        best_dist = float('inf')
        best_cust = -1
        for c in unvisited:
            d = distance_matrix[current, c]
            if d < best_dist or (d == best_dist and c < best_cust):
                best_dist = d
                best_cust = c
        tour.append(best_cust)
        unvisited.remove(best_cust)
        current = best_cust
    
    # Precompute distances for segments of the tour
    # We'll work with the sequence of customers
    # Helper: compute route distance for a list of customers in order, including depot at both ends
    def route_dist(cust_list):
        if not cust_list:
            return 0.0
        d = distance_matrix[0, cust_list[0]]
        for i in range(len(cust_list)-1):
            d += distance_matrix[cust_list[i], cust_list[i+1]]
        d += distance_matrix[cust_list[-1], 0]
        return d
    
    # Compute upper bound: full tour distance (all customers in one route)
    full_dist = route_dist(tour)
    # Binary search on max route distance
    low = 0.0
    high = full_dist
    # Tolerance for floating point
    eps = 1e-9
    
    # Feasibility check: can we split tour into at most truck_count segments each with distance <= L?
    def feasible(L):
        seg_count = 0
        i = 0
        while i < len(tour):
            seg_count += 1
            if seg_count > truck_count:
                return False
            # Start new segment
            j = i
            # Accumulate distance from depot to first customer
            first = tour[j]
            seg_dist = distance_matrix[0, first]
            # Add customers until exceeding L
            while j < len(tour):
                # If we have more than one customer, add distance from previous to next
                if j == i:
                    # first customer already accounted
                    pass
                else:
                    prev = tour[j-1]
                    curr = tour[j]
                    add = distance_matrix[prev, curr]
                    if seg_dist + add > L + eps:
                        break
                    seg_dist += add
                # Check if going back to depot from this customer would exceed L
                back_to_depot = distance_matrix[tour[j], 0]
                if seg_dist + back_to_depot > L + eps:
                    # This customer alone is too much, but if it's the first, then impossible
                    if j == i:
                        return False
                    else:
                        # Do not include this customer
                        break
                j += 1
            # If we broke because j == i, it means a single customer segment exceeds L => infeasible
            if j == i:
                return False
            i = j
        return seg_count <= truck_count
    
    # Binary search for minimal L
    for _ in range(60):  # 60 iterations for double precision
        mid = (low + high) / 2
        if feasible(mid):
            high = mid
        else:
            low = mid
    L_min = high
    
    # Reconstruct routes using L_min greedy
    routes = []
    i = 0
    while i < len(tour):
        # Start new segment
        j = i
        first = tour[j]
        seg_dist = distance_matrix[0, first]
        while j < len(tour):
            if j == i:
                # first customer already added
                pass
            else:
                prev = tour[j-1]
                curr = tour[j]
                add = distance_matrix[prev, curr]
                if seg_dist + add > L_min + eps:
                    break
                seg_dist += add
            back = distance_matrix[tour[j], 0]
            if seg_dist + back > L_min + eps:
                if j == i:
                    break  # but this shouldn't happen because L_min is feasible
                else:
                    break
            j += 1
        segment = tour[i:j]
        route = [0] + segment + [0]
        routes.append(route)
        i = j
    # Pad with empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])
    # Ensure exactly truck_count
    routes = routes[:truck_count]
    
    # Call report_best_vrp (harness function, assumed available)
    report_best_vrp(routes)
    return routes