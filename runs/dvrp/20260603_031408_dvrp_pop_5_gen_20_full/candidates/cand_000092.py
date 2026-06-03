def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    import numpy as np
    
    if len(available_customers) == 0:
        return None
    
    # Current truck's immediate return distance
    current_return = np.linalg.norm(current_position - depot_position)
    
    # Max distance from any other truck to depot
    max_other = 0.0
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            dist = np.linalg.norm(depot_position - pos)
            if dist > max_other:
                max_other = dist
    
    current_max = max(current_return, max_other)
    
    best_idx = None
    best_new_max = float('inf')
    best_min_ot = -float('inf')
    
    for i, cust in enumerate(available_customers):
        # Distance from current truck to customer then to depot
        finish = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(finish, max_other)
        
        # Compute minimum distance from this customer to any other truck (for tie-breaking)
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        
        # Choose customer that minimizes new_max, tie-break by larger min_ot
        if new_max < best_new_max or (new_max == best_new_max and min_ot > best_min_ot):
            best_new_max = new_max
            best_min_ot = min_ot
            best_idx = i
    
    if best_idx is None:
        return None
    
    # Wait if serving any customer would increase the current max return time
    if best_new_max > current_max:
        return None
    
    return best_idx