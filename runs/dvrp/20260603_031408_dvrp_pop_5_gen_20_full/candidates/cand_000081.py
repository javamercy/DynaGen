def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # Compute distances
    d_current_depot = np.linalg.norm(current_position - depot_position)
    max_truck_depot = max(np.linalg.norm(pos - depot_position) for pos in truck_positions)
    
    best_idx = None
    best_score = -float('inf')
    best_dist_to_truck = None
    
    for i, cust in enumerate(available_customers):
        d_truck = np.linalg.norm(current_position - cust)
        d_depot = np.linalg.norm(depot_position - cust)
        # Compute min distance to any other truck (exclude current truck's position? Wait, truck_positions includes all trucks, including current? We need to exclude current position. Since current_position matches one of truck_positions, we can identify that truck's index.
        # But we don't have index. To be safe, compute min over all trucks but skip the one closest to current_position (which is likely the same). However, if two trucks are at same spot, it's fine. 
        # Alternative: assume current_position is the first truck? No, it's arbitrary. Use all truck positions and subtract a small epsilon? Simpler: compute min distance to any other truck by considering all except the one that matches current_position exactly.
        # Since floating point, compare with tolerance.
        mask = np.any(np.abs(truck_positions - current_position) > 1e-9, axis=1)
        other_positions = truck_positions[mask]
        if len(other_positions) == 0:
            d_other = 0.0
        else:
            d_other = np.min(np.linalg.norm(other_positions - cust, axis=1))
        score = d_depot - d_truck - 0.5 * d_other
        if score > best_score:
            best_score = score
            best_idx = i
            best_dist_to_truck = d_truck
    
    # Waiting condition
    if d_current_depot < 0.2 * max_truck_depot and best_dist_to_truck > 2 * d_current_depot:
        return None
    
    return best_idx