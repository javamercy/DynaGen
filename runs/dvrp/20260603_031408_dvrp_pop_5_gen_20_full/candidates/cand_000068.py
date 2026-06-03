def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Determine active truck index
    active_idx = None
    for idx, pos in enumerate(truck_positions):
        if np.allclose(pos, current_position):
            active_idx = idx
            break
    if active_idx is None:
        # fallback: treat as first truck? Should not happen.
        active_idx = 0
    # Distances from trucks to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_active_dist = truck_depot_dists[active_idx]
    # Max distance of other trucks
    other_max = np.max(np.delete(truck_depot_dists, active_idx)) if len(truck_depot_dists) > 1 else -np.inf
    best_idx = None
    best_max = float('inf')
    best_new_active = float('inf')
    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        cust_to_depot = np.linalg.norm(cust - depot_position)
        new_active_dist = dist_to_cust + cust_to_depot
        new_max = max(other_max, new_active_dist)
        # Prefer smaller new_max, then smaller new_active_dist
        if new_max < best_max or (new_max == best_max and new_active_dist < best_new_active):
            best_max = new_max
            best_new_active = new_active_dist
            best_idx = i
    return best_idx