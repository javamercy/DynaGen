def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    num_trucks = len(truck_positions)
    distances_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_dist = np.linalg.norm(current_position - depot_position)
    # find active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    if num_trucks > 1:
        other_dists = np.delete(distances_to_depot, active_idx)
        M_other = np.max(other_dists)
    else:
        M_other = -np.inf  # single truck: serves always
    M_current = np.max(distances_to_depot)
    best_idx = None
    best_candidate = np.inf
    for i, cust in enumerate(available_customers):
        R_active = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        candidate = max(R_active, M_other)
        if candidate < best_candidate:
            best_candidate = candidate
            best_idx = i
    if num_trucks > 1 and best_candidate > M_current:
        return None
    else:
        return best_idx