def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    active_dist = np.linalg.norm(current_position - depot_position)
    other_dists = [np.linalg.norm(pos - depot_position) for pos in truck_positions if not np.allclose(pos, current_position)]
    max_other = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_new_max = float('inf')
    for i, cust in enumerate(available_customers):
        new_active = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(new_active, max_other)
        if new_max < best_new_max:
            best_new_max = new_max
            best_idx = i
    return best_idx