def choose_next_customer(current_position, depot_position, truck_positions, available_customers, current_time):
    if len(available_customers) == 0:
        return None
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    mask = np.all(truck_positions == current_position, axis=1)
    current_idx = np.where(mask)[0][0]
    current_dist = dist_to_depot[current_idx]
    other_dists = np.delete(dist_to_depot, current_idx)
    other_max = np.max(other_dists) if len(other_dists) > 0 else 0

    best_idx = None
    best_max = np.inf
    best_candidate = np.inf
    for i, customer in enumerate(available_customers):
        candidate = np.linalg.norm(current_position - customer) + np.linalg.norm(customer - depot_position)
        max_time = max(candidate, other_max)
        if max_time < best_max:
            best_max = max_time
            best_candidate = candidate
            best_idx = i
        elif max_time == best_max and candidate < best_candidate:
            best_candidate = candidate
            best_idx = i
    return best_idx