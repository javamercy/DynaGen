def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Find index of current truck
    current_idx = None
    for idx, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            current_idx = idx
            break
    if current_idx is None:
        # fallback: treat as an unknown truck; assume we can still compute
        current_idx = -1
    # Compute distances to depot for all trucks
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Current truck's distance to depot
    cur_dist_depot = dist_to_depot[current_idx] if current_idx != -1 else np.linalg.norm(current_position - depot_position)
    # Max distance to depot among other trucks
    other_mask = np.ones(len(truck_positions), dtype=bool)
    if current_idx != -1:
        other_mask[current_idx] = False
    if np.any(other_mask):
        max_other = np.max(dist_to_depot[other_mask])
    else:
        # no other trucks? shouldn't happen, but handle
        max_other = -np.inf
    # Current worst-case (if we do nothing)
    current_max = max(cur_dist_depot, max_other)
    best_idx = None
    best_max = np.inf
    for i, cust in enumerate(available_customers):
        d_cur_cust = np.linalg.norm(current_position - cust)
        d_cust_depot = np.linalg.norm(cust - depot_position)
        round_trip = d_cur_cust + d_cust_depot
        candidate_max = max(round_trip, max_other)
        if candidate_max < best_max:
            best_max = candidate_max
            best_idx = i
    if best_max < current_max:
        return best_idx
    else:
        return None