def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Exclude current truck from other trucks
    other_mask = ~np.all(truck_positions == current_position, axis=1)
    other_positions = truck_positions[other_mask]
    if len(other_positions) == 0:
        min_other_depot_dist = float('inf')
    else:
        min_other_depot_dist = np.min(np.linalg.norm(depot_position - other_positions, axis=1))
    best_idx = None
    best_score = -float('inf')
    best_cust = None
    for i, cust in enumerate(available_customers):
        cust_to_depot = np.linalg.norm(depot_position - cust)
        cust_to_truck = np.linalg.norm(current_position - cust)
        if len(other_positions) == 0:
            min_ot = 0.0
        else:
            dists_to_other = np.linalg.norm(other_positions - cust, axis=1)
            min_ot = np.min(dists_to_other)
        score = cust_to_depot - cust_to_truck - 1.0 * min_ot
        if score > best_score:
            best_score = score
            best_idx = i
            best_cust = cust
    # Waiting conditions
    if best_cust is not None:
        estimated_return = np.linalg.norm(current_position - best_cust) + np.linalg.norm(best_cust - depot_position)
        if min_other_depot_dist < float('inf') and estimated_return > 1.5 * min_other_depot_dist:
            return None
        if best_score < 0:
            return None
    return best_idx