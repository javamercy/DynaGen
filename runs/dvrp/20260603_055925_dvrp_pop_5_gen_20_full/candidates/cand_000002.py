def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        # Only one truck: go to nearest customer
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    
    # Find index of current truck
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    best_regret = -float('inf')
    best_idx = None
    best_secondary = None
    for i, cust in enumerate(available_customers):
        current_dist = np.linalg.norm(current_position - cust)
        other_dists = []
        for j, pos in enumerate(truck_positions):
            if j == current_truck_idx:
                continue
            other_dists.append(np.linalg.norm(pos - cust))
        min_other = min(other_dists) if other_dists else float('inf')
        regret = min_other - current_dist
        # Secondary tie-breaker: smaller current distance is better
        secondary = -current_dist
        if (regret > best_regret) or (regret == best_regret and secondary > best_secondary):
            best_regret = regret
            best_idx = i
            best_secondary = secondary
    if best_regret >= 0:
        return best_idx
    else:
        return None