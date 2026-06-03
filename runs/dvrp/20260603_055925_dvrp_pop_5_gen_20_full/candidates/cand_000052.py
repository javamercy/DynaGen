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
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    best_value = -float('inf')
    best_idx = None
    for i, cust in enumerate(available_customers):
        current_dist = np.linalg.norm(current_position - cust)
        cust_depot_dist = np.linalg.norm(cust - depot_position)
        extra = (current_dist + cust_depot_dist) - current_depot_dist
        other_dists = []
        for j, pos in enumerate(truck_positions):
            if j == current_truck_idx:
                continue
            other_dists.append(np.linalg.norm(pos - cust))
        min_other = min(other_dists) if other_dists else float('inf')
        regret = min_other - current_dist - extra  # penalty for extra distance
        if regret > best_value:
            best_value = regret
            best_idx = i
    if best_value >= 0:
        return best_idx
    else:
        return None