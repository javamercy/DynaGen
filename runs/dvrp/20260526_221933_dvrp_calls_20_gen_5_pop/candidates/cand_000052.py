def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Distance from current truck to each customer
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    # Distance from each customer to depot
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # Distance from each customer to nearest other truck
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
    # Composite score: smaller is better
    score = dist_current - nearest_other + 0.6 * dist_depot
    # Waiting condition: if current truck is much farther than others from nearest customer
    if len(other_trucks) > 0:
        # For current truck, distance to its nearest customer
        current_nearest = np.min(dist_current)
        # For each other truck, distance to its nearest available customer
        other_min_dists = []
        for ot in other_trucks:
            dist_to_cust = np.linalg.norm(available_customers - ot, axis=1)
            if len(dist_to_cust) > 0:
                other_min_dists.append(np.min(dist_to_cust))
        if len(other_min_dists) > 1:
            avg_other_min = np.mean(other_min_dists)
            if current_nearest > 1.5 * avg_other_min:
                return None
    best_idx = np.argmin(score)
    return int(best_idx)