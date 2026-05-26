def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    d_curr = np.linalg.norm(available_customers - current_position, axis=1)
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        # Waiting condition: if current truck's best customer is much farther than other trucks' best
        min_curr = np.min(d_curr)
        best_other = np.min(nearest_other)
        if min_curr > best_other * 1.5:
            return None
    # Adaptive depot coefficient
    depot_coeff = 0.3
    if len(truck_positions) > 1:
        centroid = np.mean(truck_positions, axis=0)
        dist_to_centroid = np.linalg.norm(current_position - centroid)
        avg_dist = np.mean(np.linalg.norm(truck_positions - centroid))
        if avg_dist > 0 and dist_to_centroid > avg_dist * 1.2:
            depot_coeff = 0.6
    score = d_curr - nearest_other + depot_coeff * d_depot
    best_idx = np.argmin(score)
    return int(best_idx)