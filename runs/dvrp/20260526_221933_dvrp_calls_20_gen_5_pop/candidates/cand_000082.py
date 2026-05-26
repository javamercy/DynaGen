def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        nearest_other = np.zeros(len(available_customers))
        wait = False
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        # waiting condition
        min_dist_current = np.min(dist_current)
        other_min_to_cust = np.min(np.min(dist_other, axis=0))  # min per other truck then min
        if min_dist_current > 1.5 * other_min_to_cust:
            wait = True
        else:
            wait = False
    if wait:
        return None
    # dynamic depot coefficient
    centroid = np.mean(truck_positions, axis=0)
    dists_to_centroid = np.linalg.norm(truck_positions - centroid, axis=1)
    mean_dist_to_centroid = np.mean(dists_to_centroid)
    current_dist_to_centroid = np.linalg.norm(current_position - centroid)
    ratio = current_dist_to_centroid / (mean_dist_to_centroid + 1e-8)
    depot_coeff = 0.3 * max(1.0, ratio)
    score = dist_current - nearest_other + depot_coeff * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)