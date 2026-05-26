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
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)

    # Fleet centroid
    centroid = np.mean(truck_positions, axis=0)
    dist_curr_centroid = np.linalg.norm(current_position - centroid)
    avg_dist_centroid = np.mean(np.linalg.norm(truck_positions - centroid, axis=1))

    if len(other_trucks) == 0:
        dep_coef = 0.3
    else:
        if dist_curr_centroid > 1.2 * avg_dist_centroid:
            dep_coef = 0.6
        else:
            dep_coef = 0.3

    score = dist_current - nearest_other + dep_coef * dist_depot

    # Waiting condition: current truck much farther than others from all customers
    if len(other_trucks) > 0:
        min_curr = np.min(dist_current)
        min_other = np.min(nearest_other)
        if min_curr > 1.5 * min_other and min_other != np.inf:
            return None

    best_idx = np.argmin(score)
    return int(best_idx)