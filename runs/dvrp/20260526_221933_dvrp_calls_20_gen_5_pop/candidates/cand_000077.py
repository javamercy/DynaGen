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
    # Identify current truck index
    idx_current = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    dist_all_to_centroid = np.linalg.norm(truck_positions - centroid, axis=1)
    dist_cur_centroid = dist_all_to_centroid[idx_current]
    if np.any(mask):
        mean_other_centroid = np.mean(dist_all_to_centroid[mask])
        if dist_cur_centroid > mean_other_centroid:
            depot_coeff = 0.5
        else:
            depot_coeff = 0.3
    else:
        depot_coeff = 0.3
    # Waiting condition: if current truck is much farther than others from all customers
    if np.any(mask):
        if np.all(dist_current > 1.5 * nearest_other):
            return None
    score = dist_current - nearest_other + depot_coeff * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)