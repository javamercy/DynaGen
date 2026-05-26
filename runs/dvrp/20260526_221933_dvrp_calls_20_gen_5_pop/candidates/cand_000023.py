def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Identify current truck index by closest position
    dist_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_trucks)
    # Other trucks mask
    mask = np.ones(truck_positions.shape[0], dtype=bool)
    mask[current_idx] = False
    other_trucks = truck_positions[mask]
    # Compute distances
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
    # Composite score: smaller is better
    score = dist_current + 0.7 * dist_depot - 0.3 * nearest_other
    best_idx = np.argmin(score)
    best_score = score[best_idx]
    # Dynamic threshold based on average distance to depot of remaining customers
    mean_dist_depot = np.mean(dist_depot)
    threshold = 0.6 * mean_dist_depot
    if best_score > threshold:
        return None
    return int(best_idx)