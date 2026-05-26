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
    # other trucks
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
        avg_depot_other = 0.0
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
        avg_depot_other = np.mean(other_depot_dists)
    # balance term: penalize if customer depot distance > avg other truck depot distance
    balance_penalty = np.maximum(0, dist_depot - avg_depot_other)
    score = dist_current + 2.0 * dist_depot - 0.3 * nearest_other + 0.5 * balance_penalty
    best_idx = np.argmin(score)
    return int(best_idx)