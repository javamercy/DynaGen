def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Compute distances
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # Compute distance to nearest other truck for each customer
    # Create mask for current truck (assuming current_position is one of truck_positions)
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        # Compute pairwise distances between each customer and each other truck
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
    # Composite score: smaller is better
    # Weights: alpha=0.5 (depot), beta=0.3 (other truck proximity)
    score = dist_current + 0.5 * dist_depot - 0.3 * nearest_other
    best_idx = np.argmin(score)
    return int(best_idx)