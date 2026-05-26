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
    score = dist_current - nearest_other + 0.5 * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)