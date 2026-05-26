def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # distances from current truck
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    # distances to depot
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        # single truck: minimize return time
        best = np.argmin(dist_current + dist_depot)
        return int(best)
    # identify index of current truck
    mask = np.all(truck_positions == current_position, axis=1)
    current_idx = np.where(mask)[0][0]
    other_trucks = truck_positions[~mask]
    # nearest distance from each customer to any other truck
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)
    # linear score: dist_current - nearest_other + 0.5 * dist_depot
    scores = dist_current - nearest_other + 0.5 * dist_depot
    best = np.argmin(scores)
    return int(best)