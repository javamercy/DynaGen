def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))

    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))

    # Compute distances to depot for all trucks
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    original_makespan = np.max(dist_to_depot)

    best_idx = None
    best_regret = -float('inf')
    best_secondary = -float('inf')

    for i, cust in enumerate(available_customers):
        current_dist = np.linalg.norm(current_position - cust)
        new_est_current = current_dist + np.linalg.norm(cust - depot_position)
        new_makespan = max(new_est_current, np.max(dist_to_depot[np.arange(n_trucks) != current_truck_idx]))
        delta = new_makespan - original_makespan
        if delta > 0:
            continue

        # Compute regret
        other_dists = np.linalg.norm(truck_positions[np.arange(n_trucks) != current_truck_idx] - cust, axis=1)
        min_other = np.min(other_dists) if len(other_dists) > 0 else float('inf')
        regret = min_other - current_dist
        secondary = -current_dist

        if (regret > best_regret) or (regret == best_regret and secondary > best_secondary):
            best_regret = regret
            best_idx = i
            best_secondary = secondary

    return best_idx