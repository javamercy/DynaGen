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
    n_avail = len(available_customers)
    n_trucks = len(truck_positions)
    # dynamic depot weight: higher when few customers remain (density low)
    density_ratio = min(n_avail / (n_trucks * 2), 1.0)  # cap at 1
    w_depot = 0.3 + 0.7 * (1 - density_ratio)  # range [0.3, 1.0]
    score = dist_current - nearest_other + w_depot * dist_depot
    best_idx = np.argmin(score)
    # waiting condition: if best customer is far compared to distance to depot
    dist_to_depot = np.linalg.norm(current_position - depot_position)
    if dist_current[best_idx] > 1.5 * dist_to_depot:
        return None
    return int(best_idx)