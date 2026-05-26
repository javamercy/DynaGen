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
    other_positions = truck_positions[mask]
    if len(other_positions) == 0:
        best_idx = np.argmin(dist_current)
        return int(best_idx)
    other_dist_to_cust = np.linalg.norm(
        available_customers[None, :, :] - other_positions[:, None, :], axis=2
    )
    other_nearest_min = np.min(other_dist_to_cust, axis=1)  # per other truck
    current_nearest = np.min(dist_current)
    # waiting condition: current truck's nearest customer is farther than max of other trucks' nearest
    if current_nearest > np.max(other_nearest_min):
        return None
    # dynamic depot weight
    other_depot_dist = np.linalg.norm(other_positions - depot_position, axis=1)
    avg_other_depot = np.mean(other_depot_dist)
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    depot_weight = 0.3
    if current_depot_dist > avg_other_depot:
        ratio = current_depot_dist / avg_other_depot
        depot_weight += 0.2 * (ratio - 1)
        depot_weight = min(depot_weight, 0.6)
    nearest_other = np.min(other_dist_to_cust, axis=0)  # per customer
    score = dist_current - nearest_other + depot_weight * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)