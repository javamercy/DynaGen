def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_other = len(truck_positions) - 1
    # distances from current to customers
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    min_current = np.min(dist_current)
    # mask to exclude current truck
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        # only one truck, no waiting condition
        wait = False
    else:
        # for each other truck, find distance to nearest customer
        diff = available_customers[np.newaxis, :, :] - other_trucks[:, np.newaxis, :]  # (n_other, n_cust, 2)
        dist_other = np.linalg.norm(diff, axis=2)  # (n_other, n_cust)
        min_other_per_truck = np.min(dist_other, axis=1)  # (n_other,)
        min_other = np.min(min_other_per_truck)
        wait = min_current > 1.2 * min_other + 1e-6
    if wait:
        return None
    # compute nearest other distance for each customer
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]  # (n_cust, n_other, 2)
        dist_other = np.linalg.norm(diff, axis=2)  # (n_cust, n_other)
        nearest_other = np.min(dist_other, axis=1)
    # dynamic depot coefficient based on distance to fleet centroid
    centroid = np.mean(truck_positions, axis=0)
    dist_current_to_centroid = np.linalg.norm(current_position - centroid)
    distances_to_centroid = np.linalg.norm(truck_positions - centroid, axis=1)
    avg_dist_to_centroid = np.mean(distances_to_centroid)
    if avg_dist_to_centroid > 0:
        coeff = 0.3 + 0.1 * (dist_current_to_centroid / avg_dist_to_centroid)
        coeff = min(coeff, 0.6)  # clamp to avoid extreme
    else:
        coeff = 0.3
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    score = dist_current - nearest_other + coeff * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)