def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # distances from current truck to customers
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from depot to customers
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # mask to exclude current truck
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        # no other trucks, cannot wait
        nearest_other = np.full(len(available_customers), np.inf)
        wait_flag = False
    else:
        # nearest other truck distance to each customer
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        # waiting condition: current truck's min dist > 1.5 * average min dist of other trucks
        min_other_dists = np.min(dist_other, axis=0)  # min dist per other truck to any customer
        avg_min_other = np.mean(min_other_dists)
        min_current = np.min(dist_current)
        wait_flag = min_current > 1.5 * avg_min_other and len(other_trucks) > 0
    if wait_flag:
        return None
    # fleet centroid and current truck's distance to it
    fleet_centroid = np.mean(truck_positions, axis=0)
    centroid_dist = np.linalg.norm(current_position - fleet_centroid)
    # normalize centroid distance by max distance among trucks from centroid
    centroid_dists_all = np.linalg.norm(truck_positions - fleet_centroid, axis=1)
    max_centroid_dist = np.max(centroid_dists_all)
    if max_centroid_dist == 0:
        ratio = 0.0
    else:
        ratio = centroid_dist / max_centroid_dist
    depot_coef = 0.3 + 0.5 * ratio
    # compute score
    score = dist_current - nearest_other + depot_coef * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)