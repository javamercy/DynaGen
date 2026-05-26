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
        other_nearest_min = np.inf
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        # compute other trucks' nearest distances to any customer
        other_nearest = np.min(dist_other, axis=0)  # minimum per other truck
        other_nearest_min = np.min(other_nearest) if len(other_nearest) > 0 else np.inf
    
    # Fleet centroid and dynamic depot coefficient
    centroid = np.mean(truck_positions, axis=0)
    dist_to_centroid = np.linalg.norm(current_position - centroid)
    avg_centroid_dist = np.mean(np.linalg.norm(truck_positions - centroid, axis=1))
    depot_coef = 0.3 + 0.3 * (dist_to_centroid / (avg_centroid_dist + 1e-6))
    depot_coef = np.clip(depot_coef, 0.3, 1.0)
    
    # Waiting condition: if current truck's min distance is much larger than other trucks' min
    current_min = np.min(dist_current)
    if len(other_trucks) > 0 and current_min > 1.2 * other_nearest_min + 0.5 and current_min > 1.0:
        return None
    
    score = dist_current - nearest_other + depot_coef * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)