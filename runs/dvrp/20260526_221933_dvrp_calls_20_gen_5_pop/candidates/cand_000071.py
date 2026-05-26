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
    
    # Distance to nearest other truck (excluding current)
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
    
    # Dynamic depot coefficient based on current_time and max truck distance
    max_truck_dist = np.max(np.linalg.norm(truck_positions - depot_position, axis=1))
    if max_truck_dist < 1e-6:
        max_truck_dist = 1.0
    depot_weight = 0.3 * (1 + current_time / max_truck_dist)
    depot_weight = min(depot_weight, 1.0)  # cap at 1.0
    
    # Score: lower is better
    score = dist_current - nearest_other + depot_weight * dist_depot
    best_idx = np.argmin(score)
    
    # Waiting condition: if best customer is far compared to distance to depot
    dist_to_depot = np.linalg.norm(current_position - depot_position)
    if dist_to_depot > 1e-6 and dist_current[best_idx] > 2.0 * dist_to_depot:
        return None
    
    return int(best_idx)