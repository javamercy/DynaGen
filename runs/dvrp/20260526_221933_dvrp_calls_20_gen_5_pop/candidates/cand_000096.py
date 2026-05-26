import numpy as np

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
    
    # Identify other trucks (excluding current)
    mask = np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        score = dist_current + 0.3 * dist_depot
        best_idx = np.argmin(score)
        return int(best_idx)
    
    # Compute nearest other truck distance for each customer
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)
    
    # Dynamic depot coefficient based on distance to fleet centroid
    fleet_centroid = np.mean(truck_positions, axis=0)
    truck_to_centroid = np.linalg.norm(current_position - fleet_centroid)
    all_trucks_to_centroid = np.linalg.norm(truck_positions - fleet_centroid, axis=1)
    mean_centroid_dist = np.mean(all_trucks_to_centroid)
    if mean_centroid_dist < 1e-6:
        depot_coefficient = 0.3
    else:
        depot_coefficient = 0.2 + 0.2 * (truck_to_centroid / mean_centroid_dist)
        depot_coefficient = min(depot_coefficient, 0.5)
    
    # Wait condition: threshold and not too close to depot
    wait_threshold = 1.15
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    avg_other_depot_dist = np.mean(other_depot_dists) if len(other_depot_dists) > 0 else 0.0
    
    if np.all(dist_current > wait_threshold * nearest_other) and current_depot_dist > 0.5 * avg_other_depot_dist:
        return None
    
    score = dist_current - nearest_other + depot_coefficient * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)