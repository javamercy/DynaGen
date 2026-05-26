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
    
    mask = np.all(truck_positions == current_position, axis=1)
    if mask.any():
        other_trucks = truck_positions[~mask]
    else:
        other_trucks = truck_positions
    
    if len(other_trucks) == 0:
        score = dist_current + 0.3 * dist_depot
        best_idx = np.argmin(score)
        return int(best_idx)
    
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)
    
    centroid = np.mean(truck_positions, axis=0)
    truck_to_centroid = np.linalg.norm(current_position - centroid)
    mean_centroid_dist = np.mean(np.linalg.norm(truck_positions - centroid, axis=1))
    if mean_centroid_dist < 1e-6:
        depot_coefficient = 0.3
    else:
        depot_coefficient = 0.3 + 0.1 * (truck_to_centroid / mean_centroid_dist)
        depot_coefficient = min(depot_coefficient, 0.5)
    
    current_to_depot = np.linalg.norm(current_position - depot_position)
    mean_depot_dist = np.mean(np.linalg.norm(truck_positions - depot_position, axis=1))
    if current_to_depot < 0.5 * mean_depot_dist:
        pass
    else:
        wait_threshold = 1.15
        if np.all(dist_current > wait_threshold * nearest_other):
            return None
    
    score = dist_current - nearest_other + depot_coefficient * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)