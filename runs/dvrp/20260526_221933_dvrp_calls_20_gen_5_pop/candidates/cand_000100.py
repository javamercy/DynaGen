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
    
    # Fleet-state-based depot coefficient
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    min_depot = np.min(truck_depot_dists)
    max_depot = np.max(truck_depot_dists)
    if max_depot - min_depot < 1e-6:
        depot_coefficient = 0.3
    else:
        # Normalize current truck's depot distance between 0 and 1
        ratio = (current_depot_dist - min_depot) / (max_depot - min_depot)
        depot_coefficient = 0.2 + 0.3 * ratio  # range [0.2, 0.5]
    
    # Waiting condition
    wait_threshold = 1.3
    if np.all(dist_current > wait_threshold * nearest_other):
        return None
    
    score = dist_current - nearest_other + depot_coefficient * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)