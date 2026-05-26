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
    
    # Identify other trucks (exclude current truck)
    mask = np.all(truck_positions == current_position, axis=1)
    if mask.any():
        other_trucks = truck_positions[~mask]
    else:
        other_trucks = truck_positions
    
    if len(other_trucks) == 0:
        score = dist_current + 0.3 * dist_depot
        best_idx = np.argmin(score)
        return int(best_idx)
    
    # Nearest other truck distance for each customer
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)
    
    # Adaptive depot coefficient based on other trucks' distances to depot
    max_dist_to_depot = np.max(dist_depot)
    if max_dist_to_depot < 1e-8:
        depot_coefficient = 0.2
    else:
        avg_other_dist_to_depot = np.mean(np.linalg.norm(other_trucks - depot_position, axis=1))
        depot_coefficient = 0.2 + 0.2 * (1 - avg_other_dist_to_depot / max_dist_to_depot)
        depot_coefficient = np.clip(depot_coefficient, 0.1, 0.4)
    
    # Wait condition with threshold 1.2
    wait_threshold = 1.2
    if np.all(dist_current > wait_threshold * nearest_other):
        return None
    
    score = dist_current - nearest_other + depot_coefficient * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)