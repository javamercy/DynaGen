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
    
    # other trucks
    mask = np.all(truck_positions == current_position, axis=1)
    if mask.any():
        other_trucks = truck_positions[~mask]
    else:
        other_trucks = truck_positions
    
    if len(other_trucks) == 0:
        score = dist_current + 0.3 * dist_depot
        best_idx = np.argmin(score)
        return int(best_idx)
    
    # nearest other truck distance for each customer
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)
    
    # dynamic beta based on current truck's depot distance relative to other trucks
    my_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    max_other_depot = np.max(other_depot_dists) if len(other_depot_dists) > 0 else 1.0
    if max_other_depot < 1e-6:
        beta = 0.3
    else:
        beta = 0.2 + 0.5 * (my_depot_dist / max_other_depot)
        beta = min(max(beta, 0.2), 0.7)
    
    # waiting condition: all customers are far compared to nearest other truck AND current truck is far from depot
    wait_threshold = 1.15
    all_far = np.all(dist_current > wait_threshold * nearest_other)
    median_other_depot = np.median(other_depot_dists)
    truck_far_from_depot = my_depot_dist > 0.8 * median_other_depot if len(other_depot_dists) > 0 else True
    if all_far and truck_far_from_depot:
        return None
    
    score = dist_current - nearest_other + beta * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)