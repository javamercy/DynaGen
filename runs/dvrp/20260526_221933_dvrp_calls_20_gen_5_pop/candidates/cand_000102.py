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
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        best = np.argmin(dist_current)
        return int(best)
    
    # distances from each available customer to other trucks
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)
    
    wait_threshold = 1.3
    if np.all(dist_current > wait_threshold * nearest_other):
        return None
    
    # adaptive beta based on current truck's depot distance vs median of other trucks
    cur_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    median_other = np.median(other_depot_dists)
    beta = 0.7 if cur_depot_dist > median_other else 0.2
    
    score = dist_current - nearest_other + beta * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)