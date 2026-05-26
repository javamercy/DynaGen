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
    
    cur_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = cur_to_cust + cust_to_depot
    
    mask = ~np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[mask]
    
    if len(other_trucks) == 0:
        return int(np.argmin(cost_now))
    
    other_to_cust = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
    best_other_cost = np.min(other_to_cust, axis=1) + cust_to_depot
    regret = best_other_cost - cost_now
    
    centroid_other = np.mean(other_trucks, axis=0)
    dist_to_centroid = np.linalg.norm(available_customers - centroid_other, axis=1)
    alpha = 0.5
    regret += alpha * dist_to_centroid
    
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    current_max_depot = max(current_depot_dist, np.max(other_depot_dists))
    new_max = np.maximum(np.max(other_depot_dists), cust_to_depot)
    reduction = current_max_depot - new_max
    beta = 0.05
    regret += beta * reduction
    
    max_regret = np.max(regret)
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    is_farthest = current_depot_dist >= np.max(other_depot_dists) - 1e-6
    if is_farthest or len(available_customers) > 5:
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    else:
        return None