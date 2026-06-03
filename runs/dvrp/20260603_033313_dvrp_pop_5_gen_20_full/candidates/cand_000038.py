import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    dist_active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    
    all_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    other_dists = np.copy(all_dists)
    other_dists[:, active_idx] = np.inf
    min_other_dists = np.min(other_dists, axis=1)
    
    active_is_closest = all_dists[:, active_idx] <= min_other_dists
    num_closest = np.sum(active_is_closest)
    
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    median_depot_dist = np.median(np.linalg.norm(truck_positions - depot_position, axis=1))
    
    if active_depot_dist < median_depot_dist and num_closest == 0:
        return None
    
    if num_closest > 0:
        candidate_indices = np.where(active_is_closest)[0]
        best_idx = candidate_indices[np.argmax(dist_cust_to_depot[candidate_indices])]
    else:
        costs = dist_active_to_cust + dist_cust_to_depot
        best_idx = np.argmin(costs)
    
    return int(best_idx)