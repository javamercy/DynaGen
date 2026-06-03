import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # Distances
    dist_active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # Active truck index
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    
    # All trucks to all customers
    all_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    # Min distance from other trucks
    other_dists = np.copy(all_dists)
    other_dists[:, active_idx] = np.inf
    min_other_dists = np.min(other_dists, axis=1)
    
    active_is_closest = all_dists[:, active_idx] <= min_other_dists
    
    # Number of customers where active is closest
    num_closest = np.sum(active_is_closest)
    
    # Active truck's distance to depot
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    median_depot_dist = np.median(np.linalg.norm(truck_positions - depot_position, axis=1))
    
    # Depot-return pressure: if active is closer to depot than median and not closest to any customer, wait
    if active_depot_dist < median_depot_dist and num_closest == 0:
        return None
    
    # Otherwise, same logic as parent
    if num_closest > 0:
        candidate_indices = np.where(active_is_closest)[0]
        best_idx = candidate_indices[np.argmax(dist_cust_to_depot[candidate_indices])]
    else:
        best_idx = np.argmin(dist_active_to_cust)
    
    return int(best_idx)