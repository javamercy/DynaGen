import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    dist_active_to_customers = np.linalg.norm(available_customers - current_position, axis=1)
    dist_customers_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    is_active = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_index = np.where(is_active)[0][0]
    
    all_truck_dists = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    
    # Distances to other trucks, excluding active
    other_truck_dists = np.copy(all_truck_dists)
    other_truck_dists[:, active_index] = np.inf
    min_other_dists = np.min(other_truck_dists, axis=1)
    
    active_is_closest = all_truck_dists[:, active_index] <= min_other_dists
    
    if np.any(active_is_closest):
        candidate_indices = np.where(active_is_closest)[0]
        best_idx = candidate_indices[np.argmax(dist_customers_to_depot[candidate_indices])]
    else:
        # For customers where active is not closest, compute ratio of active dist to min other dist
        # Avoid division by zero by setting infinites to large ratio
        ratios = np.where(min_other_dists == np.inf, np.inf, dist_active_to_customers / min_other_dists)
        # Among those with finite ratio, choose the smallest ratio, then farthest from depot
        # Only customers with finite min_other_dists (i.e., at least one other truck) should be considered
        valid = np.isfinite(ratios)
        if np.any(valid):
            # Sort by ratio ascending, then by depot distance descending
            indices = np.arange(len(available_customers))[valid]
            sorted_indices = indices[np.lexsort((-dist_customers_to_depot[valid], ratios[valid]))]
            best_idx = sorted_indices[0]
        else:
            # Fallback: pick nearest customer (should not happen often)
            best_idx = np.argmin(dist_active_to_customers)
    
    return int(best_idx)