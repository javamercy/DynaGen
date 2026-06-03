import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    dist_active_to_customers = np.linalg.norm(available_customers - current_position, axis=1)
    dist_customers_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # Find index of active truck in truck_positions
    is_active = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_index = np.where(is_active)[0][0]
    
    # Distances from all trucks to all customers
    all_truck_dists = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    
    # Minimum distance from other trucks
    other_truck_dists = np.copy(all_truck_dists)
    other_truck_dists[:, active_index] = np.inf
    min_other_dists = np.min(other_truck_dists, axis=1)
    
    # Is active truck closest?
    active_is_closest = all_truck_dists[:, active_index] <= min_other_dists
    
    if np.any(active_is_closest):
        candidate_indices = np.where(active_is_closest)[0]
        # Among candidates, choose the one farthest from depot
        best_idx = candidate_indices[np.argmax(dist_customers_to_depot[candidate_indices])]
    else:
        # Active not closest to any, choose nearest customer
        best_idx = np.argmin(dist_active_to_customers)
    
    return int(best_idx)