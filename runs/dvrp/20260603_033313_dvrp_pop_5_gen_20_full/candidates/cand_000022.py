import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # distances from active truck to each customer
    dist_active = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    dist_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # find index of active truck in truck_positions
    is_active = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_idx = np.where(is_active)[0][0]
    
    # distances from all trucks to all customers: shape (n_cust, n_trucks)
    all_dists = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    
    # minimum distance to any other truck
    other_dists = np.copy(all_dists)
    other_dists[:, active_idx] = np.inf
    min_other = np.min(other_dists, axis=1)
    
    # is active truck strictly the closest?
    active_is_closest = all_dists[:, active_idx] < min_other
    
    if np.any(active_is_closest):
        # eligible customers
        eligible = np.where(active_is_closest)[0]
        # choose among eligible minimizing (dist_active + dist_to_depot)
        scores = dist_active[eligible] + dist_to_depot[eligible]
        best_idx = eligible[np.argmin(scores)]
    else:
        # fallback: minimize (dist_active + dist_to_depot) overall
        scores = dist_active + dist_to_depot
        best_idx = np.argmin(scores)
    
    return int(best_idx)