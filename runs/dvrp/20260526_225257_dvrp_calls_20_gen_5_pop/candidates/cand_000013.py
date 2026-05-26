import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers, current_time):
    if len(available_customers) == 0:
        return None
    
    # Compute distances from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Identify index of current truck
    mask = np.all(truck_positions == current_position, axis=1)
    current_idx = np.where(mask)[0][0]
    current_dist = dist_to_depot[current_idx]
    
    # Other trucks' distances to depot
    other_dists = np.delete(dist_to_depot, current_idx)
    other_max = np.max(other_dists) if len(other_dists) > 0 else 0.0
    
    # Current max return time considering all trucks (including this truck if it returns now)
    current_max = max(current_dist, other_max)
    
    best_idx = None
    best_max = np.inf
    best_candidate_dist = np.inf
    
    for i, customer in enumerate(available_customers):
        # Own return time if this customer is served: travel to customer then to depot
        own_return = np.linalg.norm(current_position - customer) + np.linalg.norm(customer - depot_position)
        candidate_max = max(own_return, other_max)
        if candidate_max < best_max - 1e-9:
            best_max = candidate_max
            best_candidate_dist = own_return
            best_idx = i
        elif abs(candidate_max - best_max) < 1e-9:
            if own_return < best_candidate_dist:
                best_candidate_dist = own_return
                best_idx = i
    
    # Wait if serving does not improve (reduce) the current max
    if best_idx is not None and best_max >= current_max - 1e-9:
        return None
    return best_idx