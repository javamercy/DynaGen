import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # distances from active truck to customers and customers to depot
    a_to_c = np.linalg.norm(available_customers - current_position, axis=1)
    c_to_d = np.linalg.norm(available_customers - depot_position, axis=1)
    active_finishes = a_to_c + c_to_d
    
    # active truck's distance to depot
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    
    # other trucks' distances to depot
    all_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    other_depot_dists = np.delete(all_depot_dists, active_idx)
    max_other_dist = np.max(other_depot_dists) if len(other_depot_dists) > 0 else -np.inf
    
    # makespan if we serve a customer
    makespans = np.maximum(active_finishes, max_other_dist)
    best_makespan = np.min(makespans)
    best_indices = np.where(makespans == best_makespan)[0]
    
    # tie-breaking: among customers with same best makespan
    if len(best_indices) > 1:
        # compute which customers the active truck is closest to
        all_dists = np.linalg.norm(
            available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
        other_dists = np.copy(all_dists)
        other_dists[:, active_idx] = np.inf
        min_other_dists = np.min(other_dists, axis=1)
        active_closest = all_dists[:, active_idx] <= min_other_dists
        
        closest_best = best_indices[active_closest[best_indices]]
        if len(closest_best) > 0:
            best_idx = closest_best[0]
        else:
            # pick the customer with smallest active_finish among best
            best_idx = best_indices[np.argmin(active_finishes[best_indices])]
    else:
        best_idx = best_indices[0]
    
    # makespan if active truck waits (returns directly to depot)
    makespan_if_wait = max(active_depot_dist, max_other_dist) if len(other_depot_dists) > 0 else active_depot_dist
    
    # decision logic: serve if strictly better, or if equal but condition not met
    if best_makespan < makespan_if_wait:
        return int(best_idx)
    else:
        # check waiting condition inspired by parent
        median_depot_dist = np.median(all_depot_dists)
        # recompute active_closest (handy)
        all_dists = np.linalg.norm(
            available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
        other_dists = np.copy(all_dists)
        other_dists[:, active_idx] = np.inf
        min_other_dists = np.min(other_dists, axis=1)
        active_closest_all = np.any(all_dists[:, active_idx] <= min_other_dists)
        
        if active_depot_dist < median_depot_dist and not active_closest_all:
            return None
        else:
            return int(best_idx)