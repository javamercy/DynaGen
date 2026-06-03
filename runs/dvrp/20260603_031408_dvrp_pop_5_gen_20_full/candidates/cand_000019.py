import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_score = -float('inf')
    best_idx = None
    # Identify the index of the current truck
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    idx_current = np.argmin(diff)
    
    for i, cust in enumerate(available_customers):
        dist_to_current = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(depot_position - cust)
        all_dists = np.linalg.norm(truck_positions - cust, axis=1)
        # Exclude current truck
        mask = np.arange(len(truck_positions)) != idx_current
        other_dists = all_dists[mask]
        if len(other_dists) == 0:
            min_other = np.inf
        else:
            min_other = np.min(other_dists)
        advantage = min_other - dist_to_current
        score = advantage - dist_to_depot
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx