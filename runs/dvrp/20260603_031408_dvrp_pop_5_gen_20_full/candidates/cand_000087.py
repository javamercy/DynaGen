import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # distances of all trucks to depot
    dists_to_depot_all = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_dist_to_depot = np.linalg.norm(current_position - depot_position)
    # maximum distance among other trucks (exclude current truck's distance)
    other_dists = dists_to_depot_all[dists_to_depot_all != current_dist_to_depot]
    max_other = np.max(other_dists) if len(other_dists) > 0 else 0.0

    best_idx = None
    best_max = float('inf')
    best_new_dist = float('inf')

    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_cust_to_depot = np.linalg.norm(cust - depot_position)
        new_dist = dist_to_cust + dist_cust_to_depot
        candidate_max = max(new_dist, max_other)
        # choose smallest candidate_max, tie-break by smallest new_dist
        if candidate_max < best_max or (candidate_max == best_max and new_dist < best_new_dist):
            best_max = candidate_max
            best_new_dist = new_dist
            best_idx = i
    return best_idx