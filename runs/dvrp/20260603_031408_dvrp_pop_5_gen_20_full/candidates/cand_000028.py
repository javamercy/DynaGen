import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        dist_to_truck = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(depot_position - cust)
        dists_to_all_trucks = np.linalg.norm(truck_positions - cust, axis=1)
        sorted_dists = np.sort(dists_to_all_trucks)
        if len(sorted_dists) >= 2:
            min_dist_other = sorted_dists[1]
        else:
            min_dist_other = 0.0
        score = dist_to_depot - dist_to_truck - 0.5 * min_dist_other
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx