import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_score = -float('inf')
    epsilon = 1e-6
    alpha = 1.0
    for i, cust in enumerate(available_customers):
        dist_truck = np.linalg.norm(current_position - cust)
        dist_depot = np.linalg.norm(depot_position - cust)
        dists = np.linalg.norm(truck_positions - cust, axis=1)
        if len(dists) > 1:
            sorted_dists = np.sort(dists)
            min_other = sorted_dists[1] if sorted_dists[0] < epsilon else sorted_dists[0]
        else:
            min_other = dists[0]
        penalty = alpha / (min_other + epsilon)
        score = dist_depot / (dist_truck + epsilon) - penalty
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score < 0:
        return None
    return best_idx