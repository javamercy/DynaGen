import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    current_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    n_trucks = truck_positions.shape[0]
    best_score = -float('inf')
    best_idx = None
    for i, cust in enumerate(available_customers):
        dist_to_truck = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(depot_position - cust)
        if n_trucks > 1:
            dists = np.linalg.norm(truck_positions - cust, axis=1)
            dists[current_idx] = np.inf
            dist_to_other = np.min(dists)
        else:
            dist_to_other = 0
        gamma = 1.0
        score = dist_to_depot - dist_to_truck - gamma * dist_to_other
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx