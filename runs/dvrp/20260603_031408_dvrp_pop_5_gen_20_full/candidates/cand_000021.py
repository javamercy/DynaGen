import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_truck_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    current_dist_to_depot = dist_to_depot[current_truck_idx]
    current_max = np.max(dist_to_depot)
    best_idx = None
    best_hyp_max = np.inf
    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_cust_to_depot = np.linalg.norm(depot_position - cust)
        hyp_return_current = dist_to_cust + dist_cust_to_depot
        hyp_returns = dist_to_depot.copy()
        hyp_returns[current_truck_idx] = hyp_return_current
        hyp_max = np.max(hyp_returns)
        if hyp_max < best_hyp_max:
            best_hyp_max = hyp_max
            best_idx = i
    if n_trucks == 1:
        return best_idx
    other_dist_to_depot = np.delete(dist_to_depot, current_truck_idx)
    avg_other_dist = np.mean(other_dist_to_depot)
    if current_dist_to_depot < 0.8 * avg_other_dist and best_hyp_max > current_max:
        return None
    return best_idx