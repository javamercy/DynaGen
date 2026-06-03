import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    dist_current_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    avg_dist_cust_to_depot = np.mean(dist_cust_to_depot)
    min_idx = np.argmin(dist_current_to_cust)
    min_dist = dist_current_to_cust[min_idx]
    if min_dist > 2 * avg_dist_cust_to_depot:
        return None
    else:
        return int(min_idx)