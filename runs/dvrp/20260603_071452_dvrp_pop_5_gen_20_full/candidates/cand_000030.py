import numpy as np
def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    dist_current_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    min_idx = np.argmin(dist_current_to_cust)
    min_dist = dist_current_to_cust[min_idx]
    dist_to_depot = np.linalg.norm(current_position - depot_position)
    if min_dist > 2 * dist_to_depot:
        return None
    else:
        return int(min_idx)