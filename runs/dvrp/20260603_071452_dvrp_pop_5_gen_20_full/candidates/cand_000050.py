import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # distances from current position to each customer
    cust_dists = np.linalg.norm(available_customers - current_position, axis=1)
    min_idx = np.argmin(cust_dists)
    min_dist = cust_dists[min_idx]
    
    # distance from current position to depot
    curr_to_depot = np.linalg.norm(current_position - depot_position)
    # distances from all trucks to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    mean_truck_depot = np.mean(truck_depot_dists)
    
    # average distance of available customers to depot
    cust_depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    mean_cust_depot = np.mean(cust_depot_dists)
    
    # Decision: if truck is farther than average, always go to nearest
    if curr_to_depot > mean_truck_depot:
        return int(min_idx)
    else:
        # Wait only if nearest customer is much farther than average customer-depot distance
        if min_dist > 1.5 * mean_cust_depot:
            return None
        else:
            return int(min_idx)