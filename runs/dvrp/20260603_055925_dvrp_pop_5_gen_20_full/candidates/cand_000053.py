import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if available_customers.shape[0] == 0:
        return None
    
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    other_truck_times = truck_to_depot.copy()
    other_truck_times[current_mask] = -np.inf
    other_max = np.max(other_truck_times)
    
    new_time = curr_to_cust + cust_to_depot
    effective = np.maximum(other_max, new_time)
    
    order = np.lexsort((new_time, effective))
    best_idx = order[0]
    return int(best_idx)