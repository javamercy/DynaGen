import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if available_customers.shape[0] == 0:
        return None
    
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    all_dist = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    current_mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    all_dist[:, current_mask] = np.inf
    min_other = np.min(all_dist, axis=1)
    
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_truck_depot = np.linalg.norm(current_position - depot_position)
    max_truck_depot = np.max(truck_to_depot)
    w = 1 + current_truck_depot / max_truck_depot if max_truck_depot > 0 else 1.0
    
    score = curr_to_cust + w * cust_to_depot - min_other
    best_idx = np.argmin(score)
    return int(best_idx)