import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        # Only one truck: go to nearest customer
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    
    # Find index of current truck
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    
    # Compute distances from current truck to all customers
    current_dists = np.linalg.norm(available_customers - current_position, axis=1)
    
    # Compute current distances to depot for all trucks
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_depot = np.max(depot_dists)
    
    best_score = -np.inf
    best_idx = None
    
    for i, cust in enumerate(available_customers):
        # Distance from current truck to this customer
        d_cur_to_cust = current_dists[i]
        
        # Distance from this customer to depot
        d_cust_to_depot = np.linalg.norm(cust - depot_position)
        
        # New return time for current truck if it serves this customer and then goes to depot
        new_cur_return = d_cur_to_cust + d_cust_to_depot
        
        # New max depot distance after this assignment (other trucks unchanged)
        new_max = max(current_max_depot, new_cur_return)
        inc_max = new_max - current_max_depot
        
        # Compute regret: distance to nearest other truck minus distance to current truck
        dists_to_other_trucks = np.linalg.norm(truck_positions - cust, axis=1)
        dists_to_other_trucks = np.delete(dists_to_other_trucks, current_truck_idx)
        min_other = np.min(dists_to_other_trucks) if len(dists_to_other_trucks) > 0 else float('inf')
        regret = min_other - d_cur_to_cust
        
        score = regret - inc_max
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    if best_score > 0:
        return int(best_idx)
    else:
        return None