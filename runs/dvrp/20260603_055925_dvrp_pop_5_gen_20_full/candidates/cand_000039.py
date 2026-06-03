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
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    best_regret = -float('inf')
    best_idx = None
    best_secondary = -float('inf')
    
    n_available = len(available_customers)
    threshold = -0.1 * n_trucks / n_available
    
    for i, cust in enumerate(available_customers):
        curr_to_cust = np.linalg.norm(current_position - cust)
        pot_return_curr = curr_to_cust + cust_to_depot[i]
        
        other_returns = []
        for j, pos in enumerate(truck_positions):
            if j == current_truck_idx:
                continue
            other_to_cust = np.linalg.norm(pos - cust)
            other_return = other_to_cust + cust_to_depot[i]
            other_returns.append(other_return)
        min_other = min(other_returns) if other_returns else float('inf')
        
        regret = min_other - pot_return_curr
        secondary = -pot_return_curr
        
        if (regret > best_regret) or (regret == best_regret and secondary > best_secondary):
            best_regret = regret
            best_idx = i
            best_secondary = secondary
    
    if best_regret >= threshold:
        return best_idx
    else:
        return None