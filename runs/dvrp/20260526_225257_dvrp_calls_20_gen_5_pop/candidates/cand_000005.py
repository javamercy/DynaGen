import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    n_available = len(available_customers)
    if n_available == 0:
        return None
    
    n_trucks = truck_positions.shape[0]
    
    # find index of current truck (closest)
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    
    best_cost = float('inf')
    best_idx = None
    depot = depot_position
    
    for i in range(n_available):
        cust = available_customers[i]
        d_curr_cust = np.linalg.norm(current_position - cust)
        d_cust_depot = np.linalg.norm(cust - depot)
        cost_now = current_time + d_curr_cust + d_cust_depot
        
        # compute best other cost
        best_other = float('inf')
        for j in range(n_trucks):
            if j == current_idx:
                continue
            d_truck_cust = np.linalg.norm(truck_positions[j] - cust)
            cost_other = current_time + d_truck_cust + d_cust_depot
            if cost_other < best_other:
                best_other = cost_other
        
        if n_trucks == 1:
            regret = -float('inf')  # always serve now
        else:
            regret = cost_now - best_other
        
        # consider if current truck is best (regret <= 0)
        if regret <= 0:
            if cost_now < best_cost:
                best_cost = cost_now
                best_idx = i
    
    return best_idx