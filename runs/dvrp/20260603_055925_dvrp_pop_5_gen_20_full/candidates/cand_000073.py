import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    def dist(a, b):
        return np.linalg.norm(a - b)
    
    n_trucks = len(truck_positions)
    current_to_depot = dist(current_position, depot_position)
    
    # Single truck: pick the one with smallest immediate total
    if n_trucks == 1:
        best_idx = None
        best_total = float('inf')
        for i, cust in enumerate(available_customers):
            total = dist(current_position, cust) + dist(cust, depot_position)
            if total < best_total:
                best_total = total
                best_idx = i
        return best_idx
    
    # Parameters
    alpha = 0.3
    threshold_factor = 0.2
    
    best_score = float('inf')
    best_idx = None
    
    for i, cust in enumerate(available_customers):
        dist_current_cust = dist(current_position, cust)
        dist_cust_depot = dist(cust, depot_position)
        immediate_total = dist_current_cust + dist_cust_depot
        
        # Best other truck's total for this customer
        best_other = float('inf')
        for j, pos in enumerate(truck_positions):
            if np.array_equal(pos, current_position):
                continue
            other_total = dist(pos, cust) + dist_cust_depot
            if other_total < best_other:
                best_other = other_total
        regret = immediate_total - best_other
        
        score = regret + alpha * dist_cust_depot
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    threshold = threshold_factor * current_to_depot
    if best_score < threshold:
        return best_idx
    else:
        return None