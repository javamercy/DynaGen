import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    def dist(a, b):
        return np.linalg.norm(a - b)
    
    n_trucks = len(truck_positions)
    n_avail = len(available_customers)
    
    # Adaptive alpha: fewer remaining customers -> more willing to wait (higher alpha)
    if n_avail > 10:
        alpha = 0.15
    elif n_avail > 5:
        alpha = 0.2
    elif n_avail > 2:
        alpha = 0.3
    else:
        alpha = 0.4
    
    best_negative_candidate = None
    best_negative_regret = float('inf')
    best_negative_immediate = float('inf')
    
    # First pass: find best customer with negative regret
    for i, cust in enumerate(available_customers):
        cust_return = dist(cust, depot_position)
        immediate_total = dist(current_position, cust) + cust_return
        
        if n_trucks > 1:
            best_other = float('inf')
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):
                    continue
                deferred = dist(pos, cust) + cust_return
                if deferred < best_other:
                    best_other = deferred
            regret = immediate_total - best_other
        else:
            regret = -1.0
        
        if regret < 0:
            if regret < best_negative_regret or (regret == best_negative_regret and immediate_total < best_negative_immediate):
                best_negative_regret = regret
                best_negative_immediate = immediate_total
                best_negative_candidate = i
    
    if best_negative_candidate is not None:
        return best_negative_candidate
    
    # No negative regret: use threshold based on adaptive alpha
    best_threshold_candidate = None
    best_threshold_regret = float('inf')
    best_threshold_immediate = float('inf')
    
    for i, cust in enumerate(available_customers):
        cust_return = dist(cust, depot_position)
        immediate_total = dist(current_position, cust) + cust_return
        
        if n_trucks > 1:
            best_other = float('inf')
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):
                    continue
                deferred = dist(pos, cust) + cust_return
                if deferred < best_other:
                    best_other = deferred
            regret = immediate_total - best_other
        else:
            regret = -1.0
        
        if regret < alpha * dist(current_position, depot_position):
            if regret < best_threshold_regret or (regret == best_threshold_regret and immediate_total < best_threshold_immediate):
                best_threshold_regret = regret
                best_threshold_immediate = immediate_total
                best_threshold_candidate = i
    
    return best_threshold_candidate