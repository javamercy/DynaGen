import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    def dist(a, b):
        return np.linalg.norm(a - b)
    
    n_trucks = len(truck_positions)
    current_dist = dist(current_position, depot_position)
    
    # Compute maximum depot distance among other trucks
    max_other = 0.0
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            d = dist(pos, depot_position)
            if d > max_other:
                max_other = d
    
    # Adaptive threshold: larger factor when current is close relative to max_other
    if max_other > 0 and current_dist > 0:
        factor = min(max_other / current_dist, 2.0)
    else:
        factor = 1.0
    alpha = 0.2 * factor
    
    best_neg_idx = None
    best_neg_regret = float('inf')
    best_neg_immediate = float('inf')
    best_pos_idx = None
    best_pos_regret = float('inf')
    best_pos_immediate = float('inf')
    
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
            regret = -1.0  # always negative for single truck
        
        if regret < 0:
            if regret < best_neg_regret or (regret == best_neg_regret and immediate_total < best_neg_immediate):
                best_neg_regret = regret
                best_neg_immediate = immediate_total
                best_neg_idx = i
        else:
            if regret < alpha * current_dist:
                if regret < best_pos_regret or (regret == best_pos_regret and immediate_total < best_pos_immediate):
                    best_pos_regret = regret
                    best_pos_immediate = immediate_total
                    best_pos_idx = i
    
    if best_neg_idx is not None:
        return best_neg_idx
    return best_pos_idx