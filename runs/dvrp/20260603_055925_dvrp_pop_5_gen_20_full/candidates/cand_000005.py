import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # Pairwise distances
    def dist(a, b):
        return np.linalg.norm(a - b)
    
    current_truck_rt = dist(current_position, depot_position)
    
    best_candidate = None
    best_regret = float('inf')
    best_dist = float('inf')
    
    n_trucks = len(truck_positions)
    
    for i, cust in enumerate(available_customers):
        cust_return = dist(cust, depot_position)
        immediate_total = dist(current_position, cust) + cust_return
        
        if n_trucks > 1:
            # Compute best deferred over other trucks
            best_other = float('inf')
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):  # skip current truck
                    continue
                deferred = dist(pos, cust) + cust_return
                if deferred < best_other:
                    best_other = deferred
            regret = immediate_total - best_other
        else:
            # Single truck: must serve, regret not defined, so force negative
            regret = -1.0
        
        # Select if regret is negative and better than current best
        if regret < 0:
            # More negative regret is better
            if regret < best_regret or (regret == best_regret and immediate_total < best_dist):
                best_regret = regret
                best_dist = immediate_total
                best_candidate = i
    
    if best_candidate is None:
        return None
    else:
        return best_candidate