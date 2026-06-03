import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    def dist(a, b):
        return np.linalg.norm(a - b)
    
    n_trucks = len(truck_positions)
    # Current max return time if all trucks return directly to depot
    cur_max_return = max(dist(pos, depot_position) for pos in truck_positions)
    
    best_candidate = None
    best_score = float('inf')
    best_immediate_dist = float('inf')
    
    for i, cust in enumerate(available_customers):
        immediate_to_cust = dist(current_position, cust)
        cust_return = dist(cust, depot_position)
        immediate_total = immediate_to_cust + cust_return
        
        # Compute regret as in parent
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
        
        # Penalty for increasing max return time
        penalty = 0.5 * max(0, immediate_total - cur_max_return)
        score = regret + penalty
        
        if score < best_score or (score == best_score and immediate_total < best_immediate_dist):
            best_score = score
            best_immediate_dist = immediate_total
            best_candidate = i
    
    # Decision whether to wait
    if best_score < 0:
        return best_candidate
    else:
        # No negative score
        if dist(current_position, depot_position) < 1e-6:
            # At depot, always go
            return best_candidate
        else:
            # Waiting threshold: fraction of distance to depot
            dist_to_depot = dist(current_position, depot_position)
            alpha = 0.2
            wait_threshold = alpha * dist_to_depot
            if best_score < wait_threshold:
                return best_candidate
            else:
                return None