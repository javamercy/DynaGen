import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    def dist(a, b):
        return np.linalg.norm(a - b)
    
    n_trucks = len(truck_positions)
    best_candidate = None
    best_regret = float('inf')
    best_immediate_dist = float('inf')
    
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
            if regret < best_regret or (regret == best_regret and immediate_total < best_immediate_dist):
                best_regret = regret
                best_immediate_dist = immediate_total
                best_candidate = i
        else:
            # collect for threshold-based selection
            if regret < best_regret:
                best_regret = regret
                best_immediate_dist = immediate_total
                best_candidate_candidate = i
    
    # If negative regret found, return it
    if best_candidate is not None:
        return best_candidate
    
    # No negative regret: decide whether to still pick based on threshold
    # Always pick if truck is at depot
    if np.linalg.norm(current_position - depot_position) < 1e-6:
        best_regret = float('inf')
        best_candidate_from_threshold = None
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
            if regret < best_regret:
                best_regret = regret
                best_immediate_dist = immediate_total
                best_candidate_from_threshold = i
        return best_candidate_from_threshold
    else:
        # compute threshold as fraction of distance to depot
        dist_to_depot = np.linalg.norm(current_position - depot_position)
        alpha = 0.2  # tunable
        wait_threshold = alpha * dist_to_depot
        best_regret = float('inf')
        best_candidate_threshold = None
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
            if regret < wait_threshold and regret < best_regret:
                best_regret = regret
                best_immediate_dist = immediate_total
                best_candidate_threshold = i
            elif regret < wait_threshold and regret == best_regret and immediate_total < best_immediate_dist:
                best_immediate_dist = immediate_total
                best_candidate_threshold = i
        return best_candidate_threshold