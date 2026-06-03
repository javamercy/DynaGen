import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    def dist(a, b):
        return np.linalg.norm(a - b)
    
    current_dist = dist(current_position, depot_position)
    
    # Find best immediate total distance among available customers
    best_immediate = float('inf')
    for cust in available_customers:
        to_cust = dist(current_position, cust)
        from_cust = dist(cust, depot_position)
        immediate = to_cust + from_cust
        if immediate < best_immediate:
            best_immediate = immediate
    
    # Waiting condition: avoid taking a customer that would cause a long trip
    threshold = 1.5
    if best_immediate > threshold * current_dist:
        return None
    
    # Otherwise, select customer using regret-aware score
    n_trucks = len(truck_positions)
    best_score = float('inf')
    best_idx = None
    for i, cust in enumerate(available_customers):
        to_cust = dist(current_position, cust)
        from_cust = dist(cust, depot_position)
        immediate = to_cust + from_cust
        
        # Compute regret if multiple trucks
        if n_trucks > 1:
            best_other = float('inf')
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):
                    continue
                deferred = dist(pos, cust) + from_cust
                if deferred < best_other:
                    best_other = deferred
            regret = immediate - best_other
        else:
            regret = -1.0
        
        # Score: penalize positive regret, reward negative regret slightly
        if regret < 0:
            score = immediate - 0.1 * regret  # negative regret reduces score
        else:
            score = immediate + 0.5 * regret
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    return best_idx