import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # compute current distance to depot for all trucks, and max among others
    cur_to_depot = np.linalg.norm(depot_position - current_position)
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dists) if other_dists else 0.0
    
    # waiting score: if we wait, remaining distance = cur_to_depot, penalty relative to max_other
    penalty_wait = max(0.0, cur_to_depot - max_other)
    wait_score = -1.5 * penalty_wait  # only penalty term, no isolation
    
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        cust_to_depot = np.linalg.norm(depot_position - cust)
        cust_to_truck = np.linalg.norm(current_position - cust)
        finish = cust_to_truck + cust_to_depot
        penalty = max(0.0, finish - max_other)
        # isolation: distance to nearest other truck
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        # savings term: difference between customer-to-depot and truck-to-customer
        savings = cust_to_depot - cust_to_truck
        score = savings + 0.8 * min_ot - 1.5 * penalty
        if score > best_score:
            best_score = score
            best_idx = i
    
    # if waiting yields higher score than best customer, wait
    if best_idx is not None and best_score <= wait_score:
        return None
    return best_idx