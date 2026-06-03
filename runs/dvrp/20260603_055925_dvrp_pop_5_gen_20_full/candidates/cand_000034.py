import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    
    if n_trucks == 1:
        best_idx = None
        best_immediate = float('inf')
        for i, cust in enumerate(available_customers):
            total = dist(current_position, cust) + dist(cust, depot_position)
            if total < best_immediate:
                best_immediate = total
                best_idx = i
        return best_idx
    
    current_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            current_idx = i
            break
    max_other = 0.0
    for i, pos in enumerate(truck_positions):
        if i != current_idx:
            d = dist(pos, depot_position)
            if d > max_other:
                max_other = d
    
    best_candidate = None
    best_regret = float('inf')
    best_immediate = float('inf')
    
    for i, cust in enumerate(available_customers):
        immediate_total = dist(current_position, cust) + dist(cust, depot_position)
        regret = immediate_total - max_other
        if regret < 0:
            if regret < best_regret or (regret == best_regret and immediate_total < best_immediate):
                best_regret = regret
                best_immediate = immediate_total
                best_candidate = i
    
    return best_candidate