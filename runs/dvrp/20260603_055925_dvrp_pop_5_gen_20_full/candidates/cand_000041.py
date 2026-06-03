import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    # Current distances to depot for all trucks
    current_dists = np.array([dist(pos, depot_position) for pos in truck_positions])
    current_max = np.max(current_dists)
    # Identify index of current truck
    current_idx = np.argmin(np.sum((truck_positions - current_position)**2, axis=1))
    
    # Compute max distance among other trucks
    if n_trucks == 1:
        max_other = -np.inf
    else:
        other_mask = np.ones(n_trucks, dtype=bool)
        other_mask[current_idx] = False
        max_other = np.max(current_dists[other_mask])
    
    best_candidate = None
    best_improvement = -np.inf
    best_immediate = np.inf

    for i, cust in enumerate(available_customers):
        immediate_total = dist(current_position, cust) + dist(cust, depot_position)
        if n_trucks == 1:
            new_max = immediate_total
        else:
            new_max = max(immediate_total, max_other)
        improvement = current_max - new_max
        if improvement > 0:
            if improvement > best_improvement or (improvement == best_improvement and immediate_total < best_immediate):
                best_improvement = improvement
                best_immediate = immediate_total
                best_candidate = i

    return best_candidate