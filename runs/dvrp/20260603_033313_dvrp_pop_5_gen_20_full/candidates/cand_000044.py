import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # distances from each truck to depot
    dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = dists.max()
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    best_max = float('inf')
    best_idx = None
    best_cost = float('inf')
    for i, cust in enumerate(available_customers):
        cost_after = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_dists = dists.copy()
        new_dists[active_idx] = cost_after
        new_max = new_dists.max()
        if new_max < best_max or (new_max == best_max and cost_after < best_cost):
            best_max = new_max
            best_idx = i
            best_cost = cost_after
    if best_max > current_max:
        return None
    return best_idx