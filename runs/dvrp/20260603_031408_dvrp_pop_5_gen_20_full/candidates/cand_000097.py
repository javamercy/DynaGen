import numpy as np
def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # identify other trucks
    is_current = np.all(truck_positions == current_position, axis=1)
    other_positions = truck_positions[~is_current]
    if len(other_positions) > 0:
        other_max = np.max(np.linalg.norm(depot_position - other_positions, axis=1))
    else:
        other_max = -np.inf
    best_idx = None
    best_tuple = (float('inf'), float('inf'))
    for i, cust in enumerate(available_customers):
        active_finish = np.linalg.norm(current_position - cust) + np.linalg.norm(depot_position - cust)
        new_max = max(active_finish, other_max)
        candidate = (new_max, active_finish)
        if candidate < best_tuple:
            best_tuple = candidate
            best_idx = i
    return best_idx