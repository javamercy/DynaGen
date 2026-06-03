import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # distances from all trucks to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # identify active truck (the one at current_position)
    diffs = truck_positions - current_position
    dists_to_active = np.linalg.norm(diffs, axis=1)
    active_idx = np.argmin(dists_to_active)
    # other trucks' distances to depot
    other_dists = np.delete(truck_depot_dists, active_idx)
    max_other = np.max(other_dists) if other_dists.size > 0 else 0.0
    best_idx = None
    best_new_max = float('inf')
    best_travel = float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        cust_to_depot = np.linalg.norm(cust - depot_position)
        active_return = travel + cust_to_depot
        new_max = max(max_other, active_return)
        if new_max < best_new_max:
            best_new_max = new_max
            best_travel = travel
            best_idx = i
        elif np.isclose(new_max, best_new_max):
            if travel < best_travel:
                best_travel = travel
                best_idx = i
    return best_idx