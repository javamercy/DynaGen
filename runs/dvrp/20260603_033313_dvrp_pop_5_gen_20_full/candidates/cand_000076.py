import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # Find index of active truck (assuming current_position is exactly one of truck_positions)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Distances from each truck to depot
    dist_truck_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    best_idx = None
    best_score = np.inf
    best_T_active = np.inf
    for i, cust in enumerate(available_customers):
        T_active = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if n_trucks > 1:
            max_other = np.max(dist_truck_depot[np.arange(n_trucks) != active_idx])
        else:
            max_other = -np.inf  # for single truck, potential = T_active
        potential = max(T_active, max_other)
        if potential < best_score or (potential == best_score and T_active < best_T_active):
            best_score = potential
            best_T_active = T_active
            best_idx = i
    return int(best_idx)