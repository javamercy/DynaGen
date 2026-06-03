import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Compute distances of each truck to depot
    truck_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Find index of the active truck (closest to current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Maximum distance to depot among other trucks
    if len(truck_positions) > 1:
        other_max = np.max(truck_depot[np.arange(len(truck_depot)) != active_idx])
    else:
        other_max = 0.0
    best_idx = None
    best_max = float('inf')
    for i, cust in enumerate(available_customers):
        my_total = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        candidate_max = max(my_total, other_max)
        if candidate_max < best_max:
            best_max = candidate_max
            best_idx = i
        elif candidate_max == best_max:
            # Tie-break: choose closer customer
            if np.linalg.norm(current_position - cust) < np.linalg.norm(current_position - available_customers[best_idx]):
                best_idx = i
    return best_idx