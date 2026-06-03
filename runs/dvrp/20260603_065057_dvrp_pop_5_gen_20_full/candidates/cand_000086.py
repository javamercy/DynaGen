import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # Identify current truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    # Direct return distances for all trucks
    direct_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Other trucks' direct return distances (exclude current)
    other_direct = np.delete(direct_to_depot, current_idx)
    max_other_direct = np.max(other_direct) if len(other_direct) > 0 else 0.0
    best_max = np.inf
    best_idx = -1
    best_this_return = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        max_return = max(this_return, max_other_direct)
        if max_return < best_max or (max_return == best_max and this_return < best_this_return):
            best_max = max_return
            best_idx = i
            best_this_return = this_return
    return best_idx