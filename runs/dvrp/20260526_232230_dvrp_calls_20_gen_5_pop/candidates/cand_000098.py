import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    if len(truck_positions) == 1:
        max_others = -np.inf
    else:
        max_others = np.max(np.delete(dist_to_depot, active_idx))
    best_idx = None
    best_new_max = np.inf
    best_active_return = np.inf
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(active_return, max_others)
        if new_max < best_new_max or (new_max == best_new_max and active_return < best_active_return):
            best_new_max = new_max
            best_active_return = active_return
            best_idx = i
    return best_idx