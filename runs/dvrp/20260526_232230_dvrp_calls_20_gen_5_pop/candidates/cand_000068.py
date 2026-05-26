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
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    best_idx = None
    best_max = np.inf
    for i, cust in enumerate(available_customers):
        returns = dist_to_depot.copy()
        returns[active_idx] = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        max_return = np.max(returns)
        if max_return < best_max:
            best_max = max_return
            best_idx = i
    return best_idx