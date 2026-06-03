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
    dist_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_trucks)
    # Pre-compute baseline return times for other trucks (distance to depot)
    other_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    max_other = np.max(np.delete(other_returns, current_idx))
    best_score = np.inf
    best_customer_idx = -1
    best_current_return = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        new_current_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(new_current_return, max_other)
        if new_max < best_score or (new_max == best_score and new_current_return < best_current_return):
            best_score = new_max
            best_customer_idx = i
            best_current_return = new_current_return
    return best_customer_idx