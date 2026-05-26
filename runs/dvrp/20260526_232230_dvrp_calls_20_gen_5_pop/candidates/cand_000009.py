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

    # Find active truck index
    dists = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dists)

    # Current return times: all trucks' straight line to depot
    current_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(current_returns)

    best_idx = None
    best_new_max = np.inf
    best_active_return = np.inf

    for i, cust in enumerate(available_customers):
        # Active truck's new return if serves this customer and goes back to depot
        d_to_cust = np.linalg.norm(current_position - cust)
        d_cust_to_depot = np.linalg.norm(cust - depot_position)
        active_new_return = d_to_cust + d_cust_to_depot

        # New returns array: replace active truck's current return with new one
        new_returns = current_returns.copy()
        new_returns[active_idx] = active_new_return
        new_max = np.max(new_returns)

        if new_max < best_new_max or (new_max == best_new_max and active_new_return < best_active_return):
            best_new_max = new_max
            best_active_return = active_new_return
            best_idx = i

    # If serving any customer increases the max, wait
    if best_new_max > current_max:
        return None
    else:
        return best_idx