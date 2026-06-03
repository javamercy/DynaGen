import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Identify which truck is at current_position (assumed closest)
    diffs = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diffs)
    # Precompute other trucks' direct return distances
    other_return = np.delete(np.linalg.norm(truck_positions - depot_position, axis=1), current_idx)
    best_idx = None
    best_max = np.inf
    best_this = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        max_cost = this_cost
        if len(other_return) > 0:
            max_cost = max(this_cost, np.max(other_return))
        if max_cost < best_max or (max_cost == best_max and this_cost < best_this):
            best_max = max_cost
            best_this = this_cost
            best_idx = i
    return best_idx