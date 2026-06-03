import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    cur_diffs = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(cur_diffs)
    current_return = depot_dists[current_idx]
    other_returns = np.delete(depot_dists, current_idx)
    if other_returns.size > 0:
        max_others = np.max(other_returns)
    else:
        max_others = -np.inf
    best_new_max = np.inf
    best_idx = -1
    best_this_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(this_cost, max_others)
        if new_max < best_new_max or (new_max == best_new_max and this_cost < best_this_cost):
            best_new_max = new_max
            best_idx = i
            best_this_cost = this_cost
    return best_idx