import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Find index of current truck
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # Compute direct distances to depot for all trucks
    direct_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    max_other_direct = np.max(np.delete(direct_dists, current_idx))
    best_new_max = np.inf
    best_this_cost = np.inf
    best_customer_idx = -1
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(this_cost, max_other_direct)
        if new_max < best_new_max or (new_max == best_new_max and this_cost < best_this_cost):
            best_new_max = new_max
            best_this_cost = this_cost
            best_customer_idx = i
    return best_customer_idx