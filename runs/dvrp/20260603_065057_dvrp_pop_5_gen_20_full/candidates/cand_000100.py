import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Identify which truck is the current one
    dist_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_trucks)
    # Precompute direct return times for all trucks (if they go straight to depot)
    direct_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    best_score = np.inf
    best_idx = -1
    best_this_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Return times for all trucks
        returns = direct_returns.copy()
        returns[current_idx] = this_cost
        max_return = np.max(returns)
        if max_return < best_score or (max_return == best_score and this_cost < best_this_cost):
            best_score = max_return
            best_idx = i
            best_this_cost = this_cost
    return best_idx