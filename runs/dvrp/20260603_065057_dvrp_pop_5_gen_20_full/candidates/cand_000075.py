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
    # Find current truck index (closest to current_position)
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # Precompute other trucks' current return distances (to depot)
    other_current_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    # For current truck, its current return distance if it does nothing
    current_return_now = np.linalg.norm(current_position - depot_position)
    # Max current return across all trucks
    current_max = max(np.max(other_current_returns), current_return_now)
    best_score = np.inf  # we want to minimize new_max, so score = -new_max, but we'll track best_new_max
    best_new_max = np.inf
    best_customer_idx = -1
    best_this_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Estimate new max after serving this customer
        # For current truck, new return = this_cost
        # For other trucks, returns remain same
        new_max = this_cost
        for j in range(n_trucks):
            if j == current_idx:
                continue
            if other_current_returns[j] > new_max:
                new_max = other_current_returns[j]
        # Score: negative new_max (higher is better), but we want to minimize new_max
        # So we compare new_max directly; smaller is better
        if new_max < best_new_max or (new_max == best_new_max and this_cost < best_this_cost):
            best_new_max = new_max
            best_customer_idx = i
            best_this_cost = this_cost
    return best_customer_idx