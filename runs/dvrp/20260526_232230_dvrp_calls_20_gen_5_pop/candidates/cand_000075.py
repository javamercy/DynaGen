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
    # Compute current max return time (all trucks return directly to depot)
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(dist_to_depot)
    # Active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    best_idx = None
    best_score = np.inf
    best_new_return = np.inf
    min_new_max = np.inf
    # Evaluate each customer
    for i, cust in enumerate(available_customers):
        new_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(current_max, new_return)
        excess = max(0.0, new_return - current_max)
        score = new_max + 0.1 * excess
        # Track min new_max for wait decision
        if new_max < min_new_max:
            min_new_max = new_max
        # Tie-break: prefer smaller new_return
        if (score < best_score) or (score == best_score and new_return < best_new_return):
            best_score = score
            best_new_return = new_return
            best_idx = i
    # Wait decision: if current_max > 0 and all customers increase max by more than 10%?
    # Actually we check if the best (minimum) new_max is > 1.1 * current_max
    if current_max > 0 and min_new_max > 1.1 * current_max:
        return None
    return best_idx