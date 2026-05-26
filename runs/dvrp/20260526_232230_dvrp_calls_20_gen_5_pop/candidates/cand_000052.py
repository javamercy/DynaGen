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
    # Identify active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Current distances from each truck to depot
    current_dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(current_dist_to_depot)
    w_mean = 0.3
    epsilon = 0.01
    best_score = np.inf
    best_new_max = None
    best_idx = None
    for i, cust in enumerate(available_customers):
        # Active truck's estimated return time if it serves this customer
        active_new_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Build return times: others stay at current distances
        new_returns = current_dist_to_depot.copy()
        new_returns[active_idx] = active_new_return
        new_max = np.max(new_returns)
        new_mean = np.mean(new_returns)
        score = new_max + w_mean * new_mean
        # Tie-breaking by active truck's own return time
        if (score < best_score) or (np.isclose(score, best_score) and active_new_return < best_own_return):
            best_score = score
            best_new_max = new_max
            best_own_return = active_new_return
            best_idx = i
    # Waiting decision: if best assignment increases max return beyond threshold, wait
    if best_new_max > current_max + epsilon:
        return None
    return best_idx