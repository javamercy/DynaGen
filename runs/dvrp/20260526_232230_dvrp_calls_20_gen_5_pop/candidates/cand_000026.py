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
    # Identify the active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Current distances from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    best_idx = None
    best_score = np.inf
    alpha = 0.1  # imbalance penalty weight
    for i, cust in enumerate(available_customers):
        # Active truck's estimated return time if it serves this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Build return times: others stay at current distances
        returns = dist_to_depot.copy()
        returns[active_idx] = active_return
        max_r = np.max(returns)
        min_r = np.min(returns)
        score = max_r + alpha * (max_r - min_r)
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx