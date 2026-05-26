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
    # Identify active truck index (closest to current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Current distances from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    best_idx = None
    best_score = np.inf
    w_max = 1.0
    w_mean = 0.2
    w_depot = 0.2
    for i, cust in enumerate(available_customers):
        # Active truck's return time if it serves this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Build return times: others stay at current distances
        returns = dist_to_depot.copy()
        returns[active_idx] = active_return
        max_r = np.max(returns)
        mean_r = np.mean(returns)
        depot_dist = np.linalg.norm(cust - depot_position)
        score = w_max * max_r + w_mean * mean_r + w_depot * depot_dist
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx