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
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(dist_to_depot)
    # waiting threshold: 5% increase
    threshold = current_max * 1.05
    best_idx = None
    best_score = np.inf
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        returns = dist_to_depot.copy()
        returns[active_idx] = active_return
        new_max = np.max(returns)
        if new_max <= threshold:  # only consider customers that don't increase max too much
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            score = new_max + 0.3 * mean_r + 0.1 * std_r
            if score < best_score:
                best_score = score
                best_idx = i
    if best_idx is None:
        return None  # wait, because all candidates exceed threshold
    return best_idx