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
    current_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_return = np.max(current_returns)
    best_idx = None
    best_score = np.inf
    beta = 0.3
    gamma = 0.1
    delta = 0.05
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_returns = current_returns.copy()
        new_returns[active_idx] = active_return
        max_r = np.max(new_returns)
        mean_r = np.mean(new_returns)
        var_r = np.var(new_returns)
        cust_depot_dist = np.linalg.norm(cust - depot_position)
        score = max_r + beta * mean_r + gamma * var_r + delta * cust_depot_dist
        if score < best_score:
            best_score = score
            best_idx = i
    # waiting condition: if best candidate increases max return by >10%
    if best_score >= current_max_return * 1.1:
        return None
    return best_idx