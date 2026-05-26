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
    best_idx = None
    best_score = np.inf
    beta = 0.4
    gamma = 0.05
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        returns = dist_to_depot.copy()
        returns[active_idx] = active_return
        max_r = np.max(returns)
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        score = max_r + beta * mean_r + gamma * std_r
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx