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
    # Current max return if all go to depot now
    current_max = np.max(dist_to_depot)
    # Estimate current return for active truck if it heads to depot now
    active_current_return = np.linalg.norm(current_position - depot_position)
    current_max_actual = max(current_max, active_current_return)
    
    best_idx = None
    best_score = np.inf
    beta = 0.3
    gamma = 0.05
    epsilon = 1e-4
    
    scores = []
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        returns = dist_to_depot.copy()
        returns[active_idx] = active_return
        max_r = np.max(returns)
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        score = max_r + beta * mean_r + gamma * std_r
        scores.append((score, max_r, i))
        if score < best_score:
            best_score = score
            best_idx = i
    # Check waiting condition: if all candidates increase max return above current_max_actual + epsilon, wait
    min_max_r = min(s[1] for s in scores)
    if min_max_r > current_max_actual + epsilon:
        return None
    return best_idx