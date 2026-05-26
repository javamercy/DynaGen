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
    # Current distances to depot for all trucks
    current_dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = current_dist_to_depot.max()
    threshold_factor = 1.1
    beta = 0.2  # weight on mean for tie-breaking
    best_idx = None
    best_score = np.inf
    min_candidate_max = np.inf
    for i, cust in enumerate(available_customers):
        active_return = (np.linalg.norm(current_position - cust) +
                         np.linalg.norm(cust - depot_position))
        candidate_returns = current_dist_to_depot.copy()
        candidate_returns[active_idx] = active_return
        candidate_max = candidate_returns.max()
        candidate_mean = candidate_returns.mean()
        score = candidate_max + beta * candidate_mean
        if score < best_score:
            best_score = score
            best_idx = i
            min_candidate_max = candidate_max
    # If the best candidate increases max return by more than 10%, wait
    if min_candidate_max > current_max * threshold_factor:
        return None
    return best_idx