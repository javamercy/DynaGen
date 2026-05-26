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
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dist_to_current)
    # Current distances to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_return = np.max(dist_to_depot).item()
    # Threshold for waiting (20% increase, or absolute small)
    threshold_relative = 1.2
    abs_threshold = 0.1  # if current_max_return is tiny, ignore waiting
    # Precompute scores and own returns
    n = len(available_customers)
    scores = np.empty(n)
    own_returns = np.empty(n)
    max_returns = np.empty(n)
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        returns = dist_to_depot.copy()
        returns[active_idx] = active_return
        max_r = np.max(returns).item()
        mean_r = np.mean(returns).item()
        std_r = np.std(returns).item()
        scores[i] = max_r + 0.3 * mean_r + 0.1 * std_r
        own_returns[i] = active_return
        max_returns[i] = max_r
    # Determine if waiting is beneficial
    if current_max_return > abs_threshold:
        valid_mask = max_returns <= threshold_relative * current_max_return
        if not np.any(valid_mask):
            return None
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmin(scores[valid_indices])]
        # Tie-breaking by own return
        best_score = scores[best_idx]
        candidates = np.where(scores == best_score)[0]
        if len(candidates) > 1:
            best_idx = candidates[np.argmin(own_returns[candidates])]
        return int(best_idx)
    else:
        # If current max is tiny, just pick best score
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        candidates = np.where(scores == best_score)[0]
        if len(candidates) > 1:
            best_idx = candidates[np.argmin(own_returns[candidates])]
        return int(best_idx)