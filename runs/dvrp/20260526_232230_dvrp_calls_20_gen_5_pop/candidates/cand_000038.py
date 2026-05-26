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
    # Identify active truck as the one closest to current_position
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Current distances to depot for all trucks
    current_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_current_depot_dist = current_depot_dists[active_idx]
    # Average depot distance of other trucks
    other_depot_dists = np.delete(current_depot_dists, active_idx)
    mean_other_depot_dist = np.mean(other_depot_dists) if len(other_depot_dists) > 0 else 0.0
    active_behind = active_current_depot_dist > mean_other_depot_dist
    
    best_idx = None
    best_score = np.inf
    beta = 0.2
    alpha = 0.1   # imbalance weight
    penalty_weight = 0.1  # penalty for far customer when active is behind
    
    for i, cust in enumerate(available_customers):
        # Return time for active truck if it serves this customer then goes to depot
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # For other trucks, assume they go directly to depot from current position
        returns = other_depot_dists.copy()  # distances for other trucks
        # Insert active truck's new return at active_idx
        returns = np.insert(returns, active_idx, active_return)
        max_r = np.max(returns)
        mean_r = np.mean(returns)
        min_r = np.min(returns)
        imbalance = max_r - min_r
        score = max_r + beta * mean_r + alpha * imbalance
        if active_behind:
            customer_depot_dist = np.linalg.norm(cust - depot_position)
            score += penalty_weight * customer_depot_dist
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx