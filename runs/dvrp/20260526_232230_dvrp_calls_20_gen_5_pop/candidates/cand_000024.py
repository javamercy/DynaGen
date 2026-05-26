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
    # Distances from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Active truck index (closest to current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    active_current_dist = dist_to_depot[active_idx]
    # Other trucks' distances to depot (excluding active)
    other_dist = np.delete(dist_to_depot, active_idx)
    max_other = np.max(other_dist) if len(other_dist) > 0 else 0.0
    # Fleet imbalance: max - min among all trucks
    imbalance = np.max(dist_to_depot) - np.min(dist_to_depot)
    mean_dist = np.mean(dist_to_depot)
    behind = active_current_dist > mean_dist
    # Penalty weight
    alpha = 0.1
    best_customer = None
    best_score = np.inf
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        M_i = max(active_return, max_other)
        # Normalize customer depot distance (use max among available to avoid scale issues)
        max_cust_dist = np.max(np.linalg.norm(available_customers - depot_position, axis=1))
        cust_dist_norm = np.linalg.norm(cust - depot_position) / max_cust_dist if max_cust_dist > 0 else 0.0
        if behind:
            score = M_i + alpha * imbalance * cust_dist_norm
        else:
            score = M_i - alpha * imbalance * cust_dist_norm
        if score < best_score:
            best_score = score
            best_customer = i
    return best_customer