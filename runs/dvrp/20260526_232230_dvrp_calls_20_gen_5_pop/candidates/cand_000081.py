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
    best_customer = None
    best_score = np.inf
    best_active_return = np.inf
    # Weights for penalties
    imbalance_weight = 0.1
    depot_weight = 0.5
    for i, cust in enumerate(available_customers):
        # Active truck's estimated return if it goes to this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Build return times array with active truck's value replaced
        return_times = dist_to_depot.copy()
        return_times[active_idx] = active_return
        max_return = np.max(return_times)
        # Penalize imbalance (gap) if active truck's return is less than max
        gap = max_return - active_return if active_return < max_return else 0.0
        # Customer's distance to depot
        cust_depot_dist = np.linalg.norm(cust - depot_position)
        # Score
        score = max_return + imbalance_weight * gap + depot_weight * cust_depot_dist
        # Tie-break by active truck's own return time
        if score < best_score or (score == best_score and active_return < best_active_return):
            best_score = score
            best_active_return = active_return
            best_customer = i
    return best_customer