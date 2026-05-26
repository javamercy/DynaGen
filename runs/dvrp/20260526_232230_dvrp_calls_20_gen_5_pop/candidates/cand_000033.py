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
    alpha = 0.5  # penalty weight for imbalance
    for i, cust in enumerate(available_customers):
        # Active truck's estimated return if it goes to this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Build return times array with active truck's value replaced
        return_times = dist_to_depot.copy()
        return_times[active_idx] = active_return
        max_return = np.max(return_times)
        min_return = np.min(return_times)
        imbalance = max_return - min_return
        score = max_return + alpha * imbalance
        if score < best_score:
            best_score = score
            best_customer = i
    return best_customer