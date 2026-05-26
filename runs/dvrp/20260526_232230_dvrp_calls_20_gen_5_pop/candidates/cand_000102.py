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
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    best_customer = None
    best_score = np.inf
    best_active_return = np.inf
    penalty_weight = 0.5  # increased from 0.01
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        return_times = dist_to_depot.copy()
        return_times[active_idx] = active_return
        max_return = np.max(return_times)
        cust_depot_dist = np.linalg.norm(cust - depot_position)
        score = max_return + penalty_weight * cust_depot_dist
        if score < best_score or (score == best_score and active_return < best_active_return):
            best_score = score
            best_active_return = active_return
            best_customer = i
    return best_customer