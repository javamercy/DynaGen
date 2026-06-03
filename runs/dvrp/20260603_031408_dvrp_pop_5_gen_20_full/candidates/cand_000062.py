import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_dist = np.linalg.norm(current_position - depot_position)
    overall_max = np.max(truck_to_depot)
    threshold = 0.3
    if active_dist < threshold * overall_max and overall_max > 0:
        return None

    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_to_cust + cust_to_depot

    candidate_max = np.maximum(new_return_times, overall_max)
    best_idx = int(np.argmin(candidate_max))
    min_val = candidate_max[best_idx]
    ties = np.where(candidate_max == min_val)[0]
    if len(ties) > 1:
        best_idx = int(ties[np.argmin(new_return_times[ties])])

    return best_idx