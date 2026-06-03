import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # Identify active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_dist = truck_to_depot[active_idx]
    other_dist = np.delete(truck_to_depot, active_idx)
    if other_dist.size > 0:
        other_max = np.max(other_dist)
        other_min = np.min(other_dist)
    else:
        other_max = -np.inf
        other_min = np.inf

    # Distances to and from customers
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_to_cust + cust_to_depot

    overall_max = np.max(truck_to_depot)
    candidate_max = np.maximum(new_return_times, overall_max)

    # Choose customer with min candidate_max, tie-break by new_return_time
    best_idx = int(np.argmin(candidate_max))
    min_val = candidate_max[best_idx]
    ties = np.where(candidate_max == min_val)[0]
    if len(ties) > 1:
        best_idx = int(ties[np.argmin(new_return_times[ties])])

    # Waiting condition: active truck closest to depot and dispatch would increase max
    if other_dist.size > 0 and active_dist < other_min and min_val > other_max:
        return None

    return best_idx