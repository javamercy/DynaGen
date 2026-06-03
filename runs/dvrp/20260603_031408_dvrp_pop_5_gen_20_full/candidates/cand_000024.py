import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # Compute current distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    overall_max = np.max(truck_to_depot)

    # Distance from active truck to depot
    my_dist_to_depot = np.linalg.norm(current_position - depot_position)

    # For each available customer, compute new return time for active truck
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_to_cust + cust_to_depot

    # Candidate max: new return time for active truck vs current max of others
    candidate_max = np.maximum(new_return_times, overall_max)

    # Choose customer minimizing candidate max; tie-break by new return time
    best_idx = int(np.argmin(candidate_max))
    min_val = candidate_max[best_idx]
    ties = np.where(candidate_max == min_val)[0]
    if len(ties) > 1:
        best_idx = int(ties[np.argmin(new_return_times[ties])])

    # Compute maximum distance to depot among other trucks
    max_other = overall_max

    # Waiting condition: active truck close to depot and best customer far, and other trucks far
    if my_dist_to_depot < 0.3 * max_other:
        if dist_to_cust[best_idx] > 1.5 * my_dist_to_depot:
            return None

    return best_idx