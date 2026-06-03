import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # Parameter: minimum improvement fraction to dispatch
    min_improvement_ratio = 0.05

    # Compute distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(truck_to_depot)

    # Compute for each available customer: new return time for active truck
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_to_cust + cust_to_depot

    # Candidate new max = max(new_return_time for active, current_max for others)
    candidate_max = np.maximum(new_return_times, current_max)

    # Find customer with smallest candidate_max
    best_idx = np.argmin(candidate_max)
    min_val = candidate_max[best_idx]

    # Tie-breaking by new_return_time
    ties = np.where(candidate_max == min_val)[0]
    if len(ties) > 1:
        best_idx_in_ties = np.argmin(new_return_times[ties])
        best_idx = ties[best_idx_in_ties]

    # Compute improvement
    best_new_max = candidate_max[best_idx]
    improvement = current_max - best_new_max
    # If improvement is small (relative to current_max) and current_max > 0, wait
    if current_max > 0 and improvement / current_max < min_improvement_ratio:
        return None
    else:
        return int(best_idx)