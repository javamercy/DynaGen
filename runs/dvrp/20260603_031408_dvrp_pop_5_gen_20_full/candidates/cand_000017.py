import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    n_trucks = truck_positions.shape[0]
    truck_dist = np.linalg.norm(truck_positions - depot_position, axis=1)
    overall_max = np.max(truck_dist)
    active_dist = np.linalg.norm(current_position - depot_position)

    # Identify active truck index
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0]
    if len(active_idx) == 0:
        # Should not happen; fallback: use active_dist to find closest truck
        active_idx = np.argmin(np.abs(truck_dist - active_dist))
    else:
        active_idx = active_idx[0]

    if n_trucks == 1:
        # Single truck: always dispatch, minimize new_return
        dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
        cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
        new_return = dist_to_cust + cust_to_depot
        best_idx = np.argmin(new_return)
        return int(best_idx)

    # Compute max over other trucks
    other_dists = np.delete(truck_dist, active_idx)
    max_other = np.max(other_dists) if len(other_dists) > 0 else 0.0

    # Compute new return times for each customer
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return = dist_to_cust + cust_to_depot

    # Candidate max values
    candidate_max = np.maximum(new_return, max_other)

    # Find best candidate (min candidate_max)
    best_idx = np.argmin(candidate_max)
    best_new_max = candidate_max[best_idx]

    # Wait if best new max > current overall max (i.e., would increase makespan)
    if best_new_max > overall_max:
        return None

    # Tie-break: smallest new_return among ties
    min_val = candidate_max[best_idx]
    ties = np.where(candidate_max == min_val)[0]
    if len(ties) > 1:
        best_idx_in_ties = np.argmin(new_return[ties])
        best_idx = ties[best_idx_in_ties]

    return int(best_idx)