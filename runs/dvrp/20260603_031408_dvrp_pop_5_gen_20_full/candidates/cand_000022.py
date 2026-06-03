import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_dist = np.linalg.norm(current_position - depot_position)
    current_max = np.max(truck_to_depot)

    # identify active truck index
    active_idx = np.where((truck_positions == current_position).all(axis=1))[0]
    if len(active_idx) != 1:
        # fallback: assume first matching (should be unique)
        active_idx = active_idx[0]
    else:
        active_idx = active_idx[0]

    # max depot distance among other trucks (excluding active truck)
    other_mask = np.ones(truck_positions.shape[0], dtype=bool)
    other_mask[active_idx] = False
    if np.any(other_mask):
        other_max = np.max(truck_to_depot[other_mask])
    else:
        other_max = 0.0

    # compute candidate metrics
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_to_cust + cust_to_depot
    candidate_max = np.maximum(new_return_times, other_max)

    # find best candidate (minimizing candidate_max, then new_return_time)
    best_max_val = np.min(candidate_max)
    best_candidates = np.where(candidate_max == best_max_val)[0]
    if len(best_candidates) > 1:
        best_idx = best_candidates[np.argmin(new_return_times[best_candidates])]
    else:
        best_idx = best_candidates[0]

    # waiting condition: if active truck is near depot and any customer increases the max
    if best_max_val > current_max and current_dist < 0.5 * current_max:
        return None

    return int(best_idx)