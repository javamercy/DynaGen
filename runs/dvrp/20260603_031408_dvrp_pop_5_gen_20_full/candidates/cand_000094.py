import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # Find active truck index
    distances_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(distances_to_current)

    # Compute distances to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_to_depot = truck_to_depot[active_idx]

    # Maximum distance among other trucks (excluding active)
    if len(truck_positions) == 1:
        other_max = 0.0
    else:
        other_mask = np.ones(len(truck_positions), dtype=bool)
        other_mask[active_idx] = False
        other_max = np.max(truck_to_depot[other_mask])

    current_max = max(active_to_depot, other_max)

    # For each customer, compute new return time and new max
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_to_cust + cust_to_depot

    new_max = np.maximum(other_max, new_return_times)
    best_new_max = np.min(new_max)

    # If no customer can improve the current max, wait
    if best_new_max >= current_max:
        return None

    # Otherwise, choose the customer with smallest new max (tie-break by new_return_time)
    best_idx = int(np.argmin(new_max))
    min_val = new_max[best_idx]
    ties = np.where(new_max == min_val)[0]
    if len(ties) > 1:
        best_idx = int(ties[np.argmin(new_return_times[ties])])

    return best_idx