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
    # Distance from each truck to depot (current return times if they head straight back)
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_return = np.max(dist_to_depot)
    # Active truck index (closest to current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    best_customer = None
    best_score = np.inf
    best_active_return = np.inf
    for i, cust in enumerate(available_customers):
        # Active truck's estimated return if it goes to this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Build return times array with active truck's value replaced
        return_times = dist_to_depot.copy()
        return_times[active_idx] = active_return
        # Compute score components
        max_ret = np.max(return_times)
        mean_ret = np.mean(return_times)
        std_ret = np.std(return_times)
        score = max_ret + 0.3 * mean_ret + 0.1 * std_ret
        # Tie-break by active return
        if score < best_score or (score == best_score and active_return < best_active_return):
            best_score = score
            best_active_return = active_return
            best_customer = i
    # Wait if best new max return is more than 1.1x current max return (i.e., would significantly increase)
    if best_customer is not None:
        # Recompute max return for best customer
        cust = available_customers[best_customer]
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        return_times = dist_to_depot.copy()
        return_times[active_idx] = active_return
        new_max = np.max(return_times)
        if new_max > current_max_return * 1.1:
            return None
    return best_customer