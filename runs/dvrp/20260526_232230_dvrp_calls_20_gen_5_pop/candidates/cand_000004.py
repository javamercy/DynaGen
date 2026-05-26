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
    # Compute distances from each truck to depot
    other_trucks = truck_positions != current_position  # but careful: positions might be identical? Assume unique
    # Actually, to identify the current truck, we compare positions. But if two trucks at same point, this fails.
    # Better: assume current_position is one of the truck_positions, but not known which index.
    # Instead, simply compute for all trucks and then replace the active truck's estimated return with the candidate.
    # Compute distance from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # For each customer candidate
    best_customer = None
    best_max_return = np.inf
    for i, cust in enumerate(available_customers):
        # Active truck's distance to customer and then to depot
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_cust_to_depot = np.linalg.norm(cust - depot_position)
        active_return = dist_to_cust + dist_cust_to_depot
        # Build array of all truck estimated return times: other trucks use current dist_to_depot, active uses candidate
        # We need to replace the active truck's value. Since we don't know its index, subtract its current dist and add candidate.
        # But active truck's current dist_to_depot is its distance from current_position to depot.
        active_current_dist = np.linalg.norm(current_position - depot_position)
        # This is one of the values in dist_to_depot. We'll compute max as:
        # max of (other trucks' dist_to_depot, and active_return)
        # Since other trucks might include the active if positions coincide, but we'll compute max without replacement.
        # To be safe, compute max as max of active_return and all dist_to_depot that are not from active truck.
        # But we can compute an approximate: max of all dist_to_depot and active_return, then subtract the active's current if it's the max? 
        # Simpler: compute max of (dist_to_depot) and active_return, but if active's current dist is the max, that might overcount.
        # Actually, we want to replace the active's current estimated return with the candidate.
        # So construct a list: for each truck, if it's the active truck, use active_return; else use its dist_to_depot.
        # Since we don't have index, we can compute the max as:
        # max(active_return, np.max(dist_to_depot[dist_to_depot != active_current_dist])) but if multiple trucks have same distance, that's wrong.
        # Instead, compute the max by taking the maximum of (active_return and all dist_to_depot except one that we assume is active). But we don't know which.
        # Alternative: assume the active truck is the one with position == current_position. Use np.where to find index.
        # This is robust: find index where truck_positions equals current_position (with tolerance for floating point).
        # Use np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)) to get closest, and if distance zero, assume it's the active.
        active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
        # Now build return times list
        return_times = dist_to_depot.copy()
        return_times[active_idx] = active_return
        max_return = np.max(return_times)
        if max_return < best_max_return:
            best_max_return = max_return
            best_customer = i
    return best_customer