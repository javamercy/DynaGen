def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        # Only one truck: go to nearest customer
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    
    # Find index of current truck
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    
    # Distances from current truck to customers
    curr_dists = np.linalg.norm(available_customers - current_position, axis=1)
    current_to_depot = np.linalg.norm(current_position - depot_position)
    
    # Distances from other trucks to depot (lower bound on return time)
    other_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    other_to_depot = np.delete(other_to_depot, current_truck_idx)
    max_other_ret = np.max(other_to_depot) if len(other_to_depot) > 0 else 0.0
    
    best_score = -np.inf
    best_idx = None
    for i, cust in enumerate(available_customers):
        d_curr = curr_dists[i]
        # Compute distances from all trucks to this customer
        dists_to_cust = np.linalg.norm(truck_positions - cust, axis=1)
        # Exclude current truck's distance
        other_dists = np.delete(dists_to_cust, current_truck_idx)
        if len(other_dists) > 0:
            other_min = np.min(other_dists)
        else:
            other_min = float('inf')
        regret = other_min - d_curr
        if regret <= 0:
            continue
        cust_to_depot = np.linalg.norm(cust - depot_position)
        new_return = d_curr + cust_to_depot
        penalty = max(0.0, new_return - max_other_ret)
        score = regret - 0.5 * penalty
        if score > best_score:
            best_score = score
            best_idx = i
        elif score == best_score:
            # Tie-break: smaller current distance
            if d_curr < curr_dists[best_idx]:
                best_idx = i
    
    if best_idx is not None:
        return best_idx
    else:
        return None