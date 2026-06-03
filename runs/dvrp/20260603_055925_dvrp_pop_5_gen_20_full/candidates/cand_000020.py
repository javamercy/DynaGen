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
    
    # Find index of current truck (the one at current_position)
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    
    # Precompute distance from each customer to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    best_regret = -float('inf')
    best_idx = None
    best_secondary = -float('inf')
    
    for i, cust in enumerate(available_customers):
        # Distance from current truck to this customer
        curr_to_cust = np.linalg.norm(current_position - cust)
        # Potential return time if current truck serves this customer and goes back to depot
        pot_return_curr = curr_to_cust + cust_to_depot[i]
        
        # Compute potential return times for other trucks
        other_returns = []
        for j, pos in enumerate(truck_positions):
            if j == current_truck_idx:
                continue
            other_to_cust = np.linalg.norm(pos - cust)
            other_return = other_to_cust + cust_to_depot[i]
            other_returns.append(other_return)
        min_other = min(other_returns) if other_returns else float('inf')
        
        # Regret: how much better (or worse) it is for current truck to serve this customer
        regret = min_other - pot_return_curr
        # Secondary criterion: smaller pot_return_curr is better for tie-breaking
        secondary = -pot_return_curr
        
        if (regret > best_regret) or (regret == best_regret and secondary > best_secondary):
            best_regret = regret
            best_idx = i
            best_secondary = secondary
    
    # Only assign if regret >= 0 (current truck can serve and return at least as fast as any other)
    if best_regret >= 0:
        return best_idx
    else:
        return None