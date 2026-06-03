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
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    
    # Identify current truck index
    current_truck_idx = int(np.argmin(
        np.linalg.norm(truck_positions - current_position, axis=1)
    ))
    
    # Precompute distances from each truck to depot (direct return)
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    
    # For the current truck, its direct return to depot
    curr_to_depot = truck_to_depot[current_truck_idx]
    
    best_regret = -float('inf')
    best_idx = None
    best_secondary = -float('inf')
    
    for i, cust in enumerate(available_customers):
        # Distance from current truck to customer
        curr_to_cust = np.linalg.norm(current_position - cust)
        # Current truck's return if it serves this customer
        curr_return = curr_to_cust + np.linalg.norm(cust - depot_position)
        
        # TTT if current truck serves: max of its return and other trucks going directly to depot
        ttt_current = max(curr_return, np.max(np.delete(truck_to_depot, current_truck_idx)))
        
        # Compute best TTT among other trucks serving this customer
        min_other_ttt = float('inf')
        for j in range(n_trucks):
            if j == current_truck_idx:
                continue
            # Return of truck j if it serves this customer
            j_return = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            # TTT if truck j serves: max of its return, current truck's direct return, and all other trucks' direct returns (excluding j and current)
            other_depots = np.delete(truck_to_depot, [current_truck_idx, j])
            max_other = np.max(other_depots) if len(other_depots) > 0 else 0
            ttt_j = max(j_return, curr_to_depot, max_other)
            if ttt_j < min_other_ttt:
                min_other_ttt = ttt_j
        
        # Regret: improvement in TTT if current truck serves instead of best other
        regret = min_other_ttt - ttt_current
        # Secondary: smaller current return is better
        secondary = -curr_return
        
        if (regret > best_regret) or (regret == best_regret and secondary > best_secondary):
            best_regret = regret
            best_idx = i
            best_secondary = secondary
    
    if best_regret >= 0:
        return best_idx
    else:
        return None