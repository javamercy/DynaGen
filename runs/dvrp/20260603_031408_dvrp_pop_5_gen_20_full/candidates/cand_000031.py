def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # find index of current truck (closest position)
    diff = truck_positions - current_position
    dists = np.linalg.norm(diff, axis=1)
    current_idx = np.argmin(dists)
    
    # compute own remaining distance to depot
    own_remaining = np.linalg.norm(current_position - depot_position)
    
    # compute max remaining of other trucks
    other_remaining = [np.linalg.norm(truck_positions[j] - depot_position) for j in range(len(truck_positions)) if j != current_idx]
    other_max = max(other_remaining) if other_remaining else 0.0
    wait_TTT = max(own_remaining, other_max)
    
    best_idx = None
    best_ttt = float('inf')
    best_own = float('inf')
    
    for i, cust in enumerate(available_customers):
        candidate_own = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        candidate_ttt = max(candidate_own, other_max)
        
        if candidate_ttt < best_ttt or (candidate_ttt == best_ttt and candidate_own < best_own):
            best_ttt = candidate_ttt
            best_own = candidate_own
            best_idx = i
    
    # only take if it doesn't worsen makespan (allowing equality)
    if best_ttt <= wait_TTT:
        return best_idx
    else:
        return None