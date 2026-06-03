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

    # Find index of current truck
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    
    best_score = -float('inf')
    best_idx = None
    best_current_dist = None
    best_depot_dist = None
    
    for i, cust in enumerate(available_customers):
        current_dist = np.linalg.norm(current_position - cust)
        # distance from other trucks to cust
        other_dists = np.linalg.norm(truck_positions - cust, axis=1)
        other_dists = np.delete(other_dists, current_truck_idx)
        min_other = np.min(other_dists) if len(other_dists) > 0 else float('inf')
        regret = min_other - current_dist
        
        depot_dist = np.linalg.norm(cust - depot_position)
        # depot penalty weight 0.5
        score = regret - 0.5 * depot_dist
        
        # tie-breaking: better current_dist (smaller), then depot_dist (smaller)
        if (score > best_score) or \
           (score == best_score and (best_current_dist is None or current_dist < best_current_dist)) or \
           (score == best_score and current_dist == best_current_dist and (best_depot_dist is None or depot_dist < best_depot_dist)):
            best_score = score
            best_idx = i
            best_current_dist = current_dist
            best_depot_dist = depot_dist
    
    if best_score >= 0:
        return best_idx
    else:
        return None