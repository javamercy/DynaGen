import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    n_trucks = len(truck_positions)
    # find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # distances from each truck to depot
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(depot_dists)
    max_other = np.max(np.delete(depot_dists, active_idx))
    
    best_score = -np.inf
    best_idx = None
    
    for i, cust in enumerate(available_customers):
        # active truck's return if it serves this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        active_new_max = max(active_return, max_other)
        
        # compute best possible max if another truck serves
        other_returns = np.linalg.norm(truck_positions - cust, axis=1) + np.linalg.norm(cust - depot_position)
        best_other_new_max = np.inf
        for j in range(n_trucks):
            if j == active_idx:
                continue
            # max of depot distances excluding j
            d_excl_j = np.delete(depot_dists, j)
            max_excl_j = np.max(d_excl_j)
            candidate = max(other_returns[j], max_excl_j)
            if candidate < best_other_new_max:
                best_other_new_max = candidate
        
        improvement = best_other_new_max - active_new_max
        reduction = current_max - active_new_max
        score = improvement + 0.5 * reduction if improvement > 0 else improvement  # only positive improvement considered beneficial
        if score > best_score:
            best_score = score
            best_idx = i
    
    # if no positive improvement, wait
    if best_score <= 0:
        return None
    return best_idx