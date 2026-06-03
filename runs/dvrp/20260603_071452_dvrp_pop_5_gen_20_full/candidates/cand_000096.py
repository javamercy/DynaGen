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
    
    # distances to depot for all trucks
    dists_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(dists_to_depot)
    
    # single truck case: always pick best active round-trip
    if n_trucks == 1:
        best_idx = 0
        best_cost = np.inf
        for i, cust in enumerate(available_customers):
            active_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            if active_cost < best_cost:
                best_cost = active_cost
                best_idx = i
        return best_idx
    
    best_idx = None
    best_new_max = np.inf
    best_active_cost = np.inf
    
    other_indices = [i for i in range(n_trucks) if i != active_idx]
    other_dists = dists_to_depot[other_indices]
    max_other = np.max(other_dists) if len(other_dists) > 0 else 0.0
    
    for i, cust in enumerate(available_customers):
        active_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(active_cost, max_other)
        if (new_max < best_new_max) or (new_max == best_new_max and active_cost < best_active_cost):
            best_new_max = new_max
            best_active_cost = active_cost
            best_idx = i
    
    # only assign if it does not increase the current max
    if best_new_max <= current_max:
        return best_idx
    else:
        return None