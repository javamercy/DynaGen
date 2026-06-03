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
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    n_avail = len(available_customers)
    threshold = 1.2 if n_avail <= 5 else 1.1
    
    best_index = None
    best_cost = np.inf
    best_active_time = np.inf
    best_depot_dist = np.inf
    
    for i in range(n_avail):
        cust = available_customers[i]
        active_time = np.linalg.norm(current_position - cust) + depot_dists[i]
        
        if n_trucks == 1:
            # only one truck, always consider
            if active_time < best_active_time or (active_time == best_active_time and depot_dists[i] < best_depot_dist):
                best_index = i
                best_active_time = active_time
                best_depot_dist = depot_dists[i]
            continue
        
        # compute min_other_time
        min_other_time = np.inf
        for j in range(n_trucks):
            if j == active_idx:
                continue
            other_time = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            if other_time < min_other_time:
                min_other_time = other_time
        
        # determine if consider
        if active_time <= min_other_time:
            consider = True
        else:
            consider = (active_time <= threshold * min_other_time)
        
        if not consider:
            continue
        
        cost = max(active_time, min_other_time)
        # tie-break: cost, then active_time, then depot distance
        if cost < best_cost or (cost == best_cost and active_time < best_active_time) or (cost == best_cost and active_time == best_active_time and depot_dists[i] < best_depot_dist):
            best_index = i
            best_cost = cost
            best_active_time = active_time
            best_depot_dist = depot_dists[i]
    
    return best_index