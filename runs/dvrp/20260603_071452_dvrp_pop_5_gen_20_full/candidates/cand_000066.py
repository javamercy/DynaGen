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
    # find active index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position, atol=1e-8):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not in truck_positions")
    
    # distances to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_dist = truck_depot_dists[active_idx]
    
    # single truck case
    if n_trucks == 1:
        best_idx = None
        best_T = np.inf
        for i, cust in enumerate(available_customers):
            T = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            if T < best_T:
                best_T = T
                best_idx = i
        return best_idx
    
    # multiple trucks
    max_other = max(truck_depot_dists[j] for j in range(n_trucks) if j != active_idx)
    new_max_wait = max(max_other, active_dist)
    
    best_new_max = np.inf
    best_candidates = []
    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        cust_to_depot = np.linalg.norm(cust - depot_position)
        T_active = dist_to_cust + cust_to_depot
        new_max_serve = max(max_other, T_active)
        if new_max_serve < best_new_max:
            best_new_max = new_max_serve
            best_candidates = [(i, T_active, cust_to_depot)]
        elif new_max_serve == best_new_max:
            best_candidates.append((i, T_active, cust_to_depot))
    
    if best_new_max <= new_max_wait:
        # tie-break: smallest T_active, then smallest cust_to_depot
        best_idx = min(best_candidates, key=lambda x: (x[1], x[2]))[0]
        return best_idx
    else:
        return None