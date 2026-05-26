def choose_next_customer(current_position, depot_position, truck_positions, available_customers, current_time):
    import numpy as np
    if len(available_customers) == 0:
        return None
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    d_active = np.linalg.norm(available_customers - current_position, axis=1)
    active_return = d_active + cust_to_depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(truck_to_depot)
    increase_active = np.maximum(0, active_return - current_max)
    n_trucks = len(truck_positions)
    n_avail = len(available_customers)
    best_other_dist = np.empty(n_avail)
    for i in range(n_avail):
        cust = available_customers[i]
        dists_to_all = np.linalg.norm(truck_positions - cust, axis=1)
        if n_trucks == 1:
            best_other_dist[i] = np.inf
        else:
            sorted_dists = np.sort(dists_to_all)
            if np.abs(sorted_dists[0] - d_active[i]) < 1e-10:
                best_other_dist[i] = sorted_dists[1] if n_trucks >= 2 else np.inf
            else:
                best_other_dist[i] = sorted_dists[0]
    best_other_return = best_other_dist + cust_to_depot
    increase_other = np.maximum(0, best_other_return - current_max)
    regret = increase_active - increase_other
    valid = regret <= 0
    if not np.any(valid):
        return None
    candidates = np.where(valid)[0]
    best_idx = candidates[np.argmin(active_return[candidates])]
    return int(best_idx)