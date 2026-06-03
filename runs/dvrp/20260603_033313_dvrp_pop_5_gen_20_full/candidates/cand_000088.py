import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        cost_me = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
        return int(np.argmin(cost_me))
    # Multi-truck
    # Find index of current truck
    idx_current = None
    for i in range(n_trucks):
        if np.array_equal(truck_positions[i], current_position):
            idx_current = i
            break
    # Compute distances to depot for all trucks
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    other_max = np.max(np.delete(depot_dists, idx_current))
    # Precompute cost_me and best_other for each customer
    n_cust = len(available_customers)
    regret = np.zeros(n_cust)
    original_cost = np.zeros(n_cust)
    for j, cust in enumerate(available_cust):
        cost_me = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        original_cost[j] = cost_me
        pressure = max(0, cost_me - other_max)
        adjusted_cost_me = cost_me + pressure
        best_other = np.inf
        for k in range(n_trucks):
            if k == idx_current:
                continue
            other_cost = np.linalg.norm(truck_positions[k] - cust) + np.linalg.norm(cust - depot_position)
            if other_cost < best_other:
                best_other = other_cost
        regret[j] = adjusted_cost_me - best_other
    min_regret = np.min(regret)
    if min_regret > 0:
        return None
    # Among those with regret <= 0, pick minimal regret, tie-break by original cost
    candidates = np.where(regret <= 0)[0]
    best_idx = candidates[0]
    best_regret = regret[best_idx]
    best_orig = original_cost[best_idx]
    for idx in candidates:
        if regret[idx] < best_regret or (regret[idx] == best_regret and original_cost[idx] < best_orig):
            best_idx = idx
            best_regret = regret[idx]
            best_orig = original_cost[idx]
    return int(best_idx)