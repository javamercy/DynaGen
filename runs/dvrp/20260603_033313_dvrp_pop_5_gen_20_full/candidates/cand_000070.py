import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = truck_positions.shape[0]
    cost_me = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
    if n_trucks == 1:
        return int(np.argmin(cost_me))
    # for each customer, compute best cost among other trucks
    best_other = np.full(len(available_customers), np.inf)
    for j, cust in enumerate(available_customers):
        for k in range(n_trucks):
            if np.array_equal(truck_positions[k], current_position):
                continue
            other_cost = np.linalg.norm(truck_positions[k] - cust) + np.linalg.norm(cust - depot_position)
            if other_cost < best_other[j]:
                best_other[j] = other_cost
    regret = cost_me - best_other
    min_regret = np.min(regret)
    if min_regret > 0:
        return None
    # among those with regret <= 0, pick smallest regret (most negative) then smallest cost_me
    candidates = np.where(regret <= 0)[0]
    best_idx = candidates[0]
    best_regret = regret[best_idx]
    best_cost = cost_me[best_idx]
    for idx in candidates:
        if regret[idx] < best_regret or (regret[idx] == best_regret and cost_me[idx] < best_cost):
            best_idx = idx
            best_regret = regret[idx]
            best_cost = cost_me[idx]
    return int(best_idx)