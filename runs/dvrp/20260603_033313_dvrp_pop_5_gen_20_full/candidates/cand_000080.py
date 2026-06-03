import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = truck_positions.shape[0]
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    cost_me = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
    if n_trucks == 1:
        return int(np.argmin(cost_me))
    # compute regret
    best_other = np.full(len(available_customers), np.inf)
    current_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            current_idx = i
            break
    for j, cust in enumerate(available_customers):
        for k in range(n_trucks):
            if k == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[k] - cust) + np.linalg.norm(cust - depot_position)
            if other_cost < best_other[j]:
                best_other[j] = other_cost
    regret = cost_me - best_other
    candidates = np.where(regret <= 0)[0]
    if len(candidates) == 0:
        return None
    # compute current max from other trucks
    others_max = np.max(np.delete(depot_dists, current_idx))
    candidate_costs = cost_me[candidates]
    new_max = np.maximum(others_max, candidate_costs)
    # select best candidate: minimize new_max, then regret, then cost_me
    min_new_max = np.min(new_max)
    best_candidates = candidates[new_max == min_new_max]
    best_idx = best_candidates[0]
    best_regret = regret[best_idx]
    best_cost = cost_me[best_idx]
    for idx in best_candidates:
        if regret[idx] < best_regret or (regret[idx] == best_regret and cost_me[idx] < best_cost):
            best_idx = idx
            best_regret = regret[idx]
            best_cost = cost_me[idx]
    return int(best_idx)