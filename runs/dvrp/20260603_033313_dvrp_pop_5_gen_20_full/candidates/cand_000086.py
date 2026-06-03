import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = truck_positions.shape[0]
    cost_me = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
    if n_trucks == 1:
        return int(np.argmin(cost_me))
    # find index of current truck
    my_idx = np.where((truck_positions == current_position).all(axis=1))[0][0]
    other_indices = [i for i in range(n_trucks) if i != my_idx]
    other_pos = truck_positions[other_indices]
    # current max direct return time for other trucks
    max_other_direct = max(np.linalg.norm(other_pos - depot_position, axis=1))
    # precompute min cost among other trucks for each customer
    min_other_cost = np.min([np.linalg.norm(pos - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1) for pos in other_pos], axis=0)
    regret = cost_me - min_other_cost
    # only consider customers where current truck is at least as good as the best other
    candidates = np.where(regret <= 0)[0]
    if len(candidates) == 0:
        return None
    # compute new max if we serve each candidate
    new_max = np.maximum(cost_me[candidates], max_other_direct)
    # select best: min new_max, then min cost_me, then min regret (most negative)
    best_idx = candidates[0]
    best_new = new_max[0]
    best_cost = cost_me[candidates[0]]
    best_regret = regret[candidates[0]]
    for i, idx in enumerate(candidates):
        if new_max[i] < best_new or (new_max[i] == best_new and (cost_me[idx] < best_cost or (cost_me[idx] == best_cost and regret[idx] < best_regret))):
            best_idx = idx
            best_new = new_max[i]
            best_cost = cost_me[idx]
            best_regret = regret[idx]
    return int(best_idx)