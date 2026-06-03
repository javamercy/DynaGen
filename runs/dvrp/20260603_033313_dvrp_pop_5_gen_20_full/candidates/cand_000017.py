import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_cust = len(available_customers)
    n_trucks = len(truck_positions)
    d_to = np.linalg.norm(current_position - available_customers, axis=1)
    d_from = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = d_to + d_from
    current_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    min_other = np.full(n_cust, np.inf)
    for i in range(n_cust):
        cust = available_customers[i]
        dists = np.linalg.norm(truck_positions - cust, axis=1)
        dists[current_idx] = np.inf
        min_other[i] = np.min(dists)
    regret = d_to - min_other
    exclusive = d_to <= min_other + 1e-6
    if np.any(exclusive):
        exclusive_indices = np.where(exclusive)[0]
        best_idx = exclusive_indices[np.argmin(d_from[exclusive_indices])]
        return int(best_idx)
    else:
        at_depot = np.linalg.norm(current_position - depot_position) < 1e-6
        best_regret = np.min(regret)
        if at_depot and best_regret > 0:
            return None
        else:
            min_regret = np.min(regret)
            candidates = np.where(np.abs(regret - min_regret) < 1e-6)[0]
            best_idx = candidates[np.argmin(cost_now[candidates])]
            return int(best_idx)