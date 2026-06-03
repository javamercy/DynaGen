import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Determine which truck is current
    if len(truck_positions) == 1:
        current_truck_idx = 0
    else:
        dists = np.linalg.norm(truck_positions - current_position, axis=1)
        current_truck_idx = np.argmin(dists)
    other_mask = np.ones(len(truck_positions), dtype=bool)
    other_mask[current_truck_idx] = False
    other_trucks = truck_positions[other_mask]
    n_other = len(other_trucks)
    if n_other > 0:
        other_direct_return = np.linalg.norm(other_trucks - depot_position, axis=1)
        max_other_return = np.max(other_direct_return)
    else:
        max_other_return = None

    best_regret = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if n_other > 0:
            alt_costs = np.linalg.norm(other_trucks - cust, axis=1) + np.linalg.norm(cust - depot_position)
            min_alt = np.min(alt_costs)
        else:
            min_alt = cost_now
        regret = cost_now - min_alt
        if n_other > 0:
            penalty = max(0.0, cost_now - max_other_return)
            regret += penalty
        if regret < best_regret or (regret == best_regret and cost_now < best_cost_now):
            best_regret = regret
            best_idx = i
            best_cost_now = cost_now
    if best_regret > 0 and np.linalg.norm(current_position - depot_position) < 1e-6:
        return None
    return best_idx