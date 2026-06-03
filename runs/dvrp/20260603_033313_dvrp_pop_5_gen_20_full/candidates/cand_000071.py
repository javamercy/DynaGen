import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_available = len(available_customers)
    penalty_factor = 1.0 + n_available / (n_available + 10.0)
    # compute current distances to depot for all trucks
    all_dist_to_depot = [np.linalg.norm(p - depot_position) for p in truck_positions]
    active_dist = np.linalg.norm(current_position - depot_position)
    # find max distance among other trucks
    sorted_dists = sorted(all_dist_to_depot)
    if np.isclose(sorted_dists[-1], active_dist, atol=1e-8):
        max_other_dist = sorted_dists[-2] if len(sorted_dists) > 1 else 0.0
    else:
        max_other_dist = sorted_dists[-1]
    
    best_regret = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(all_costs)
        # min alternative (excluding current truck)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[0], cost_now, atol=1e-8):
            min_alt = sorted_costs[1]
        else:
            min_alt = sorted_costs[0]
        # max alternative (excluding current truck)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[-1], cost_now, atol=1e-8):
            max_alt = sorted_costs[-2]
        else:
            max_alt = sorted_costs[-1]
        regret = (cost_now - min_alt) + penalty_factor * max(0, cost_now - max_alt)
        # depot pressure term
        depot_penalty = max(0, cost_now - max_other_dist) * 0.5
        regret += depot_penalty
        if regret < best_regret or (regret == best_regret and cost_now < best_cost_now):
            best_regret = regret
            best_idx = i
            best_cost_now = cost_now
    # Wait if best_regret is positive and significant relative to cost_now
    if best_regret > 0 and best_regret > 0.1 * best_cost_now:
        return None
    return best_idx