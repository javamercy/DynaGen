import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    # Identify current truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    # Direct returns from current truck positions to depot
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    if n_trucks > 1:
        max_other_return = np.max(np.delete(depot_dists, current_idx))
    else:
        max_other_return = -np.inf
    best_overall = float('inf')
    best_idx = None
    best_cost = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if n_trucks == 1:
            overall = cost_now
        else:
            # Compute best alternative cost from other trucks
            alt_costs = np.linalg.norm(truck_positions - cust, axis=1) + np.linalg.norm(cust - depot_position)
            sorted_costs = np.sort(alt_costs)
            if np.isclose(sorted_costs[0], cost_now):
                min_alt = sorted_costs[1]
            else:
                min_alt = sorted_costs[0]
            regret = cost_now - min_alt
            excess = max(0, cost_now - max_other_return)
            overall = regret + excess
        if overall < best_overall or (overall == best_overall and cost_now < best_cost):
            best_overall = overall
            best_idx = i
            best_cost = cost_now
    # Wait at depot if overall > 0
    if best_overall > 0 and np.linalg.norm(current_position - depot_position) < 1e-6:
        return None
    return best_idx