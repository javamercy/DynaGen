import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_regret = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        alt_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        min_alt = min(alt_costs)
        regret = cost_now - min_alt
        if regret < best_regret or (regret == best_regret and cost_now < best_cost_now):
            best_regret = regret
            best_idx = i
            best_cost_now = cost_now
    if best_regret > 0 and np.linalg.norm(current_position - depot_position) < 1e-6:
        return None
    return best_idx