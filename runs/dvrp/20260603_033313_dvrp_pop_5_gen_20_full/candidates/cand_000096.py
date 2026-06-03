import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_regret = float('inf')
    best_idx = None
    best_cost = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(all_costs)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[0], cost_now, atol=1e-8):
            min_alt = sorted_costs[1]
        else:
            min_alt = sorted_costs[0]
        regret = cost_now - min_alt
        if regret < best_regret or (regret == best_regret and cost_now < best_cost):
            best_regret = regret
            best_idx = i
            best_cost = cost_now
    if best_regret > 0 and best_regret > 0.5 * best_cost:
        return None
    return best_idx