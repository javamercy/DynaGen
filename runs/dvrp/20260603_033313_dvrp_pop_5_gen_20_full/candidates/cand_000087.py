import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_regret = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    best_min_alt = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(all_costs)
        # Find min alternative cost (excluding current truck's contribution)
        min_alt = sorted_costs[0]
        if np.isclose(min_alt, cost_now, atol=1e-8) and len(sorted_costs) > 1:
            min_alt = sorted_costs[1]
        # Find max alternative cost (excluding current truck's contribution)
        max_alt = sorted_costs[-1]
        if np.isclose(max_alt, cost_now, atol=1e-8) and len(sorted_costs) > 1:
            max_alt = sorted_costs[-2]
        # Adaptive penalty factor
        if max_alt > 0:
            penalty = 1.0 + 0.5 * (min_alt / max_alt)
        else:
            penalty = 1.5
        regret = (cost_now - min_alt) + penalty * max(0, cost_now - max_alt)
        if regret < best_regret:
            best_regret = regret
            best_idx = i
            best_cost_now = cost_now
            best_min_alt = min_alt
    # Wait if best regret is positive and cost is significantly higher than best alternative
    if best_regret > 0 and best_cost_now > 1.1 * best_min_alt:
        return None
    return best_idx