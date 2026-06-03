import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_score = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    best_regret = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        alt_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_asc = sorted(alt_costs)
        sorted_desc = sorted(alt_costs, reverse=True)
        # best alternative (minimum among other trucks)
        if len(sorted_asc) > 1 and np.isclose(sorted_asc[0], cost_now):
            min_alt = sorted_asc[1]
        else:
            min_alt = sorted_asc[0]
        # worst alternative (maximum among other trucks)
        if len(sorted_desc) > 1 and np.isclose(sorted_desc[0], cost_now):
            max_alt = sorted_desc[1]
        else:
            max_alt = sorted_desc[0]
        regret = cost_now - min_alt
        penalty = max(0, cost_now - max_alt)
        score = regret + penalty
        if score < best_score or (np.isclose(score, best_score) and cost_now < best_cost_now):
            best_score = score
            best_idx = i
            best_cost_now = cost_now
            best_regret = regret
    # Wait at depot if best_regret > 0
    if best_regret > 0 and np.linalg.norm(current_position - depot_position) < 1e-6:
        return None
    return best_idx