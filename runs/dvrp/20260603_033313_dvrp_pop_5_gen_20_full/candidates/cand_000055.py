import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_score = float('inf')
    best_cost = float('inf')
    best_idx = None
    for i, cust in enumerate(available_customers):
        cost_cur = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_costs = []
        for truck in truck_positions:
            if not np.allclose(truck, current_position):
                other_costs.append(np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position))
        if len(other_costs) == 0:
            min_other = cost_cur
            max_other = cost_cur
        else:
            min_other = min(other_costs)
            max_other = max(other_costs)
        regret = cost_cur - min_other
        bottleneck_penalty = max(0, cost_cur - max_other)
        score = regret + 2 * bottleneck_penalty
        if score < best_score or (np.isclose(score, best_score) and cost_cur < best_cost):
            best_score = score
            best_cost = cost_cur
            best_idx = i
    at_depot = np.linalg.norm(current_position - depot_position) < 1e-6
    if at_depot and best_score > 1e-6:
        return None
    return best_idx