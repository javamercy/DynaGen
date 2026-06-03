import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # find index of current truck in truck_positions
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    best_score = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        all_costs = np.linalg.norm(truck_positions - cust, axis=1) + np.linalg.norm(cust - depot_position)
        # exclude current truck
        other_costs = np.delete(all_costs, current_idx)
        if len(other_costs) == 0:
            # single truck case
            best_alt = cost_now
            max_alt = cost_now
        else:
            best_alt = np.min(other_costs)
            max_alt = np.max(other_costs)
        regret = cost_now - best_alt
        penalty = max(0, cost_now - max_alt)
        score = regret + penalty
        if score < best_score or (score == best_score and cost_now < best_cost_now):
            best_score = score
            best_idx = i
            best_cost_now = cost_now
    # Wait if at depot and best_score > 0
    if best_score > 0 and np.linalg.norm(current_position - depot_position) < 1e-6:
        return None
    return best_idx