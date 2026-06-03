import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Compute distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Identify current truck index (closest match)
    distances_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(distances_to_current)
    # Maximum return time among other trucks
    max_other_return = np.max(np.delete(truck_to_depot, current_idx))
    best_score = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Alternative costs for all trucks
        alt_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(alt_costs)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[0], cost_now):
            min_alt = sorted_costs[1]
        else:
            min_alt = sorted_costs[0]
        regret = cost_now - min_alt
        # Imbalance penalty
        projected_return = cost_now
        if projected_return > max_other_return:
            penalty = projected_return - max_other_return
        else:
            penalty = 0.0
        score = regret + penalty
        if score < best_score or (score == best_score and cost_now < best_cost_now):
            best_score = score
            best_idx = i
            best_cost_now = cost_now
    # Wait if at depot and best_score > 0
    if best_score > 0 and np.linalg.norm(current_position - depot_position) < 1e-6:
        return None
    return best_idx