import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Current truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    # Current truck's distance to depot (for wait condition)
    current_to_depot = np.linalg.norm(current_position - depot_position)
    at_depot = np.isclose(current_to_depot, 0.0)
    # Distances of all trucks to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Max distance among other trucks to depot
    max_other_return = np.max(np.delete(dist_to_depot, current_idx))
    
    best_adj_regret = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Alternative costs: each truck serves this customer and returns
        alt_costs = np.array([np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions])
        # Best alternative (exclude current truck if its cost is equal to cost_now)
        sorted_idx = np.argsort(alt_costs)
        if len(alt_costs) > 1 and np.isclose(alt_costs[sorted_idx[0]], cost_now):
            min_alt = alt_costs[sorted_idx[1]]
        else:
            min_alt = alt_costs[sorted_idx[0]]
        regret = cost_now - min_alt
        # Fleet-balance penalty: if cost_now exceeds max other return, add excess
        if cost_now > max_other_return:
            penalty = cost_now - max_other_return
        else:
            penalty = 0.0
        adj_regret = regret + penalty
        # Tie-break by cost_now
        if adj_regret < best_adj_regret or (np.isclose(adj_regret, best_adj_regret) and cost_now < best_cost_now):
            best_adj_regret = adj_regret
            best_idx = i
            best_cost_now = cost_now
    # Wait at depot if best adjusted regret > 0
    if at_depot and best_adj_regret > 0:
        return None
    return best_idx