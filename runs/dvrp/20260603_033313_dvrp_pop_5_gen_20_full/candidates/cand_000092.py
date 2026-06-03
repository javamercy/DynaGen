import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    # Identify active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Compute max distance from other trucks to depot
    others_dist = np.linalg.norm(np.delete(truck_positions, active_idx, axis=0) - depot_position, axis=1)
    max_others_dist = np.max(others_dist) if len(others_dist) > 0 else 0.0
    
    best_score = float('inf')
    best_idx = None
    best_cost_now = 0.0
    best_min_alt = 0.0
    
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # All potential costs for this customer across all trucks
        all_costs = [np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position) for j in range(n_trucks)]
        sorted_costs = sorted(all_costs)
        # Find min alternative (not from active truck)
        alt_min = sorted_costs[0]
        if np.isclose(alt_min, cost_now, atol=1e-8) and len(sorted_costs) > 1:
            alt_min = sorted_costs[1]
        # Find max alternative (not from active truck)
        alt_max = sorted_costs[-1]
        if np.isclose(alt_max, cost_now, atol=1e-8) and len(sorted_costs) > 1:
            alt_max = sorted_costs[-2]
        # Adaptive penalty factor
        pen_base = 1.0 + 0.5 * (alt_min / max(alt_max, 1e-8))
        # Fleet balance penalty: penalize if cost_now exceeds max_others_dist
        if max_others_dist > 0 and cost_now > max_others_dist:
            balance_pen = 0.3 * (cost_now - max_others_dist) / max_others_dist
        else:
            balance_pen = 0.0
        penalty = pen_base + balance_pen
        regret = (cost_now - alt_min) + penalty * max(0, cost_now - alt_max)
        score = regret  # combined score is regret with adjusted penalty
        if score < best_score:
            best_score = score
            best_idx = i
            best_cost_now = cost_now
            best_min_alt = alt_min
    # Wait decision
    if best_score > 0 and best_cost_now > 1.05 * best_min_alt and (max_others_dist == 0 or best_cost_now > max_others_dist):
        return None
    return best_idx