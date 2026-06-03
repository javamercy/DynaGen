import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_available = len(available_customers)
    n_trucks = len(truck_positions)
    # Identify active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Compute max distance from other trucks to depot
    others_dist = np.linalg.norm(np.delete(truck_positions, active_idx, axis=0) - depot_position, axis=1)
    max_others_dist = np.max(others_dist) if len(others_dist) > 0 else 0.0
    # Adaptive penalty factor
    penalty_factor = 1.0 + n_available / (n_available + 10.0)
    best_score = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        all_costs = [np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position) for j in range(n_trucks)]
        sorted_costs = sorted(all_costs)
        # min alternative (excluding current truck if it's the best)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[0], cost_now, atol=1e-8):
            min_alt = sorted_costs[1]
        else:
            min_alt = sorted_costs[0]
        # max alternative (excluding current truck if it's the worst)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[-1], cost_now, atol=1e-8):
            max_alt = sorted_costs[-2]
        else:
            max_alt = sorted_costs[-1]
        # Fleet balance penalty
        if max_others_dist > 0 and cost_now > max_others_dist:
            balance_pen = 0.2 * (cost_now - max_others_dist)
        else:
            balance_pen = 0.0
        regret_min = cost_now - min_alt
        regret_max = max(0, cost_now - max_alt)
        score = regret_min + penalty_factor * regret_max + balance_pen
        if score < best_score or (np.isclose(score, best_score) and cost_now < best_cost_now):
            best_score = score
            best_idx = i
            best_cost_now = cost_now
    # Dynamic wait threshold
    wait_threshold = 0.15 * (1.0 + n_available / 10.0) * best_cost_now
    if best_score > wait_threshold:
        return None
    return best_idx