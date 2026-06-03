import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        # Only one truck, minimize its total return time
        best_idx = None
        best_cost = float('inf')
        for i, cust in enumerate(available_customers):
            cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        return best_idx

    n_available = len(available_customers)
    best_score = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(all_costs)
        # second best cost (excluding current truck if it's the best)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[0], cost_now, atol=1e-8):
            min_alt = sorted_costs[1]
        else:
            min_alt = sorted_costs[0]
        max_cost = sorted_costs[-1]
        regret_min = cost_now - min_alt
        regret_max = max_cost - cost_now
        score = regret_min - 0.5 * regret_max
        if score < best_score or (score == best_score and cost_now < best_cost_now):
            best_score = score
            best_idx = i
            best_cost_now = cost_now
    # Dynamic wait threshold: increases with number of available customers
    wait_threshold = 0.1 * (1.0 + n_available / 10.0)
    if best_score > wait_threshold * best_cost_now:
        return None
    return best_idx