import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_available = len(available_customers)
    # Compute current maximum return time among all trucks
    current_max_return = max(np.linalg.norm(t - depot_position) for t in truck_positions)
    # Compute current return time for the deciding truck
    current_return = np.linalg.norm(current_position - depot_position)
    # Fleet balance factor: higher when current truck is closer to depot than others
    balance = (current_max_return - current_return) / (current_max_return + 1e-8)
    best_score = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # All costs for this customer
        all_costs = [np.linalg.norm(t - cust) + np.linalg.norm(cust - depot_position) for t in truck_positions]
        sorted_costs = sorted(all_costs)
        # min alternative (excluding current truck if it is the best)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[0], cost_now, atol=1e-8):
            min_alt = sorted_costs[1]
        else:
            min_alt = sorted_costs[0]
        # max alternative (excluding current truck if it is the worst)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[-1], cost_now, atol=1e-8):
            max_alt = sorted_costs[-2]
        else:
            max_alt = sorted_costs[-1]
        # Base regret
        regret = (cost_now - min_alt) + (cost_now - max_alt) * 0.5
        # Depot-return urgency penalty: if cost_now exceeds current max return, add penalty
        if cost_now > current_max_return:
            regret += (cost_now - current_max_return) * 2.0
        # Score: we want to minimize regret; if tie, prefer lower cost_now
        score = regret
        if score < best_score or (score == best_score and cost_now < best_cost_now):
            best_score = score
            best_idx = i
            best_cost_now = cost_now
    # Dynamic wait threshold influenced by fleet balance and available customer count
    # Base threshold increases with n_available, and also increases when current truck is relatively close
    wait_threshold = 0.1 * (1.0 + n_available / 10.0) * (1.0 + 2.0 * balance)
    if best_score > 0 and best_score > wait_threshold * best_cost_now:
        return None
    return best_idx