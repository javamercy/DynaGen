import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # find current truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    best_score = -np.inf
    best_customer_idx = -1
    best_this_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # compute regret
        other_costs = []
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            other_costs.append(other_cost)
        if len(other_costs) > 0:
            best_other = min(other_costs)
            regret = max(0, best_other - this_cost)
        else:
            regret = 0
        # compute max cost across all trucks
        all_costs = [np.linalg.norm(truck_positions[k] - cust) + np.linalg.norm(cust - depot_position) for k in range(n_trucks)]
        max_all = max(all_costs)
        score = regret - 0.1 * max_all
        # tie-break by smaller this_cost
        if score > best_score or (score == best_score and this_cost < best_this_cost):
            best_score = score
            best_customer_idx = i
            best_this_cost = this_cost
    return best_customer_idx