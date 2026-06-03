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
    if n_trucks == 1:
        # always pick customer minimizing this_cost
        best_idx = 0
        best_this_cost = np.inf
        for i in range(available_customers.shape[0]):
            cust = available_customers[i]
            this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            if this_cost < best_this_cost:
                best_this_cost = this_cost
                best_idx = i
        return best_idx

    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    
    # current max return time if no customer assigned
    returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_return = np.max(returns)
    
    best_score = -np.inf
    best_customer_idx = None
    best_this_cost = np.inf
    
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_costs = []
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            other_costs.append(other_cost)
        if len(other_costs) > 0:
            min_other = min(other_costs)
            advantage = min_other - this_cost
        else:
            advantage = 0.0
        
        new_return = max(this_cost, np.max(returns[np.arange(n_trucks) != current_idx]))
        max_increase = max(0.0, new_return - current_max_return)
        
        score = advantage - 0.1 * this_cost - 0.5 * max_increase
        
        if score > best_score or (score == best_score and this_cost < best_this_cost):
            best_score = score
            best_customer_idx = i
            best_this_cost = this_cost
    
    if best_score <= 0.0:
        return None
    return best_customer_idx