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
    # Find index of current truck
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    
    # Single truck case: always serve best customer (min this_cost)
    if n_trucks == 1:
        best_idx = None
        best_cost = np.inf
        for i in range(available_customers.shape[0]):
            cust = available_customers[i]
            cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        return best_idx
    
    alpha = 0.3
    best_score = -np.inf
    best_customer_idx = None
    best_this_cost = None
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_costs = []
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            other_costs.append(other_cost)
        other_min = min(other_costs) if other_costs else 0.0
        regret = other_min - this_cost
        score = regret - alpha * this_cost
        if score > best_score or (score == best_score and (best_this_cost is None or this_cost < best_this_cost)):
            best_score = score
            best_customer_idx = i
            best_this_cost = this_cost
    # Wait if best score is not positive
    if best_score <= 0:
        return None
    return best_customer_idx