import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    n_trucks = len(truck_positions)
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # Compute costs for each customer: active truck cost and all truck costs
    # Cost = dist(truck_pos, customer) + dist(customer, depot)
    delta = available_customers - depot_position  # for depot distance
    depot_dist = np.linalg.norm(delta, axis=1)
    
    best = None  # best customer index
    best_savings = -np.inf
    
    for i in range(len(available_customers)):
        active_cost = np.linalg.norm(current_position - available_customers[i]) + depot_dist[i]
        # compute costs for other trucks
        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - available_customers[i]) + depot_dist[i]
            other_costs.append(cost)
        
        if n_trucks == 1:
            min_other = np.inf
            best_cost = active_cost
        else:
            min_other = min(other_costs)
            best_cost = min(active_cost, min_other)
        
        if active_cost <= min_other:
            savings = min_other - active_cost
            if savings > best_savings or (savings == best_savings and active_cost < best_cost_compare):
                best_savings = savings
                best = i
                best_cost_compare = active_cost
    
    if best is None:
        return None
    return best