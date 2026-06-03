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
    
    # Precompute depot distances for customers
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    
    best_index = None
    best_savings = -np.inf
    best_active_cost = np.inf
    fallback_index = None
    fallback_active_cost = np.inf
    
    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]
        
        # compute best other truck cost
        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            other_costs.append(cost)
        
        if n_trucks == 1:
            # only active truck, always pick cheapest
            if active_cost < best_active_cost:
                best_index = i
                best_active_cost = active_cost
            continue
        
        min_other = min(other_costs)
        
        # Primary rule: active is best (or equal best)
        if active_cost <= min_other:
            savings = min_other - active_cost
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_index = i
                best_active_cost = active_cost
        else:
            # Fallback: within 10% of best other
            if active_cost <= 1.1 * min_other:
                if active_cost < fallback_active_cost:
                    fallback_index = i
                    fallback_active_cost = active_cost
    
    if best_index is not None:
        return best_index
    elif fallback_index is not None:
        return fallback_index
    else:
        return None