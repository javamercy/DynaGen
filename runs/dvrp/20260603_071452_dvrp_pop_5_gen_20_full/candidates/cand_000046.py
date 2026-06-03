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
    
    depot = depot_position
    depot_dists = np.linalg.norm(available_customers - depot, axis=1)
    
    # distance of active truck to depot
    d_active = np.linalg.norm(current_position - depot)
    # max distance of other trucks to depot
    other_depot_dists = [np.linalg.norm(truck_positions[j] - depot) for j in range(n_trucks) if j != active_idx]
    d_other_max = max(other_depot_dists) if other_depot_dists else 1.0
    
    # dynamic threshold
    beta = 0.3
    threshold = 1.0 + beta * (d_active / (d_other_max + 1e-6))
    
    best_index = None
    best_savings = -np.inf
    best_active_cost = np.inf
    fallback_index = None
    fallback_active_cost = np.inf
    
    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]
        
        if n_trucks == 1:
            if active_cost < best_active_cost:
                best_index = i
                best_active_cost = active_cost
            continue
        
        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            other_costs.append(cost)
        min_other = min(other_costs)
        
        if active_cost <= min_other:
            savings = min_other - active_cost
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_index = i
                best_active_cost = active_cost
        else:
            if active_cost <= threshold * min_other:
                if active_cost < fallback_active_cost:
                    fallback_index = i
                    fallback_active_cost = active_cost
    
    if best_index is not None:
        return best_index
    elif fallback_index is not None:
        return fallback_index
    else:
        return None