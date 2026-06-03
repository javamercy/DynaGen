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
    
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    max_depot_dist = np.max(depot_dists) if len(depot_dists) > 0 else 1.0
    if max_depot_dist == 0:
        max_depot_dist = 1.0
    
    best_index = None
    best_savings = -np.inf
    best_active_cost = np.inf
    fallback_index = None
    fallback_active_cost = np.inf
    
    # Adaptive threshold based on number of available customers
    n_avail = len(available_customers)
    base_threshold = 1.2 if n_avail <= 5 else 1.1
    
    for i in range(n_avail):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]
        
        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            other_costs.append(cost)
        
        if n_trucks == 1:
            if active_cost < best_active_cost:
                best_index = i
                best_active_cost = active_cost
            continue
        
        min_other = min(other_costs)
        
        if active_cost <= min_other:
            savings = min_other - active_cost
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_index = i
                best_active_cost = active_cost
        else:
            # Depot pressure: scale threshold inversely with depot distance
            depot_scale = 1 - 0.2 * (depot_dists[i] / max_depot_dist)
            threshold = base_threshold * depot_scale
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