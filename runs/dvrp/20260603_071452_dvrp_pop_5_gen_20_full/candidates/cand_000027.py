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
    
    # Precompute depot distances for all customers
    depot_dist = np.linalg.norm(available_customers - depot_position, axis=1)
    
    best = None
    best_savings = -np.inf
    best_cost = np.inf
    
    epsilon_relative = 0.05  # allow 5% overhead
    
    for i in range(len(available_customers)):
        active_cost = np.linalg.norm(current_position - available_customers[i]) + depot_dist[i]
        
        # Compute minimum cost among other trucks
        other_costs = []
        for j in range(n_trucks):
            if j != active_idx:
                cost = np.linalg.norm(truck_positions[j] - available_customers[i]) + depot_dist[i]
                other_costs.append(cost)
        
        min_other = min(other_costs) if len(other_costs) > 0 else np.inf
        
        # Determine if active truck should serve this customer
        if active_cost <= min_other + epsilon_relative * (min_other if np.isfinite(min_other) else 0):
            savings = min_other - active_cost if np.isfinite(min_other) else 0
            # For tie-breaking, prefer lower active_cost
            if (savings > best_savings) or (savings == best_savings and active_cost < best_cost):
                best_savings = savings
                best_cost = active_cost
                best = i
    
    return best