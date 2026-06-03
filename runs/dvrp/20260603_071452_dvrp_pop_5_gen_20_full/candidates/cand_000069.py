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
    # find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # precompute depot distances for each customer
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # precompute current max of direct depot distances
    current_max = max(np.linalg.norm(truck_positions[j] - depot_position) for j in range(n_trucks))
    
    best_max = np.inf
    best_index = None
    best_active_cost = np.inf
    
    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]
        if n_trucks == 1:
            max_cost = active_cost
        else:
            other_costs = [np.linalg.norm(truck_positions[j] - cust) + depot_dists[i] for j in range(n_trucks) if j != active_idx]
            max_other = max(other_costs)
            max_cost = max(active_cost, max_other)
        
        if max_cost < best_max or (max_cost == best_max and active_cost < best_active_cost):
            best_max = max_cost
            best_index = i
            best_active_cost = active_cost
    
    if n_trucks == 1:
        return best_index
    else:
        if best_max <= 1.2 * current_max:
            return best_index
        else:
            return None