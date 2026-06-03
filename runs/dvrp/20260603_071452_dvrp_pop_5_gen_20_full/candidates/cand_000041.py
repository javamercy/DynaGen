import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = truck_positions.shape[0]
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    if not np.allclose(truck_positions[active_idx], current_position):
        raise ValueError("current_position not found in truck_positions")
    
    # Single truck case: always pick cheapest active cost
    if n_trucks == 1:
        active_dists = np.linalg.norm(available_customers - current_position, axis=1)
        depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
        best_idx = np.argmin(active_dists + depot_dists)
        return int(best_idx)
    
    # Current direct return times for all trucks
    current_direct = np.linalg.norm(truck_positions - depot_position, axis=1)
    wait_max = np.max(current_direct)
    
    # Precompute distances for active truck
    active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    active_costs = active_to_cust + cust_to_depot
    
    # Compute new_max for each customer
    new_maxes = np.empty(len(available_customers))
    for i, cost in enumerate(active_costs):
        temp_direct = current_direct.copy()
        temp_direct[active_idx] = cost
        new_maxes[i] = np.max(temp_direct)
    
    best_assign_max = np.min(new_maxes)
    if best_assign_max < wait_max:
        # Pick the customer with smallest new_max; break ties by smallest active cost
        candidates = np.where(new_maxes == best_assign_max)[0]
        best_idx = candidates[np.argmin(active_costs[candidates])]
        return int(best_idx)
    else:
        return None