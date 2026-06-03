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
    # Identify active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # Current return times for each truck (direct distance to depot)
    current_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_current_return = current_returns[active_idx]
    other_returns = np.delete(current_returns, active_idx)
    max_other_return = np.max(other_returns) if n_trucks > 1 else -np.inf
    
    # For each customer, compute active_service_time = dist(current, customer) + dist(customer, depot)
    dist_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    dist_customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    active_service_times = dist_to_customer + dist_customer_to_depot
    
    # New max return time if active truck serves customer i
    new_max = np.maximum(active_service_times, max_other_return)
    
    # Select customer minimizing new_max; tie-break by minimizing active_service_time
    min_new_max = np.min(new_max)
    candidates = np.where(new_max == min_new_max)[0]
    if len(candidates) == 1:
        return int(candidates[0])
    else:
        # Among candidates, pick one with smallest active_service_time
        best_idx = candidates[np.argmin(active_service_times[candidates])]
        return int(best_idx)