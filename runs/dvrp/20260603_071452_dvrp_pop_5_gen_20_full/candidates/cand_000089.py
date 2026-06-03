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
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # Current max return time assuming all trucks return directly to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(dist_to_depot)
    
    best_index = None
    best_new_max = np.inf
    best_active_time = np.inf
    
    for i, cust in enumerate(available_customers):
        active_time = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(active_time, current_max)
        # Tie-break: prefer smaller active_time
        if new_max < best_new_max or (new_max == best_new_max and active_time < best_active_time):
            best_new_max = new_max
            best_active_time = active_time
            best_index = i
    
    return best_index