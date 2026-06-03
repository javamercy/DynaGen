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
    
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    other_max = np.max(np.delete(truck_depot_dists, active_idx)) if n_trucks > 1 else 0.0
    
    best_idx = None
    best_new_max = np.inf
    best_active_new = np.inf
    
    for i, cust in enumerate(available_customers):
        active_new = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if n_trucks == 1:
            new_max = active_new
        else:
            new_max = max(active_new, other_max)
        
        if (new_max < best_new_max) or (new_max == best_new_max and active_new < best_active_new):
            best_new_max = new_max
            best_active_new = active_new
            best_idx = i
    
    return best_idx