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
    
    if n_trucks == 1:
        best_idx = None
        best_cost = np.inf
        for i in range(len(available_customers)):
            cost = np.linalg.norm(current_position - available_customers[i]) + depot_dists[i]
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        return best_idx
    
    current_dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(current_dist_to_depot)
    
    n_avail = len(available_customers)
    if n_avail <= 5:
        threshold = 1.2
    else:
        threshold = 1.1
    
    best_idx = None
    best_new_max = np.inf
    best_active_cost = np.inf
    
    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]
        other_max = np.max(np.delete(current_dist_to_depot, active_idx))
        new_max = max(active_cost, other_max)
        if new_max < best_new_max or (new_max == best_new_max and active_cost < best_active_cost):
            best_new_max = new_max
            best_active_cost = active_cost
            best_idx = i
    
    if best_new_max <= current_max * threshold:
        return best_idx
    else:
        return None