import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Current direct-return times for all trucks
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(dist_to_depot)
    # Identify active truck index (closest to current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    active_current = dist_to_depot[active_idx]
    
    best_customer = None
    best_new_max = np.inf
    best_imbalance = np.inf
    
    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_returns = dist_to_depot.copy()
        new_returns[active_idx] = active_return
        new_max = np.max(new_returns)
        new_min = np.min(new_returns)
        imbalance = new_max - new_min
        
        if active_current < current_max:  # active is not the max
            if new_max < current_max:     # serving improves max
                if new_max < best_new_max or (new_max == best_new_max and imbalance < best_imbalance):
                    best_new_max = new_max
                    best_imbalance = imbalance
                    best_customer = i
        else:  # active is part of max (or tie)
            if new_max < best_new_max or (new_max == best_new_max and imbalance < best_imbalance):
                best_new_max = new_max
                best_imbalance = imbalance
                best_customer = i
    
    if active_current < current_max and best_customer is None:
        return None  # wait
    return best_customer