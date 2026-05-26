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
    # compute distances from each truck to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # identify active truck index (closest to current_position)
    dists_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dists_to_current)
    # current return time for active truck (distance from current position to depot)
    current_active_return = np.linalg.norm(current_position - depot_position)
    # current max return time across all trucks
    other_returns = np.delete(truck_depot_dists, active_idx)
    current_max = max(np.max(other_returns), current_active_return) if len(other_returns) > 0 else current_active_return
    
    best_idx = None
    best_new_max = np.inf
    best_own_return = np.inf
    
    for i, cust in enumerate(available_customers):
        new_active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(np.max(other_returns), new_active_return) if len(other_returns) > 0 else new_active_return
        if new_max < best_new_max or (new_max == best_new_max and new_active_return < best_own_return):
            best_new_max = new_max
            best_own_return = new_active_return
            best_idx = i
    
    # if the best new max is strictly greater than current max, wait
    if best_new_max > current_max:
        return None
    return best_idx