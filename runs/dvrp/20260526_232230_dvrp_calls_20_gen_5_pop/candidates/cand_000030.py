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
    
    all_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    diff = truck_positions - current_position
    idx = np.where(np.all(np.isclose(diff, 0), axis=1))[0]
    if len(idx) == 0:
        idx = [np.argmin(np.linalg.norm(diff, axis=1))]
    current_idx = idx[0]
    
    current_max = np.max(all_depot_dists)
    best_customer = None
    best_ttt = float('inf')
    best_dist_to_current = float('inf')
    alpha = 0.1
    
    for i, cust in enumerate(available_customers):
        new_route = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        updated_dists = all_depot_dists.copy()
        updated_dists[current_idx] = new_route
        candidate_max = np.max(updated_dists)
        candidate_min = np.min(updated_dists)
        penalty = alpha * (candidate_max - candidate_min)
        candidate_ttt = candidate_max + penalty
        dist_to_current = np.linalg.norm(current_position - cust)
        
        if (candidate_ttt < best_ttt) or (candidate_ttt == best_ttt and dist_to_current < best_dist_to_current):
            best_ttt = candidate_ttt
            best_dist_to_current = dist_to_current
            best_customer = i
    
    if best_customer is None:
        return None
    if best_ttt >= current_max:
        return None
    return best_customer