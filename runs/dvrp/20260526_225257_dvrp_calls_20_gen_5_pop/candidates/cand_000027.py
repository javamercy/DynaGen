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
    
    cur_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = cur_to_cust + cust_to_depot
    
    cur_to_depot = np.linalg.norm(current_position - depot_position)
    
    # Identify other trucks
    mask = np.all(np.isclose(truck_positions, current_position, rtol=1e-8, atol=1e-8), axis=1)
    other_mask = ~mask
    
    if not other_mask.any():
        # No other trucks: minimize own return time
        best_idx = np.argmin(own_score)
        return int(best_idx)
    
    other_trucks = truck_positions[other_mask]
    other_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    max_other_return = current_time + np.max(other_dists)
    w = max_other_return
    
    # Serve options
    projected_return = current_time + own_score
    serve_max = np.maximum(projected_return, w)
    best_serve_max = np.min(serve_max)
    best_serve_idx = np.argmin(serve_max)
    
    # Wait option
    wait_max = max(current_time + cur_to_depot, w)
    
    eps = 1e-9
    if wait_max < best_serve_max - eps:
        return None
    else:
        # If multiple customers have same serve_max, break tie with smallest own_score
        candidate_indices = np.where(np.abs(serve_max - best_serve_max) < eps)[0]
        if len(candidate_indices) > 1:
            best_serve_idx = candidate_indices[np.argmin(own_score[candidate_indices])]
        return int(best_serve_idx)