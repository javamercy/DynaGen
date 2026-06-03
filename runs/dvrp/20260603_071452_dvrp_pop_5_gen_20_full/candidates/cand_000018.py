import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Compute distances
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    candidate_returns = dist_to_cust + cust_to_depot
    # Find index of current truck
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    idx_current = int(np.argmin(diff))
    all_truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    if len(truck_positions) == 1:
        candidate_ttt = candidate_returns
    else:
        other_max = np.max(np.delete(all_truck_depot_dists, idx_current))
        candidate_ttt = np.maximum(candidate_returns, other_max)
    # Modified selection: among customers with TTT within tolerance of min, pick closest
    min_ttt = np.min(candidate_ttt)
    tol = 1e-9
    mask = candidate_ttt <= min_ttt + tol
    # Among those, pick one with smallest distance to current position
    distances_eligible = dist_to_cust[mask]
    best_local_idx = int(np.argmin(distances_eligible))
    # Find global index
    global_indices = np.where(mask)[0]
    best_idx = int(global_indices[best_local_idx])
    return best_idx