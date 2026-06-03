import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # distances
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    candidate_returns = dist_to_cust + cust_to_depot
    all_truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # identify current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    idx_current = int(np.argmin(diff))
    # other trucks' max depot distance
    if len(truck_positions) == 1:
        candidate_ttt = candidate_returns
    else:
        other_max = np.max(np.delete(all_truck_depot_dists, idx_current))
        candidate_ttt = np.maximum(candidate_returns, other_max)
    # best customer
    best_idx = int(np.argmin(candidate_ttt))
    # wait condition: only assign if candidate_return <= current_max
    current_max = np.max(all_truck_depot_dists)
    if candidate_returns[best_idx] <= current_max:
        return best_idx
    else:
        return None