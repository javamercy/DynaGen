import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    candidate_returns = dist_to_cust + cust_to_depot
    all_truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # find index of current truck in truck_positions by nearest position
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    idx_current = int(np.argmin(diff))
    if len(truck_positions) == 1:
        candidate_ttt = candidate_returns
    else:
        other_max = np.max(np.delete(all_truck_depot_dists, idx_current))
        candidate_ttt = np.maximum(candidate_returns, other_max)
    best_idx = int(np.argmin(candidate_ttt))
    return best_idx