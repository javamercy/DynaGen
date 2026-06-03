import numpy as np

def choose_next_customer(current_position: np.ndarray, depot_position: np.ndarray, truck_positions: np.ndarray, available_customers: np.ndarray) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    match = np.all(np.isclose(truck_positions, current_position), axis=1)
    if np.any(match):
        active_idx = np.where(match)[0][0]
        other_max = np.max(truck_to_depot[np.arange(len(truck_to_depot)) != active_idx], initial=-np.inf)
    else:
        other_max = np.max(truck_to_depot)
    candidate_ttt = np.maximum(curr_to_cust + cust_to_depot, other_max)
    min_ttt = np.min(candidate_ttt)
    candidates = np.where(np.isclose(candidate_ttt, min_ttt))[0]
    if len(candidates) > 1:
        best = candidates[np.argmin(curr_to_cust[candidates])]
    else:
        best = candidates[0]
    return int(best)