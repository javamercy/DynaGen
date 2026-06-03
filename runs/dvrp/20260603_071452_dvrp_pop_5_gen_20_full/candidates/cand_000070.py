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
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # Compute distances
    dist_current_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    active_finish = dist_current_to_cust + dist_cust_to_depot

    # Compute other trucks' current distance to depot (as their estimated finish time)
    dist_truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    max_other_finish = 0.0
    for j in range(n_trucks):
        if j != active_idx:
            max_other_finish = max(max_other_finish, dist_truck_to_depot[j])

    # Projected TTT for each customer
    projected_ttt_cust = np.maximum(active_finish, max_other_finish)

    # For waiting
    wait_finish = np.linalg.norm(current_position - depot_position)
    projected_ttt_wait = max(wait_finish, max_other_finish)

    # Find best customer (min projected TTT, tie-break by smaller active_finish)
    min_proj = np.min(projected_ttt_cust)
    best_candidates = np.where(projected_ttt_cust == min_proj)[0]
    if len(best_candidates) > 1:
        # tie-break: smallest active_finish
        best_candidates = best_candidates[np.argsort(active_finish[best_candidates])]
    best_cust_idx = best_candidates[0]

    if min_proj < projected_ttt_wait:
        return int(best_cust_idx)
    elif min_proj == projected_ttt_wait:
        # prefer to serve customer to make progress
        return int(best_cust_idx)
    else:
        return None