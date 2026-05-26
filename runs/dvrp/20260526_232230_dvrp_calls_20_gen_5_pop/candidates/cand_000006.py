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

    n_trucks = len(truck_positions)
    # Identify active truck index
    active_idx = np.where(np.all(np.isclose(truck_positions, current_position), axis=1))[0][0]

    # Distances from current position to each customer and from each customer to depot
    dist_curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # Base distance from current position to depot (for detour calculation)
    base_active = np.linalg.norm(current_position - depot_position)
    detour_active = dist_curr_to_cust + dist_cust_to_depot - base_active

    if n_trucks == 1:
        # Only one truck, must serve; choose smallest detour
        best_idx = np.argmin(detour_active)
        return int(best_idx)

    # Compute detour for other trucks
    other_trucks = np.delete(truck_positions, active_idx, axis=0)
    other_base = np.linalg.norm(other_trucks - depot_position, axis=1)  # (n_other,)
    other_to_cust = np.linalg.norm(other_trucks[:, None] - available_customers[None, :], axis=2)  # (n_other, n_cust)
    detour_other = other_to_cust + dist_cust_to_depot[None, :] - other_base[:, None]  # (n_other, n_cust)
    min_other = np.min(detour_other, axis=0)  # per customer

    regret = detour_active - min_other
    better_mask = regret <= 1e-9  # active truck is best or equal
    if not np.any(better_mask):
        return None

    # Current distances from each truck to depot (for TTT estimation)
    all_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)

    better_indices = np.where(better_mask)[0]
    best_customer = None
    best_ttt = float('inf')
    best_dist_to_current = float('inf')

    for i in better_indices:
        new_route = dist_curr_to_cust[i] + dist_cust_to_depot[i]
        potential_dists = all_depot_dists.copy()
        potential_dists[active_idx] = new_route
        candidate_ttt = np.max(potential_dists)
        dist_to_current = dist_curr_to_cust[i]

        if (candidate_ttt < best_ttt) or (candidate_ttt == best_ttt and dist_to_current < best_dist_to_current):
            best_ttt = candidate_ttt
            best_dist_to_current = dist_to_current
            best_customer = i

    return int(best_customer)