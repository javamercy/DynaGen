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
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    other_trucks = np.delete(truck_positions, active_idx, axis=0)

    base_active = np.linalg.norm(current_position - depot_position)
    dist_curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    detour_active = dist_curr_to_cust + dist_cust_to_depot - base_active

    if n_trucks == 1:
        best_idx = np.argmin(detour_active)
        return int(best_idx)

    other_base = np.linalg.norm(other_trucks - depot_position, axis=1)
    other_to_cust = np.linalg.norm(other_trucks[:, None] - available_customers[None, :], axis=2)
    detour_other = other_to_cust + dist_cust_to_depot[None, :] - other_base[:, None]
    min_other = np.min(detour_other, axis=0)

    regret = detour_active - min_other
    active_dist_to_depot = base_active
    epsilon = 0.05
    threshold = active_dist_to_depot * epsilon

    better_mask = regret <= threshold
    if np.any(better_mask):
        better_indices = np.where(better_mask)[0]
        sorted_indices = better_indices[np.lexsort((detour_active[better_indices], regret[better_indices]))]
        return int(sorted_indices[0])
    else:
        return None