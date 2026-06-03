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
    # find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # precompute other trucks' direct depot distances
    other_depot_dists = []
    for j in range(n_trucks):
        if j == active_idx:
            continue
        other_depot_dists.append(np.linalg.norm(truck_positions[j] - depot_position))
    if other_depot_dists:
        max_other = max(other_depot_dists)
    else:
        max_other = -np.inf  # single truck

    best_index = None
    best_max = np.inf
    best_active = np.inf

    for i, cust in enumerate(available_customers):
        active_new = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        this_max = max(active_new, max_other)
        if this_max < best_max or (this_max == best_max and active_new < best_active):
            best_max = this_max
            best_active = active_new
            best_index = i

    return best_index