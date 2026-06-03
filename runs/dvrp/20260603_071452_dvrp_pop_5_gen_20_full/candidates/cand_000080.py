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
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_return = np.max(truck_depot_dists)

    best_idx = None
    best_new_max = np.inf
    fallback_idx = None
    fallback_active_cost = np.inf

    n_avail = len(available_customers)
    if n_avail <= 5:
        base_threshold = 1.2
    else:
        base_threshold = 1.1

    dist_active_to_depot = np.linalg.norm(current_position - depot_position)
    median_depot_dist = np.median(truck_depot_dists)
    if dist_active_to_depot > median_depot_dist:
        base_threshold += 0.05

    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + depot_dists[i]
        other_trucks_depot = np.delete(truck_depot_dists, active_idx)
        other_max = np.max(other_trucks_depot) if len(other_trucks_depot) > 0 else -np.inf
        new_max = max(active_return, other_max)

        if new_max <= current_max_return * 1.2:
            if new_max < best_new_max:
                best_new_max = new_max
                best_idx = i

        min_other = np.inf
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            if cost < min_other:
                min_other = cost
        if min_other == np.inf:
            min_other = 0
        if active_return <= base_threshold * min_other:
            if active_return < fallback_active_cost:
                fallback_active_cost = active_return
                fallback_idx = i

    if best_idx is not None:
        return best_idx
    elif fallback_idx is not None:
        return fallback_idx
    else:
        return None