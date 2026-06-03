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
    # Identify active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.array_equal(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # Compute distances from each truck to depot
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_depot = depot_dists[active_idx]
    # Distances for other trucks
    other_depot = np.delete(depot_dists, active_idx)
    max_other = np.max(other_depot) if len(other_depot) > 0 else -np.inf

    best_index = None
    best_max = np.inf
    best_active_return = np.inf

    for i, cust in enumerate(available_customers):
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if len(other_depot) == 0:
            new_max = active_return
        else:
            new_max = max(active_return, max_other)
        # Prefer lower new_max; tie-break with lower active_return
        if new_max < best_max or (new_max == best_max and active_return < best_active_return):
            best_max = new_max
            best_active_return = active_return
            best_index = i

    return best_index