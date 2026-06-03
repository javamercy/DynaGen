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

    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(dist_to_depot)

    best_idx = None
    best_new_max = np.inf
    best_active_rt = np.inf

    for i, cust in enumerate(available_customers):
        active_rt = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if n_trucks > 1:
            other_max = np.max(dist_to_depot[np.arange(n_trucks) != active_idx])
        else:
            other_max = 0.0
        new_max = max(active_rt, other_max)

        if new_max < best_new_max or (new_max == best_new_max and active_rt < best_active_rt):
            best_new_max = new_max
            best_active_rt = active_rt
            best_idx = i

    if n_trucks == 1:
        return best_idx
    else:
        if best_idx is not None and best_new_max <= current_max:
            return best_idx
        else:
            return None