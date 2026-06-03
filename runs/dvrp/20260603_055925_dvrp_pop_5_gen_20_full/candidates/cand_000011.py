import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # Find current truck index
    cur_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    # Compute direct return times for other trucks
    other_return_times = []
    for j in range(n_trucks):
        if j != cur_idx:
            other_return_times.append(np.linalg.norm(truck_positions[j] - depot_position))
    max_other = max(other_return_times) if other_return_times else -np.inf
    # For each customer, compute completion time if served by current truck
    t_cur = np.linalg.norm(available_customers - current_position, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
    if n_trucks == 1:
        # Only one truck: must serve, pick smallest t_cur
        return int(np.argmin(t_cur))
    # Check if best customer does not increase max other
    min_idx = int(np.argmin(t_cur))
    if t_cur[min_idx] <= max_other:
        return min_idx
    else:
        return None