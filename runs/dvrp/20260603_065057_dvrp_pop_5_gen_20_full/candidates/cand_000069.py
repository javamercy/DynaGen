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
    # Current times if each truck goes directly to depot
    direct_times = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Identify current truck index (closest to current_position)
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    current_direct = direct_times[current_idx]
    # Current max time
    current_max = np.max(direct_times)
    best_score = np.inf
    best_idx = -1
    best_new_current = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        new_current_time = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Max of other trucks' direct times and new current time
        other_max = np.max(direct_times)  # includes current, but we'll replace
        # Actually compute max excluding current
        if n_trucks == 1:
            new_max = new_current_time
        else:
            other_times = np.delete(direct_times, current_idx)
            new_max = max(np.max(other_times), new_current_time)
        increase = new_max - current_max
        if increase < best_score or (increase == best_score and new_current_time < best_new_current):
            best_score = increase
            best_idx = i
            best_new_current = new_current_time
    return best_idx