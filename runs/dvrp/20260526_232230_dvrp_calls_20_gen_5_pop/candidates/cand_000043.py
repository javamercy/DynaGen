import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # Exclude current truck from others
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_positions = truck_positions[mask]
    if other_positions.shape[0] > 0:
        other_returns = np.linalg.norm(other_positions - depot_position, axis=1)
        max_other_return = np.max(other_returns)
    else:
        # No other trucks, always assign
        max_other_return = -np.inf

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = np.linalg.norm(current_position - customer)
        d_cust_depot = np.linalg.norm(customer - depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, max_other_return)
        if (new_max < best_new_max) or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    # Wait decision: if other trucks exist and best_new_max exceeds max_other_return by threshold
    if other_positions.shape[0] > 0:
        # Compute threshold as 10% of max customer-depot distance
        max_cust_dist = np.max(np.linalg.norm(available_customers - depot_position, axis=1))
        threshold = 0.1 * max_cust_dist
        if best_new_max > max_other_return + threshold:
            return None

    return best_idx