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

    # Exclude current truck from other trucks
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_positions = truck_positions[mask]
    # Compute distances to depot for other trucks
    if other_positions.shape[0] > 0:
        other_returns = np.linalg.norm(other_positions - depot_position, axis=1)
        max_other_return = np.max(other_returns)
    else:
        max_other_return = -np.inf  # no other trucks

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = np.linalg.norm(current_position - customer)
        d_cust_depot = np.linalg.norm(customer - depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, max_other_return)
        # Prefer smaller new_max, then smaller new_return
        if (new_max < best_new_max) or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    return best_idx