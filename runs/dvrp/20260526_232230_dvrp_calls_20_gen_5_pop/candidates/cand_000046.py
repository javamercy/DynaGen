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

    # Identify other trucks (excluding current truck)
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_positions = truck_positions[mask]

    # Current return time for this truck if it waits (goes straight to depot)
    current_return = np.linalg.norm(current_position - depot_position)
    if other_positions.shape[0] > 0:
        other_returns = np.linalg.norm(other_positions - depot_position, axis=1)
        max_other_return = np.max(other_returns)
        current_max = max(current_return, max_other_return)
    else:
        # No other trucks -> never wait
        max_other_return = -np.inf
        current_max = current_return

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

    # Wait condition: if there are other trucks and the best new max exceeds current max (with a small epsilon)
    if other_positions.shape[0] > 0 and best_new_max > current_max + 1e-9:
        return None
    return best_idx